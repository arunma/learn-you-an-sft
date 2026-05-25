"""Distil a persona corpus from Gemini: Flash for questions, Pro for answers."""
from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path

import instructor
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as atqdm

from prep import INTERIM_DIR, PERSONA_PROMPT_PATH
from prep.schema import Pair

load_dotenv()


POOL_OUTPUT_PATH = INTERIM_DIR / "question_pool.jsonl"
POOL_DEFAULT_COUNT = 1000
FLASH_MODEL = "gemini-2.5-flash"
POOL_MAX_CONCURRENT = 10

ANSWERS_SOURCE = "gemini_synth_v0"
ANSWERS_OUTPUT_PATH = INTERIM_DIR / f"{ANSWERS_SOURCE}.pairs.jsonl"
PRO_MODEL = "gemini-2.5-pro"
ANSWERS_MAX_CONCURRENT = 15
ANSWERS_MAX_RETRIES = 3

CATEGORIES: list[str] = [
    "tech opinions and recommendations — programming languages, frameworks, databases, tools, dev practices, build vs buy",
    "factual questions — geography, history, science, language, everyday facts a curious person might ask",
    "life decisions — career changes, relationships, money, big moves, whether to do or not do something significant",
    "emotional or ambiguous prompts — someone sharing news, processing something, telling you about their day in a way that's not quite a direct question",
    "venting about people — managers, colleagues, family members, partners, strangers, landlords, institutions; the asker is on their side and looking for solidarity",
    "banter / vibe-checks / sharing silly things — memes, dumb observations, sharing a cat picture, expressing exhaustion, low-stakes back-and-forth",
    "ask-back triggers — vague problem statements like 'I'm thinking of quitting' or 'something weird is happening with my mum' that need more context before a useful answer",
    "beginner technical questions — 'what even is X', 'how does Y actually work', 'why does my code do this' from someone new to the topic and not embarrassed to ask",
    "opinion questions with no clean answer — tabs vs spaces, Postgres vs Mongo, Vim vs Emacs, hot takes the asker actually wants the friend's view on",
    "specific situations needing concrete advice — fix this bug, draft this email, how should I phrase this to my boss, what's a reasonable counter-offer",
]

CATEGORY_PROMPT_TEMPLATE = """You are generating a list of realistic questions that a person might text to a smart, opinionated friend. The friend will reply later in a casual, witty voice — but your job is to generate the questions only, not the answers.

Generate {n} questions covering this category:

  {category}

Rules:
- Questions should sound like REAL things people text or ask their friends — not survey questions, not "as an AI" framings, not interview prompts.
- Mix lengths: some are 3-word questions ("postgres or mongo?"), some are 20-word ramblings ("ok so I might have just told my manager I'd take on the migration project but I have no idea what I'm doing").
- Mix register: some serious, some banter, some genuinely confused.
- Lowercase / casual punctuation is fine, matching how people text. Some questions can be properly punctuated too — variety.
- No questions that would require fabricating dangerous advice (drug synthesis, weapons, fake medical diagnoses). Casual life stuff is fine including emotional / mental health topics.

Return exactly {n} questions.
"""


class QuestionList(BaseModel):
    questions: list[str]


class PersonaAnswer(BaseModel):
    response: str


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set — add it to .env")
    return key


def load_persona_prompt() -> str:
    return PERSONA_PROMPT_PATH.read_text(encoding="utf-8")


async def generate_questions(
    count: int = POOL_DEFAULT_COUNT,
    output_path: Path = POOL_OUTPUT_PATH,
) -> None:
    if output_path.exists():
        print(f"{output_path} exists — delete to regenerate")
        return

    client = instructor.from_genai(
        genai.Client(api_key=_api_key()),
        use_async=True,
    )

    per_category = max(1, count // len(CATEGORIES) + 10)
    semaphore = asyncio.Semaphore(POOL_MAX_CONCURRENT)

    async def fetch(category: str) -> list[str]:
        prompt = CATEGORY_PROMPT_TEMPLATE.format(n=per_category, category=category)
        async with semaphore:
            try:
                result = await client.chat.completions.create(
                    model=FLASH_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=QuestionList,
                    max_retries=2,
                )
            except Exception:
                traceback.print_exc()
                return []
        return [
            q.strip() for q in result.questions
            if isinstance(q, str) and len(q.strip()) >= 4
        ]

    results = await asyncio.gather(*[fetch(cat) for cat in CATEGORIES])

    df = (
        pd.DataFrame({"question": [q for batch in results for q in batch]})
        .assign(_key=lambda d: d["question"].str.lower())
        .drop_duplicates("_key")
        .drop(columns="_key")
        .sample(frac=1)
        .head(count)
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"wrote {len(df)} questions to {output_path}")


async def _generate_one(
    client: "instructor.AsyncInstructor",
    persona: str,
    question: str,
    semaphore: asyncio.Semaphore,
) -> Pair | None:
    async with semaphore:
        try:
            result = await client.chat.completions.create(
                model=PRO_MODEL,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "user", "content": question},
                ],
                response_model=PersonaAnswer,
                max_retries=ANSWERS_MAX_RETRIES,
            )
        except Exception:
            traceback.print_exc()
            return None

    text = result.response.strip()
    if not text:
        return None

    return Pair(
        prompt=question,
        response=text,
        source=ANSWERS_SOURCE,
        meta={"model": PRO_MODEL},
    )


async def generate_answers(
    questions_path: Path = POOL_OUTPUT_PATH,
    output_path: Path = ANSWERS_OUTPUT_PATH,
) -> dict:
    if not questions_path.exists():
        raise SystemExit(f"missing {questions_path} — run `python -m prep questions` first")

    questions = pd.read_json(questions_path, lines=True)["question"].tolist()

    client = instructor.from_genai(
        genai.Client(api_key=_api_key()),
        use_async=True,
    )
    persona = load_persona_prompt()

    already_done: set[str] = set()
    if output_path.exists():
        already_done = set(pd.read_json(output_path, lines=True)["prompt"])

    remaining = [q for q in questions if q not in already_done]
    if not remaining:
        return {"ok": 0, "fail": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(ANSWERS_MAX_CONCURRENT)

    with output_path.open("a", encoding="utf-8") as out_file:

        async def task(q: str) -> Pair | None:
            pair = await _generate_one(client, persona, q, semaphore)
            if pair is not None:
                out_file.write(pair.to_json() + "\n")
                out_file.flush()
            return pair

        results = await atqdm.gather(
            *(task(q) for q in remaining), desc="answers"
        )

    ok = sum(1 for r in results if r is not None)
    fail = len(results) - ok
    print(f"wrote {ok} new pairs ({fail} dropped) — total in {output_path.name}: {ok + len(already_done)}")
    return {"ok": ok, "fail": fail}
