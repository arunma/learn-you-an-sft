"""Distill a persona corpus from Gemini.

Two stages, always run in order:
  generate_questions()  → Gemini Flash writes a diverse question pool
  generate_answers()    → Gemini Pro answers each question in persona voice

Both write JSONL output. Question generation is idempotent (skips if the pool
file already exists). Answer generation is resume-safe (skips prompts already
present in the output file).

Structured output for the question pool goes through `instructor` + Pydantic.
Free-text persona responses use the native google-genai async client.
"""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import instructor
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from prep import INTERIM_DIR, PERSONA_PROMPT_PATH
from prep.schema import Pair

load_dotenv()


POOL_OUTPUT_PATH = INTERIM_DIR / "question_pool.jsonl"
POOL_DEFAULT_COUNT = 1000
FLASH_MODEL = "gemini-2.5-flash"
POOL_MAX_CONCURRENT = 10
POOL_PER_CALL_MAX = 200  # Flash output cap; each question ~10 tokens.

ANSWERS_SOURCE = "gemini_synth_v0"
ANSWERS_OUTPUT_PATH = INTERIM_DIR / f"{ANSWERS_SOURCE}.pairs.jsonl"
PRO_MODEL = "gemini-2.5-pro"
ANSWERS_MAX_CONCURRENT = 15
ANSWERS_MAX_RETRIES = 3
ANSWERS_RETRY_BACKOFF_BASE = 2.0
ANSWERS_PER_CALL_TIMEOUT = 60.0
ANSWERS_PROGRESS_EVERY = 100

RETRYABLE_PATTERNS = (
    "502", "503", "504", "500", "429",
    "deadline", "timeout", "unavailable",
    "internal error", "bad gateway", "service unavailable",
    "resource exhausted", "overloaded",
)

CATEGORIES: list[str] = [
    "tech opinions and recommendations — programming languages, "
    "frameworks, databases, tools, dev practices, build vs buy",
    "factual questions — geography, history, science, language, "
    "everyday facts a curious person might ask",
    "life decisions — career changes, relationships, money, big "
    "moves, whether to do or not do something significant",
    "emotional or ambiguous prompts — someone sharing news, "
    "processing something, telling you about their day in a way "
    "that's not quite a direct question",
    "venting about people — managers, colleagues, family members, "
    "partners, strangers, landlords, institutions; the asker is on "
    "their side and looking for solidarity",
    "banter / vibe-checks / sharing silly things — memes, dumb "
    "observations, sharing a cat picture, expressing exhaustion, "
    "low-stakes back-and-forth",
    "ask-back triggers — vague problem statements like 'I'm thinking "
    "of quitting' or 'something weird is happening with my mum' "
    "that need more context before a useful answer",
    "beginner technical questions — 'what even is X', 'how does Y "
    "actually work', 'why does my code do this' from someone new "
    "to the topic and not embarrassed to ask",
    "opinion questions with no clean answer — tabs vs spaces, "
    "Postgres vs Mongo, Vim vs Emacs, hot takes the asker actually "
    "wants the friend's view on",
    "specific situations needing concrete advice — fix this bug, "
    "draft this email, how should I phrase this to my boss, "
    "what's a reasonable counter-offer",
]

CATEGORY_PROMPT_TEMPLATE = """You are generating a list of realistic
questions that a person might text to a smart, opinionated friend.
The friend will reply later in a casual, witty voice — but your job
is to generate the questions only, not the answers.

Generate {n} questions covering this category:

  {category}

Rules:
- The questions should sound like REAL things people text or ask
  their friends — not survey questions, not "as an AI" framings,
  not interview prompts.
- Mix lengths: some are 3-word questions ("postgres or mongo?"),
  some are 20-word ramblings ("ok so I might have just told my
  manager I'd take on the migration project but I have no idea
  what I'm doing").
- Mix register: some serious, some banter, some genuinely confused.
- Lowercase / casual punctuation is fine, matching how people text.
  Some questions can be properly punctuated too — variety.
- No questions that would require fabricating dangerous advice
  (drug synthesis, weapons, fake medical diagnoses). Casual life
  stuff is fine including emotional / mental health topics.

Return exactly {n} questions.
"""


class QuestionList(BaseModel):
    """Pydantic schema for Flash question-pool output."""
    questions: list[str] = Field(
        description="Realistic questions a person would text a friend",
    )


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Add it to .env (see .env.example)."
        )
    return key


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pat in msg for pat in RETRYABLE_PATTERNS)


def load_persona_prompt() -> str:
    return PERSONA_PROMPT_PATH.read_text(encoding="utf-8")


# ===== Stage 1: question pool (Gemini Flash via instructor) =====

async def _generate_chunk(
    client: "instructor.AsyncInstructor",
    category: str,
    n: int,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    prompt = CATEGORY_PROMPT_TEMPLATE.format(n=n, category=category)
    async with semaphore:
        try:
            result = await client.chat.completions.create(
                model=FLASH_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_model=QuestionList,
                max_retries=2,
            )
        except Exception as e:
            print(
                f"  ERROR chunk category={category[:40]!r}: "
                f"{type(e).__name__}: {e}"
            )
            return []

    return [
        q.strip() for q in result.questions
        if isinstance(q, str) and len(q.strip()) >= 4
    ]


async def _generate_category(
    client: "instructor.AsyncInstructor",
    category: str,
    n: int,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    if n <= POOL_PER_CALL_MAX:
        return await _generate_chunk(client, category, n, semaphore)

    n_batches, remainder = divmod(n, POOL_PER_CALL_MAX)
    sizes = [POOL_PER_CALL_MAX] * n_batches + ([remainder] if remainder else [])
    results = await asyncio.gather(*[
        _generate_chunk(client, category, s, semaphore) for s in sizes
    ])
    return [q for batch in results for q in batch]


async def generate_questions(
    count: int = POOL_DEFAULT_COUNT,
    output_path: Path = POOL_OUTPUT_PATH,
) -> None:
    if output_path.exists():
        print(
            f"Question pool already exists at {output_path}. "
            f"Delete the file to regenerate."
        )
        return

    client = instructor.from_genai(
        genai.Client(api_key=_api_key()),
        use_async=True,
    )

    per_category = max(1, count // len(CATEGORIES) + 10)
    print(
        f"Generating ~{per_category} questions per category "
        f"× {len(CATEGORIES)} categories via {FLASH_MODEL}..."
    )

    semaphore = asyncio.Semaphore(POOL_MAX_CONCURRENT)
    results = await asyncio.gather(*[
        _generate_category(client, cat, per_category, semaphore)
        for cat in CATEGORIES
    ])

    df = (
        pd.DataFrame({"question": [q for batch in results for q in batch]})
        .assign(_key=lambda d: d["question"].str.lower())
        .drop_duplicates("_key")
        .drop(columns="_key")
        .sample(frac=1)
        .head(count)
        .reset_index(drop=True)
    )

    print(f"Collected {len(df)} unique questions after dedup + trim.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"Wrote {len(df)} questions to {output_path}")
    print("\n--- Sample of 10 ---")
    for q in df["question"].head(10):
        print(f"  - {q}")


# ===== Stage 2: persona answers (Gemini Pro, free text) =====

async def _generate_one(
    client: "genai.Client",
    config: "types.GenerateContentConfig",
    question: str,
    semaphore: asyncio.Semaphore,
) -> Pair | None:
    """One Pro call with exponential-backoff retries. Returns None on terminal failure."""
    async with semaphore:
        response = None
        for attempt in range(ANSWERS_MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=PRO_MODEL,
                        contents=question,
                        config=config,
                    ),
                    timeout=ANSWERS_PER_CALL_TIMEOUT,
                )
                break
            except Exception as e:
                is_timeout = isinstance(e, asyncio.TimeoutError)
                retryable = is_timeout or _is_retryable(e)
                if attempt >= ANSWERS_MAX_RETRIES or not retryable:
                    kind = "TIMEOUT" if is_timeout else "ERROR"
                    print(
                        f"  {kind} {question[:60]!r}: "
                        f"{type(e).__name__}: {str(e)[:120]}"
                    )
                    return None
                await asyncio.sleep(
                    ANSWERS_RETRY_BACKOFF_BASE ** attempt + random.uniform(0, 1)
                )

    text = response.text if response else None
    if not text or not text.strip():
        finish_reason = (
            response.candidates[0].finish_reason
            if response and response.candidates
            else "UNKNOWN"
        )
        print(f"  SAFETY {question[:60]!r}: finish_reason={finish_reason}")
        return None

    return Pair(
        prompt=question,
        response=text.strip(),
        source=ANSWERS_SOURCE,
        meta={"model": PRO_MODEL},
    )


async def generate_answers(
    questions_path: Path = POOL_OUTPUT_PATH,
    output_path: Path = ANSWERS_OUTPUT_PATH,
) -> dict:
    if not questions_path.exists():
        raise SystemExit(
            f"Question pool not found at {questions_path}. "
            "Run `python -m prep questions` first."
        )

    questions = pd.read_json(questions_path, lines=True)["question"].tolist()

    client = genai.Client(api_key=_api_key())
    persona = load_persona_prompt()
    config = types.GenerateContentConfig(system_instruction=persona)

    already_done: set[str] = set()
    if output_path.exists():
        already_done = set(pd.read_json(output_path, lines=True)["prompt"])
        if already_done:
            print(
                f"Resume: found {len(already_done)} existing pairs at "
                f"{output_path.name}. Those questions will be skipped."
            )

    remaining = [q for q in questions if q not in already_done]
    if not remaining:
        print(f"All {len(questions)} questions already answered. Nothing to do.")
        return {"ok": 0, "fail": 0}

    skipped = len(questions) - len(remaining)
    print(
        f"Generating {len(remaining)} new responses with {PRO_MODEL} "
        f"(concurrency={ANSWERS_MAX_CONCURRENT}"
        + (f", skipping {skipped} already-done" if skipped else "")
        + ")..."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(ANSWERS_MAX_CONCURRENT)
    lock = asyncio.Lock()
    progress = {"ok": 0, "fail": 0}
    total = len(remaining)

    with output_path.open("a", encoding="utf-8") as out_file:

        async def task(q: str) -> None:
            pair = await _generate_one(client, config, q, semaphore)
            async with lock:
                if pair is not None:
                    out_file.write(pair.to_json() + "\n")
                    out_file.flush()
                    progress["ok"] += 1
                else:
                    progress["fail"] += 1
                done = progress["ok"] + progress["fail"]
                if done % ANSWERS_PROGRESS_EVERY == 0 or done == total:
                    pct = 100.0 * done / total
                    print(
                        f"  ... {done}/{total} ({pct:.1f}%) "
                        f"— {progress['ok']} ok, {progress['fail']} dropped"
                    )

        await asyncio.gather(*(task(q) for q in remaining))

    total_in_file = progress["ok"] + len(already_done)
    print(
        f"\nThis run: {progress['ok']} new pairs written, "
        f"{progress['fail']} dropped (safety / errors)."
    )
    print(f"Total pairs now in {output_path.name}: {total_in_file}")
    return progress
