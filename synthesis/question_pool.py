"""Generate a diverse question pool via Gemini Flash for synthesis input."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


MODEL_ID = "gemini-2.5-flash"
DEFAULT_COUNT = 1000
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "interim" / "question_pool.jsonl"
)

MAX_CONCURRENT = 10
PER_CALL_MAX = 200  # Flash output cap; each question ~10 tokens.

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
- Each question is on its own line. No numbering. No bullets. No
  blank lines between. No explanations or category headers.
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

Output exactly {n} questions, one per line, nothing else.
"""


def _parse_questions(text: str) -> list[str]:
    questions: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and line[0].isdigit():
            for sep in (". ", ") ", " - "):
                if sep in line[:5]:
                    line = line.split(sep, 1)[1].strip()
                    break
        if len(line) < 4:
            continue
        questions.append(line)
    return questions


async def generate_chunk(
    client: "genai.Client",
    category: str,
    n: int,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    prompt = CATEGORY_PROMPT_TEMPLATE.format(n=n, category=category)
    async with semaphore:
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=1.0),
            )
        except Exception as e:
            print(
                f"  ERROR chunk category={category[:40]!r}: "
                f"{type(e).__name__}: {e}"
            )
            return []
    return _parse_questions(response.text or "")


async def generate_category(
    client: "genai.Client",
    category: str,
    n: int,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    if n <= PER_CALL_MAX:
        return await generate_chunk(client, category, n, semaphore)

    n_full_batches, remainder = divmod(n, PER_CALL_MAX)
    batch_sizes = [PER_CALL_MAX] * n_full_batches
    if remainder:
        batch_sizes.append(remainder)

    tasks = [
        generate_chunk(client, category, batch_n, semaphore)
        for batch_n in batch_sizes
    ]
    results = await asyncio.gather(*tasks)
    return [q for batch in results for q in batch]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Add it to .env (see .env.example)."
        )

    client = genai.Client(api_key=api_key)

    per_category = max(1, args.count // len(CATEGORIES) + 10)
    print(
        f"Generating ~{per_category} questions per category "
        f"× {len(CATEGORIES)} categories via {MODEL_ID}..."
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        generate_category(client, cat, per_category, semaphore)
        for cat in CATEGORIES
    ]
    results = await asyncio.gather(*tasks)

    seen: set[str] = set()
    pool: list[str] = []
    for cat_questions in results:
        for q in cat_questions:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append(q)

    print(f"Collected {len(pool)} unique questions before trim.")

    random.shuffle(pool)
    pool = pool[:args.count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for q in pool:
            f.write(json.dumps({"question": q}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pool)} questions to {args.output}")
    print("\n--- Sample of 10 ---")
    for q in pool[:10]:
        print(f"  - {q}")


if __name__ == "__main__":
    asyncio.run(main())
