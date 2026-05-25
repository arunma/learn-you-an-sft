"""Synthesise (prompt, response) pairs via Gemini 2.5 Pro with the persona prompt.

Resume-safe: re-running over an existing output file skips already-answered prompts.
Each successful pair is flushed immediately so a mid-run crash doesn't lose progress.
"""
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

from data.ingest.schema import Pair, read_jsonl

load_dotenv()


MODEL_ID = "gemini-2.5-pro"
SOURCE = "gemini_synth_v0"
PERSONA_PROMPT_PATH = Path(__file__).resolve().parent / "persona_prompt.md"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "interim" / f"{SOURCE}.pairs.jsonl"
)

MAX_CONCURRENT = 15
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
PER_CALL_TIMEOUT = 60.0
PROGRESS_EVERY = 100

RETRYABLE_PATTERNS = (
    "502", "503", "504", "500", "429",
    "deadline", "timeout", "unavailable",
    "internal error", "bad gateway", "service unavailable",
    "resource exhausted", "overloaded",
)

# Default pilot questions when --questions-file isn't passed.
TEST_QUESTIONS: list[str] = [
    "should I learn Rust?",
    "explain monads in one sentence",
    "what's the capital of Mongolia?",
    "look at this dumb cat picture",
    "ugh today was so long",
    "I'm thinking of quitting",
    "my manager took credit for my work in the all-hands today",
    "I don't want to be alive anymore",
]


def load_persona_prompt() -> str:
    return PERSONA_PROMPT_PATH.read_text(encoding="utf-8")


def load_questions(questions_file: Path | None, cap: int | None) -> list[str]:
    if questions_file is None:
        questions = list(TEST_QUESTIONS)
    else:
        questions = []
        for line in questions_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj["question"])
    if cap is not None:
        questions = questions[:cap]
    return questions


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(pat in msg for pat in RETRYABLE_PATTERNS)


async def generate_one(
    client: "genai.Client",
    config: "types.GenerateContentConfig",
    question: str,
    semaphore: asyncio.Semaphore,
) -> Pair | None:
    async with semaphore:
        response = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=MODEL_ID,
                        contents=question,
                        config=config,
                    ),
                    timeout=PER_CALL_TIMEOUT,
                )
                break
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES:
                    wait_s = RETRY_BACKOFF_BASE ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(wait_s)
                    continue
                print(
                    f"  TIMEOUT {question[:60]!r}: gave up after "
                    f"{MAX_RETRIES + 1} attempts"
                )
                return None
            except Exception as e:
                if attempt < MAX_RETRIES and _is_retryable(e):
                    wait_s = RETRY_BACKOFF_BASE ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(wait_s)
                    continue
                print(
                    f"  ERROR  {question[:60]!r}: "
                    f"{type(e).__name__}: {str(e)[:120]}"
                )
                return None

        if response is None:
            return None

    text = response.text
    if not text or not text.strip():
        finish_reason = (
            response.candidates[0].finish_reason
            if response.candidates else "UNKNOWN"
        )
        print(f"  SAFETY {question[:60]!r}: finish_reason={finish_reason}")
        return None

    return Pair(
        prompt=question,
        response=text.strip(),
        source=SOURCE,
        meta={"model": MODEL_ID},
    )


async def bulk_generate(questions: list[str], output_path: Path) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Add it to .env (see .env.example)."
        )

    client = genai.Client(api_key=api_key)
    persona = load_persona_prompt()
    config = types.GenerateContentConfig(system_instruction=persona)

    already_done: set[str] = set()
    if output_path.exists():
        for pair in read_jsonl(output_path):
            already_done.add(pair.prompt)
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
        f"Generating {len(remaining)} new responses with {MODEL_ID} "
        f"(concurrency={MAX_CONCURRENT}"
        + (f", skipping {skipped} already-done" if skipped else "")
        + ")..."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()
    progress = {"ok": 0, "fail": 0}
    total = len(remaining)

    with output_path.open("a", encoding="utf-8") as out_file:

        async def task(q: str) -> None:
            pair = await generate_one(client, config, q, semaphore)
            async with lock:
                if pair is not None:
                    out_file.write(pair.to_json() + "\n")
                    out_file.flush()
                    progress["ok"] += 1
                else:
                    progress["fail"] += 1
                done = progress["ok"] + progress["fail"]
                if done % PROGRESS_EVERY == 0 or done == total:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", type=Path, default=None)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    questions = load_questions(args.questions_file, args.count)
    asyncio.run(bulk_generate(questions, args.output))


if __name__ == "__main__":
    main()
