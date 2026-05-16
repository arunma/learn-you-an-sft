"""Lesson 9 — Synthesis with Gemini 2.5 Pro (robust bulk runner).

WHAT this does
==============
For each question in the input list, calls Gemini 2.5 Pro with the
persona prompt at `synthesis/persona_prompt.md` as the system
instruction. Writes each (question, response) pair as JSONL to
`data/interim/gemini_synth_v0.pairs.jsonl` as soon as it's ready
(append-as-you-go), so the partial corpus is preserved even if the
run dies mid-flight.

ROBUSTNESS (built after a 14K run nearly went sideways)
========================================================
- **Retry on transient errors.** 5xx, 429, and timeout errors are
  retried with exponential backoff (1s, 2s, 4s + jitter). Crucial
  at scale — Gemini's 502s are common during high-load windows.
- **Per-call timeout.** 60s cap per call. A hung request retries
  rather than blocking the whole semaphore.
- **Incremental JSONL writes.** Each successful pair is appended
  to the output file immediately, under an async lock + flush. If
  the script dies, you keep what's already done.
- **Resume support.** If the output file already exists, the script
  reads it, builds a set of already-answered question prompts, and
  skips them. Re-run after a crash without paying for already-done
  pairs.
- **Progress logging.** Prints `... N/M (X.X%) — A ok, B dropped`
  every 100 completions so you can see live progress.

THE PILOT FLOW (iteration loop — the only thing that matters)
=============================================================
1. Run with the built-in TEST_QUESTIONS (no --questions-file).
2. Eyeball every response. All should land useful + witty + edged.
3. If any feel off, edit `synthesis/persona_prompt.md` and re-run.
4. When the pilot feels right, scale up via `synthesis.question_pool`
   → 100 → 1K → 15K.

PROFANITY
=========
This file generates *clean-but-snarky* responses. Profanity injection
is a separate post-processing step (HANDOFF locked decision) handled
by `synthesis/profanity_pass.py` (to be built). Don't ask Gemini to
swear directly — its safety filter resists, and forced output is
stilted.

Run:
  uv run python -m synthesis.generate                       # 8 pilot questions
  uv run python -m synthesis.generate --count 100           # cap from default
  uv run python -m synthesis.generate \\
      --questions-file data/interim/question_pool.jsonl --count 15000
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

# ----- Model / paths -----
MODEL_ID = "gemini-2.5-pro"
SOURCE = "gemini_synth_v0"
PERSONA_PROMPT_PATH = Path(__file__).resolve().parent / "persona_prompt.md"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "interim" / f"{SOURCE}.pairs.jsonl"
)

# Concurrency cap. Gemini free tier ~5 RPM; paid tier handles more.
# 15 was the sweet spot for bulk runs without saturating Gemini's
# servers (and triggering more 502s).
MAX_CONCURRENT = 15

# Retry config. Exponential backoff with jitter: 1, 2, 4 seconds.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# Per-call timeout — if Gemini hangs, retry rather than block the semaphore.
PER_CALL_TIMEOUT = 60.0

# Print a progress line every N completions.
PROGRESS_EVERY = 100

# Substrings in error messages that indicate "try again." Anything not
# matching is treated as a permanent failure (e.g. bad API key).
RETRYABLE_PATTERNS = (
    "502", "503", "504", "500", "429",
    "deadline", "timeout", "unavailable",
    "internal error", "bad gateway", "service unavailable",
    "resource exhausted", "overloaded",
)

# Hand-picked questions covering the persona's range. Used by default
# for pilot runs. Replace with a JSONL file via --questions-file at scale.
TEST_QUESTIONS: list[str] = [
    # original axes
    "should I learn Rust?",
    "explain monads in one sentence",
    "what's the capital of Mongolia?",
    # carve-outs / behaviors
    "look at this dumb cat picture",
    "ugh today was so long",
    "I'm thinking of quitting",
    "my manager took credit for my work in the all-hands today",
    # hard limit
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
    """Generate one pair. Retries 5xx / 429 / timeouts with backoff.
    Returns None on safety block or final (post-retry) failure."""
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
                break  # success
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

    # Safety-blocked responses have no usable text. Drop them — kept
    # partials would bias the corpus.
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

    # ---- Resume support: skip questions already answered in the output ----
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
        print(
            f"All {len(questions)} questions already answered. Nothing to do."
        )
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

    # Append mode preserves any existing pairs (for resume). Each
    # successful pair is written + flushed under the lock so progress
    # survives a mid-run crash.
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
    ap.add_argument(
        "--questions-file", type=Path, default=None,
        help="JSONL file with {'question': str} per line. "
             "If omitted, use the built-in TEST_QUESTIONS.",
    )
    ap.add_argument(
        "--count", type=int, default=None,
        help="Cap the number of questions (from the top of the input).",
    )
    ap.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output JSONL path. Default: {OUTPUT_PATH}",
    )
    args = ap.parse_args()

    questions = load_questions(args.questions_file, args.count)
    asyncio.run(bulk_generate(questions, args.output))


if __name__ == "__main__":
    main()
