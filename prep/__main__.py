"""Data prep pipeline: questions → answers → filter → score-and-split.

  uv run python -m prep                       # all stages in order
  uv run python -m prep questions             # Gemini Flash → question pool
  uv run python -m prep answers               # Gemini Pro → persona Q&A pairs
  uv run python -m prep filter                # normalise / language / dedup
  uv run python -m prep score-and-split       # judge → filter passes_all → split train/val
"""
from __future__ import annotations

import argparse
import asyncio

from prep.distill import generate_answers, generate_questions
from prep.filter import run_filter
from prep.score import run_score_and_split


STAGES = [
    ("questions",       lambda: asyncio.run(generate_questions())),
    ("answers",         lambda: asyncio.run(generate_answers())),
    ("filter",          run_filter),
    ("score-and-split", run_score_and_split),
]
ALL_STAGE_NAMES = tuple(name for name, _ in STAGES)


def main() -> None:
    parser = argparse.ArgumentParser(prog="prep", description=__doc__.splitlines()[0])
    parser.add_argument(
        "stage",
        nargs="?",
        choices=ALL_STAGE_NAMES,
        help="Run a single stage. Omit to run all stages in order.",
    )
    args = parser.parse_args()

    selected = [(args.stage, dict(STAGES)[args.stage])] if args.stage else STAGES
    for name, fn in selected:
        print(f"\n=== Stage: {name} ===")
        fn()


if __name__ == "__main__":
    main()
