"""Combine scored train + val JSONLs, filter to passes_all, re-split into train/val.

Use this AFTER scoring both data/processed/train.jsonl and data/processed/val.jsonl
with `eval.score_dataset`. The intent is to fix the original methodological mistake
of splitting before filtering — by combining the scored pool and re-splitting, both
train and val get the same quality gate.

Usage:
    uv run python -m eval.repartition \\
        --scored-train eval/reports/dataset_scored_<train_ts>.jsonl \\
        --scored-val   eval/reports/dataset_scored_<val_ts>.jsonl

    # Preserve originals by writing to _v2 paths:
    uv run python -m eval.repartition \\
        --scored-train ... --scored-val ... \\
        --train-out data/processed/train_v2.jsonl \\
        --val-out   data/processed/val_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DEFAULT_TRAIN_OUT = Path("data/processed/train.jsonl")
DEFAULT_VAL_OUT = Path("data/processed/val.jsonl")
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_SEED = 42

PAIR_FIELDS = ("prompt", "response", "source", "score", "meta")


def _read_scored(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _strip_to_pair(row: dict) -> dict:
    return {k: row[k] for k in PAIR_FIELDS if k in row}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine + re-split scored JSONLs.")
    parser.add_argument("--scored-train", required=True, help="Scored train JSONL path")
    parser.add_argument("--scored-val", required=True, help="Scored val JSONL path")
    parser.add_argument(
        "--train-out",
        default=str(DEFAULT_TRAIN_OUT),
        help="Output path for new train JSONL",
    )
    parser.add_argument(
        "--val-out",
        default=str(DEFAULT_VAL_OUT),
        help="Output path for new val JSONL",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of clean pool to reserve as val (default 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Shuffle seed for the new split",
    )
    args = parser.parse_args(argv)

    scored_train_path = Path(args.scored_train)
    scored_val_path = Path(args.scored_val)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)

    if not 0.0 < args.val_fraction < 0.5:
        raise SystemExit(f"--val-fraction must be in (0, 0.5); got {args.val_fraction}")

    print(f"Reading scored train: {scored_train_path}")
    train_rows = _read_scored(scored_train_path)
    print(f"  {len(train_rows)} rows")

    print(f"Reading scored val:   {scored_val_path}")
    val_rows = _read_scored(scored_val_path)
    print(f"  {len(val_rows)} rows")

    combined = train_rows + val_rows
    total_in = len(combined)
    judge_failures = sum(1 for r in combined if r.get("eval") is None)
    passes_all = [r for r in combined if r.get("eval") and r["eval"].get("passes_all")]
    print(
        f"\nCombined: {total_in} rows "
        f"(judge_failures={judge_failures}, passes_all={len(passes_all)})"
    )

    if not passes_all:
        raise SystemExit("No rows passed the filter; nothing to write.")

    cleaned = [_strip_to_pair(r) for r in passes_all]

    rng = random.Random(args.seed)
    rng.shuffle(cleaned)

    n_val = max(1, int(round(len(cleaned) * args.val_fraction)))
    val_split = cleaned[:n_val]
    train_split = cleaned[n_val:]

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    with train_out.open("w") as f:
        for r in train_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_out.open("w") as f:
        for r in val_split:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kept_rate = len(passes_all) / total_in if total_in else 0.0
    print("\n=== New split ===")
    print(f"  passes_all kept:    {len(passes_all)} / {total_in}  ({kept_rate:.1%})")
    print(f"  val fraction:       {args.val_fraction:.0%}  (seed={args.seed})")
    print(f"  train: {len(train_split):>5d} rows -> {train_out}")
    print(f"  val:   {len(val_split):>5d} rows -> {val_out}")
    print("\nNext steps:")
    print("  - rm data/processed/train_filtered.jsonl  (superseded)")
    print("  - regenerate data/processed/manifest.json if you use it")

    return 0


if __name__ == "__main__":
    sys.exit(main())
