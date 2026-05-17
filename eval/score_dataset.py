"""Score a random sample of training pairs with the Haiku judge.

Usage:
    uv run python -m eval.score_dataset \\
        --input data/processed/train.jsonl \\
        --sample 5000 \\
        --seed 42 \\
        --concurrency 5

    # Also write a filtered JSONL containing only passes_all rows:
    uv run python -m eval.score_dataset \\
        --input data/processed/train.jsonl \\
        --sample 5000 \\
        --write-filtered data/processed/train_filtered.jsonl

Outputs (always):
    eval/reports/dataset_scored_<ISO_UTC>.jsonl   one record per sampled row + eval sub-object
    eval/reports/dataset_summary_<ISO_UTC>.json   aggregate stats

Output (optional via --write-filtered <path>):
    <path>   filtered subset of the SAMPLED rows where passes_all == True
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from data.ingest.schema import read_jsonl
from eval.judge import DEFAULT_CONCURRENCY, DEFAULT_JUDGE_MODEL, judge_pairs_async

DEFAULT_INPUT = Path("data/processed/train.jsonl")
DEFAULT_REPORTS_DIR = Path("eval/reports")
DEFAULT_SAMPLE = 5000
DEFAULT_SEED = 42


def _utc_stamp() -> tuple[str, str]:
    """Return (filename_safe, iso) UTC timestamps."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    iso = now.isoformat().replace("+00:00", "Z")
    safe = iso.replace(":", "-")
    return safe, iso


def _sample_pairs(input_path: Path, sample: int, seed: int) -> list[dict]:
    rows = [asdict(p) for p in read_jsonl(input_path)]
    if not rows:
        raise SystemExit(f"No rows read from {input_path}")
    if sample >= len(rows):
        print(
            f"  Sample size {sample} >= dataset size {len(rows)}; using all rows.",
            flush=True,
        )
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample)


def _summarize(
    scored_rows: list[dict],
    *,
    sample: int,
    judge_model: str,
    iso_ts: str,
    input_path: Path,
) -> dict:
    judged = [r for r in scored_rows if r["eval"] is not None]
    failed = [r for r in scored_rows if r["eval"] is None]
    n_judged = len(judged)

    criteria = (
        "on_persona",
        "uses_profanity_appropriately",
        "takes_stance",
        "is_helpful",
        "factual_floor",
    )
    pass_rates = {
        c: round(sum(1 for r in judged if r["eval"][c]) / n_judged, 4) if n_judged else 0.0
        for c in criteria
    }
    passes_all = sum(1 for r in judged if r["eval"]["passes_all"])
    pass_count_hist = Counter(r["eval"]["pass_count"] for r in judged)

    return {
        "timestamp_utc": iso_ts,
        "input_path": str(input_path),
        "judge_model": judge_model,
        "sample_requested": sample,
        "sample_actual": len(scored_rows),
        "judged": n_judged,
        "judge_failures": len(failed),
        "pass_rates_per_criterion": pass_rates,
        "passes_all_count": passes_all,
        "passes_all_rate": round(passes_all / n_judged, 4) if n_judged else 0.0,
        "pass_count_histogram": {str(k): pass_count_hist.get(k, 0) for k in range(6)},
        "judge_failure_examples": [
            {"index": r["_sample_index"], "error": r["_error"]} for r in failed[:5]
        ],
    }


async def _run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}", flush=True)
    rows = _sample_pairs(input_path, args.sample, args.seed)
    print(f"Sampled {len(rows)} rows (seed={args.seed})", flush=True)

    pairs = [(r["prompt"], r["response"]) for r in rows]
    print(
        f"Judging with {args.judge_model} "
        f"(concurrency={args.concurrency}, retries={args.max_retries})",
        flush=True,
    )
    results = await judge_pairs_async(
        pairs,
        model=args.judge_model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )

    safe_ts, iso_ts = _utc_stamp()
    scored_rows: list[dict] = []
    for i, (row, result) in enumerate(zip(rows, results)):
        merged = dict(row)
        merged["_sample_index"] = i
        if result.ok:
            merged["eval"] = result.score.to_dict()
            merged["_error"] = None
        else:
            merged["eval"] = None
            merged["_error"] = result.error
        scored_rows.append(merged)

    scored_path = reports_dir / f"dataset_scored_{safe_ts}.jsonl"
    with scored_path.open("w") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote scored rows -> {scored_path}", flush=True)

    summary = _summarize(
        scored_rows,
        sample=args.sample,
        judge_model=args.judge_model,
        iso_ts=iso_ts,
        input_path=input_path,
    )
    summary_path = reports_dir / f"dataset_summary_{safe_ts}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary    -> {summary_path}", flush=True)

    print("\n=== Summary ===")
    print(f"  judged:           {summary['judged']} / {summary['sample_actual']}")
    print(f"  judge failures:   {summary['judge_failures']}")
    print(f"  passes_all rate:  {summary['passes_all_rate']:.1%}")
    for criterion, rate in summary["pass_rates_per_criterion"].items():
        print(f"    {criterion:32s} {rate:.1%}")

    if args.write_filtered:
        filtered_path = Path(args.write_filtered)
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with filtered_path.open("w") as f:
            for row in scored_rows:
                ev = row.get("eval")
                if ev is None or not ev["passes_all"]:
                    continue
                clean = {
                    k: row[k]
                    for k in ("prompt", "response", "source", "score", "meta")
                    if k in row
                }
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
                kept += 1
        print(
            f"\nFiltered (passes_all) -> {filtered_path}  ({kept} rows kept)",
            flush=True,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score training pairs with Haiku.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL path")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="Random sample size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Output directory")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Anthropic model id")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent judge calls")
    parser.add_argument("--max-retries", type=int, default=3, help="Per-pair retry budget")
    parser.add_argument(
        "--write-filtered",
        default=None,
        help="Optional output path for the filtered (passes_all) subset",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
