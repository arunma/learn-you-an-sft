"""Generate Monty responses for val prompts and score them with the Haiku judge.

Usage:
    # Full eval against the trained adapter
    uv run python -m eval.run_eval \\
        --val data/processed/val.jsonl \\
        --base Qwen/Qwen2.5-3B-Instruct \\
        --adapter arunma/monty \\
        --concurrency 5

    # Baseline eval — same prompts, base model WITHOUT the adapter
    uv run python -m eval.run_eval \\
        --base Qwen/Qwen2.5-3B-Instruct \\
        --no-adapter

    # Smoke test on 10 prompts
    uv run python -m eval.run_eval --limit 10

Outputs:
    eval/reports/model_eval_<ISO_UTC>.jsonl           per-prompt rows
    eval/reports/model_eval_summary_<ISO_UTC>.json    aggregate stats
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from data.ingest.schema import read_jsonl
from eval.generate import generate_responses, load_monty
from eval.judge import DEFAULT_CONCURRENCY, DEFAULT_JUDGE_MODEL, judge_pairs_async

DEFAULT_VAL = Path("data/processed/val.jsonl")
DEFAULT_REPORTS_DIR = Path("eval/reports")
DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"   # was Qwen/Qwen2.5-3B-Instruct (round 2); update alongside MODEL_ID in train.py when retraining on a new base
DEFAULT_ADAPTER = "arunma/monty"


def _utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    iso = now.isoformat().replace("+00:00", "Z")
    safe = iso.replace(":", "-")
    return safe, iso


async def _run(args: argparse.Namespace) -> int:
    val_path = Path(args.val)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not val_path.exists():
        raise SystemExit(f"Val file not found: {val_path}")

    print(f"Reading val: {val_path}")
    pairs = [(p.prompt, p.response) for p in read_jsonl(val_path)]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"  {len(pairs)} prompts")

    adapter_id = None if args.no_adapter else args.adapter
    print(
        f"Loading model: {args.base}"
        + (f" + adapter {adapter_id}" if adapter_id else " (baseline, no adapter)")
    )
    model, tokenizer = load_monty(args.base, adapter_id)

    print(f"Generating responses (T={args.temperature}, top_p={args.top_p})")
    generations = generate_responses(
        model,
        tokenizer,
        pairs,
        system=args.system,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Judging with {args.judge_model} (concurrency={args.concurrency})")
    judge_inputs = [(g.prompt, g.response) for g in generations]
    judge_results = await judge_pairs_async(
        judge_inputs,
        model=args.judge_model,
        concurrency=args.concurrency,
    )

    rows: list[dict] = []
    for idx, (gen, result) in enumerate(zip(generations, judge_results)):
        row: dict = {
            "index": idx,
            "prompt": gen.prompt,
            "gold": gen.gold,
            "response": gen.response,
        }
        if result.ok:
            row["eval"] = result.score.to_dict()
            row["error"] = None
        else:
            row["eval"] = None
            row["error"] = result.error
        rows.append(row)

    safe_ts, iso_ts = _utc_stamp()
    rows_path = reports_dir / f"model_eval_{safe_ts}.jsonl"
    with rows_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote per-prompt eval -> {rows_path}", flush=True)

    judged = [r for r in rows if r["eval"] is not None]
    failed = [r for r in rows if r["eval"] is None]
    n = len(judged)
    criteria = (
        "on_persona",
        "uses_profanity_appropriately",
        "takes_stance",
        "is_helpful",
        "factual_floor",
    )
    pass_rates = {
        c: round(sum(1 for r in judged if r["eval"][c]) / n, 4) if n else 0.0
        for c in criteria
    }
    passes_all = sum(1 for r in judged if r["eval"]["passes_all"])
    pass_count_hist = Counter(r["eval"]["pass_count"] for r in judged)

    summary = {
        "timestamp_utc": iso_ts,
        "base_model": args.base,
        "adapter": None if args.no_adapter else args.adapter,
        "judge_model": args.judge_model,
        "system_prompt": args.system,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "val_path": str(val_path),
        "prompts": len(pairs),
        "judged": n,
        "judge_failures": len(failed),
        "pass_rates_per_criterion": pass_rates,
        "passes_all_count": passes_all,
        "passes_all_rate": round(passes_all / n, 4) if n else 0.0,
        "pass_count_histogram": {str(k): pass_count_hist.get(k, 0) for k in range(6)},
    }
    summary_path = reports_dir / f"model_eval_summary_{safe_ts}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary         -> {summary_path}", flush=True)

    print("\n=== Model eval summary ===")
    print(f"  base:           {args.base}")
    print(f"  adapter:        {summary['adapter'] or '(none — baseline)'}")
    print(f"  judged:         {n} / {len(pairs)}")
    print(f"  judge failures: {len(failed)}")
    print(f"  passes_all:     {summary['passes_all_rate']:.1%}")
    for c, rate in pass_rates.items():
        print(f"    {c:32s} {rate:.1%}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate + judge model responses on val.jsonl.")
    parser.add_argument("--val", default=str(DEFAULT_VAL), help="Val JSONL path")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base model HF id")
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="LoRA adapter HF id or local path",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Skip the adapter and evaluate the base model only (baseline)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt (default: import SYSTEM from runs.sft_v1_trl.train)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)  # was 256; round 2 saw truncation on longer Monty answers
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of prompts (useful for smoke tests)",
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.system is None:
        from runs.sft_v1_trl.train import SYSTEM as DEFAULT_SYSTEM
        args.system = DEFAULT_SYSTEM
    load_dotenv()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
