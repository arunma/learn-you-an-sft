"""Stage 2b orchestrator — run filters and produce processed/.

Reads `data/interim/*.pairs.jsonl` (Stage 1.5 synthesis output), applies
filters in order, writes:

  - data/processed/train.jsonl
  - data/processed/val.jsonl
  - data/processed/manifest.json   (counts at each stage + content hashes)

Filter order:

  1. Normalize  (~1 ms/pair)
  2. Language   (~1 ms/pair) — drop non-English survivors
  3. Dedup      (~2 ms/pair) — exact SHA1 + MinHash LSH near-dup

Quality and toxicity gates are intentionally absent: synth data from
Gemini already meets a quality floor by construction, and a toxicity
filter would gut the casually-profane persona this project targets.

Train/val split is hash-based so re-runs land each pair in the same
split: stable across pipeline invocations, regardless of source order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from ..ingest.schema import Pair, read_jsonl, write_jsonl
from .dedup import dedupe
from .language import filter_english
from .normalize import normalize

DEFAULT_INTERIM = Path(__file__).resolve().parent.parent / "interim"
DEFAULT_PROCESSED = Path(__file__).resolve().parent.parent / "processed"

VAL_FRAC = 0.02


def split_train_val(pairs: list[Pair], val_frac: float) -> tuple[list[Pair], list[Pair]]:
    """Stable train/val split keyed by SHA1 of the response.

    Hashing the response means a given pair always lands in the same
    split regardless of source ordering — re-runs of the pipeline don't
    shuffle val examples into train, which would silently leak val into
    your training run.
    """
    train: list[Pair] = []
    val: list[Pair] = []
    val_max = int(val_frac * (1 << 32))
    for pair in pairs:
        h = int(hashlib.sha1(pair.response.encode()).hexdigest()[:8], 16)
        if h < val_max:
            val.append(pair)
        else:
            train.append(pair)
    return train, val


def file_sha256(path: Path) -> str:
    """Hash a file in 8 KB chunks. Used in the reproducibility manifest."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    ap.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    args = ap.parse_args()

    args.processed.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.interim.glob("*.pairs.jsonl"))
    if not inputs:
        raise SystemExit(
            f"No interim files in {args.interim}. Run Stage 1.5 synthesis first."
        )
    print(f"Loading from {len(inputs)} sources: {[p.name for p in inputs]}")

    all_pairs: list[Pair] = []
    for path in inputs:
        all_pairs.extend(read_jsonl(path))
    counts = {"loaded": len(all_pairs)}
    print(f"Loaded {counts['loaded']:,} pairs total")

    after_normalize = list(normalize(all_pairs))
    counts["after_normalize"] = len(after_normalize)

    after_lang = list(filter_english(tqdm(after_normalize, desc="lang")))
    counts["after_language"] = len(after_lang)

    after_dedup = list(tqdm(dedupe(after_lang), desc="dedup"))
    counts["after_dedup"] = len(after_dedup)

    train, val = split_train_val(after_dedup, VAL_FRAC)
    counts["train"] = len(train)
    counts["val"] = len(val)

    write_jsonl(train, args.processed / "train.jsonl")
    write_jsonl(val, args.processed / "val.jsonl")

    manifest = {
        "counts": counts,
        "train_sha256": file_sha256(args.processed / "train.jsonl"),
        "val_sha256": file_sha256(args.processed / "val.jsonl"),
        "config": {
            "val_frac": VAL_FRAC,
        },
    }
    (args.processed / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n--- Pipeline summary ---")
    for k, v in counts.items():
        print(f"  {k:20s}: {v:>10,}")
    print(f"\ntrain.jsonl SHA256: {manifest['train_sha256']}")
    print(f"val.jsonl   SHA256: {manifest['val_sha256']}")


if __name__ == "__main__":
    main()
