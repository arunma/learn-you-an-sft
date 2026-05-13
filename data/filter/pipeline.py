"""Stage 2b orchestrator — run all filters and produce processed/.

Reads `data/interim/*.pairs.jsonl`, applies filters in order, writes:

  - data/processed/train.jsonl
  - data/processed/val.jsonl
  - data/processed/manifest.json   (counts at each stage + content hashes)

Filter order is a cost optimisation:

  1. Normalize         (~1 ms/pair)         drops 0–1%
  2. Length+score gate (~<1 ms/pair)        drops 30–40%
  3. Language          (~1 ms/pair)         drops 5–10%
  4. Dedup             (~2 ms/pair)         drops 20–40%
  5. Quality (humor)   (~20 ms/pair)        drops 20–30%
  6. Toxicity          (~20 ms/pair)        drops 5–15%

The expensive BERT-based filters at the end pay off precisely because
they only see the survivors of cheaper filters. Reordering this is the
single most common mistake people make when assembling a filter
pipeline — running quality + toxicity first triples the wall-clock.

Train/val split is hash-based so re-runs land each pair in the same
split: stable across pipeline invocations, regardless of source order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ..ingest.schema import Pair, read_jsonl, write_jsonl
from .normalize import normalize
from .language import filter_english
from .dedup import dedupe

DEFAULT_INTERIM = Path(__file__).resolve().parent.parent / "interim"
DEFAULT_PROCESSED = Path(__file__).resolve().parent.parent / "processed"

# Per-source minimum score gates (after each ingester's [0,1] normalisation).
# rJokes top quartile ≈ 0.5; bash.org top quartile ≈ 0.6;
# SARC scores are noisy so we trust the label and skip the gate.
SCORE_GATES = {
    "rjokes": 0.5,
    "bash_org": 0.6,
    "sarc": 0.0,
    "dadjokes": 0.0,    # no source-side score
}

HUMOR_THRESHOLD = 0.6
TOXICITY_THRESHOLD = 0.7
VAL_FRAC = 0.02


def length_and_score_gate(pairs: Iterable[Pair]) -> Iterable[Pair]:
    """Drop pairs below the per-source score floor. Cheapest filter — runs first."""
    for pair in pairs:
        gate = SCORE_GATES.get(pair.source, 0.0)
        if pair.score is not None and pair.score < gate:
            continue
        yield pair


def quality_gate(pairs: Iterable[Pair]) -> Iterable[Pair]:
    """Lazy-import quality so users without the model can still run the rest."""
    from .quality import annotate_quality
    for pair in annotate_quality(pairs):
        if pair.meta.get("humor_score", 0.0) >= HUMOR_THRESHOLD:
            yield pair


def toxicity_gate(pairs: Iterable[Pair]) -> Iterable[Pair]:
    from .toxicity import annotate_toxicity
    for pair in annotate_toxicity(pairs):
        if pair.meta.get("toxicity", 0.0) <= TOXICITY_THRESHOLD:
            yield pair


def split_train_val(pairs: list[Pair], val_frac: float):
    """Stable train/val split keyed by SHA1 of the response.

    Hashing the response means a given joke always lands in the same
    split regardless of source ordering — re-runs of the pipeline don't
    shuffle val examples into train, which would silently leak val into
    your training run.
    """
    train, val = [], []
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
    ap.add_argument("--no-quality", action="store_true",
                    help="Skip the humor classifier (avoid loading the BERT model).")
    ap.add_argument("--no-toxicity", action="store_true",
                    help="Skip Detoxify (avoid loading another BERT model).")
    args = ap.parse_args()

    args.processed.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.interim.glob("*.pairs.jsonl"))
    if not inputs:
        raise SystemExit(f"No interim files in {args.interim}. Run Stage 2a first.")
    print(f"Loading from {len(inputs)} sources: {[p.name for p in inputs]}")

    all_pairs: list[Pair] = []
    for path in inputs:
        all_pairs.extend(read_jsonl(path))
    counts = {"loaded": len(all_pairs)}
    print(f"Loaded {counts['loaded']:,} pairs total")

    # Materialise after each filter so we can report drop counts.
    after_normalize = list(normalize(all_pairs))
    counts["after_normalize"] = len(after_normalize)

    after_gate = list(length_and_score_gate(after_normalize))
    counts["after_score_gate"] = len(after_gate)

    after_lang = list(filter_english(tqdm(after_gate, desc="lang")))
    counts["after_language"] = len(after_lang)

    after_dedup = list(tqdm(dedupe(after_lang), desc="dedup"))
    counts["after_dedup"] = len(after_dedup)

    survivors = after_dedup
    if not args.no_quality:
        survivors = list(tqdm(quality_gate(survivors), desc="humor"))
        counts["after_humor"] = len(survivors)
    if not args.no_toxicity:
        survivors = list(tqdm(toxicity_gate(survivors), desc="toxicity"))
        counts["after_toxicity"] = len(survivors)

    train, val = split_train_val(survivors, VAL_FRAC)
    counts["train"] = len(train)
    counts["val"] = len(val)

    write_jsonl(train, args.processed / "train.jsonl")
    write_jsonl(val, args.processed / "val.jsonl")

    manifest = {
        "counts": counts,
        "train_sha256": file_sha256(args.processed / "train.jsonl"),
        "val_sha256": file_sha256(args.processed / "val.jsonl"),
        "config": {
            "score_gates": SCORE_GATES,
            "humor_threshold": HUMOR_THRESHOLD,
            "toxicity_threshold": TOXICITY_THRESHOLD,
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
