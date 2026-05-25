"""Filter pipeline: ingest → normalise → language-check → dedup.

Reads `data/interim/*.pairs.jsonl`, runs the filter chain over a single
DataFrame, writes `data/processed/cleaned.jsonl` and a manifest with
counts + SHA256.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fasttext
import ftfy
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm.auto import tqdm

from prep import INTERIM_DIR, PROCESSED_DIR


MULTI_SPACE = re.compile(r" {2,}")
MULTI_NEWLINE = re.compile(r"\n{3,}")

FASTTEXT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)
FASTTEXT_MODEL_PATH = Path.home() / ".cache" / "fasttext" / "lid.176.bin"
ENGLISH_THRESHOLD = 0.6

NUM_PERM = 128
JACCARD_THRESHOLD = 0.7
NGRAM_SIZE = 5

tqdm.pandas()


_fasttext_model = None


def get_fasttext_model():
    global _fasttext_model
    if _fasttext_model is None:
        if not FASTTEXT_MODEL_PATH.exists():
            raise SystemExit(
                f"fasttext lid.176 not found at {FASTTEXT_MODEL_PATH}.\n\n"
                f"Download once with:\n"
                f"  mkdir -p {FASTTEXT_MODEL_PATH.parent}\n"
                f"  curl -L {FASTTEXT_MODEL_URL} -o {FASTTEXT_MODEL_PATH}\n"
            )
        _fasttext_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
    return _fasttext_model


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def is_english(text: str, threshold: float = ENGLISH_THRESHOLD) -> bool:
    model = get_fasttext_model()
    sanitised = text.replace("\n", " ")[:1000]  # fasttext requires single-line
    if not sanitised.strip():
        return False
    labels, probs = model.predict(sanitised, k=1)
    return labels[0].replace("__label__", "") == "en" and probs[0] >= threshold


def _ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for ng in _ngrams(text):
        m.update(ng.encode("utf-8"))
    return m


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_pairs(interim: Path) -> pd.DataFrame:
    inputs = sorted(interim.glob("*.pairs.jsonl"))
    if not inputs:
        raise SystemExit(f"No interim files in {interim}. Run synthesis first.")
    print(f"Loading from {len(inputs)} sources: {[p.name for p in inputs]}")
    return pd.concat(
        [pd.read_json(path, lines=True) for path in inputs],
        ignore_index=True,
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            prompt=df["prompt"].map(clean_text),
            response=df["response"].map(clean_text),
        )
        .query("prompt != '' and response != ''")
        .reset_index(drop=True)
    )


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["prompt"].progress_map(is_english)
        & df["response"].progress_map(is_english)
    )
    return df[mask].reset_index(drop=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # Exact dedup on response only — templated prompts may legitimately repeat.
    df = df.drop_duplicates(subset="response").reset_index(drop=True)

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    keep: list[bool] = []
    for i, response in enumerate(tqdm(df["response"], desc="dedup")):
        m = _build_minhash(response)
        if list(lsh.query(m)):
            keep.append(False)
            continue
        lsh.insert(str(i), m)
        keep.append(True)
    return df[keep].reset_index(drop=True)


def run_filter(
    interim: Path = INTERIM_DIR,
    processed: Path = PROCESSED_DIR,
) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    def step(df: pd.DataFrame, label: str) -> pd.DataFrame:
        counts[label] = len(df)
        return df

    df = (
        load_pairs(interim)
        .pipe(step, "loaded")
        .pipe(normalize)
        .pipe(step, "after_normalize")
        .pipe(filter_english)
        .pipe(step, "after_language")
        .pipe(dedupe)
        .pipe(step, "after_dedup")
    )

    cleaned_path = processed / "cleaned.jsonl"
    df.to_json(cleaned_path, orient="records", lines=True, force_ascii=False)

    manifest = {
        "counts": counts,
        "cleaned_sha256": file_sha256(cleaned_path),
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"sha256: {manifest['cleaned_sha256']}")
