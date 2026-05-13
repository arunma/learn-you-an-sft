# Stage 2a — Source ingestion

Convert each humor source into the canonical `Pair` JSONL schema.
Filtering, dedup, and quality classification come in Stage 2b — this
stage just normalises shapes.

## Files

| File | What it does |
|---|---|
| `schema.py` | The `Pair` dataclass + JSONL read/write helpers. The whole pipeline downstream consumes this one shape. |
| `bashorg.py` | Reads `data/raw/bash_org.jsonl` (Stage 1 output) → strips nicks → builds (prior-lines, last-line) pairs. |
| `rjokes.py` | Reads rJokesData TSVs → splits title/body or Q/A or fakes a template prompt for one-liners. |
| `sarc.py` | Reads a flattened SARC JSONL → keeps only `label=1` (sarcastic) → builds (parent, comment) pairs. |
| `dadjokes.py` | Reads dad-joke dump → uses a rotating manufactured prompt. |

## Download instructions

### rJokesData (~140K useful pairs after Stage 2b filter)

```bash
cd data/raw
git clone --depth=1 https://github.com/orionw/rJokesData.git rjokes
# Verify the TSVs are where we expect them:
ls rjokes/data/preprocessed/
# Expected: dev.tsv  test.tsv  train.tsv
```

### SARC v2 (~50K useful pairs after filter)

The native SARC distribution is two large files (`comments.json.bz2` +
`main/train-balanced.csv.bz2`) that need a join. Two options:

**Option A — pre-flattened HuggingFace mirror** (faster, recommended):

```python
# tools/flatten_sarc.py — run once, produces data/raw/sarc.flat.jsonl
from datasets import load_dataset
import json
ds = load_dataset("CharlemagneXVII/SARC", split="train")  # or any current mirror
with open("data/raw/sarc.flat.jsonl", "w") as f:
    for row in ds:
        f.write(json.dumps({
            "parent": row["parent_comment"],
            "comment": row["comment"],
            "label": int(row["label"]),
            "score": row.get("score"),
        }) + "\n")
```

(Search huggingface.co/datasets for "SARC" before running — the
canonical mirror name has shifted. Verify you're using a sarcasm
corpus, not the unrelated speech-recognition dataset of the same
name.)

**Option B — native SARC join** (do it yourself):

1. Download `comments.json.bz2` and `main/train-balanced.csv.bz2`
   from https://nlp.cs.princeton.edu/SARC/2.0/
2. Build a dict from `comments.json` keyed by comment ID
3. Walk `train-balanced.csv` rows of (parent_id, response_id, label,
   response_id2, label2, ...) and for each (response_id, label), emit
   `{parent: comments[parent_id].text, comment: comments[response_id].text,
   label, score: comments[response_id].score}`.

The native files change format periodically; write your own join
script after inspecting the actual distribution.

### Dad jokes (optional, ~5K pairs)

```bash
cd data/raw
git clone --depth=1 https://github.com/egichSerg/dadjoke-v2.0.git dadjokes_repo
# The repo's exact data path varies — find the JSONL/TXT and copy/symlink:
cp dadjokes_repo/data/jokes.jsonl dadjokes.jsonl   # adjust path as needed
```

If you want to skip dad jokes, the ingester is a no-op when the file
is missing — the rest of the pipeline is unaffected.

### bash.org

Already produced by Stage 1 at `data/raw/bash_org.jsonl`. If you didn't
get permission from the QDB operator, skip this source — the bashorg
ingester is also a no-op when the file is missing.

## Run

```bash
cd /Users/arunmanivannan/projects/ai/bash_org

# Each ingester is independent; run them in any order.
uv run python -m data.ingest.bashorg
uv run python -m data.ingest.rjokes
uv run python -m data.ingest.sarc
uv run python -m data.ingest.dadjokes

# Inspect outputs
wc -l data/interim/*.pairs.jsonl
head -3 data/interim/rjokes.pairs.jsonl | python -m json.tool
```

## What you'll learn

- **Schema-first design.** One canonical shape; every quirk handled
  upstream. The training loop never branches on data source.
- **Score normalisation across heterogeneous sources.** Reddit upvotes,
  bash.org votes, and "no score" (dad jokes) are unified into [0, 1] ∪
  None. Stage 2b applies a single threshold across the mix.
- **Streaming writes** with `write_jsonl` — never load the full corpus
  in memory. SARC alone is millions of comments.
- **Defensive parsing** — every ingester silently skips malformed
  rows rather than crashing the run. A scrape over 500K rows that
  dies on row 412,318 because of one weird Unicode glyph is a bad day.

## Definition of done

- `data/interim/` contains at least two `.pairs.jsonl` files (rjokes
  and one of {bashorg, sarc, dadjokes}).
- `wc -l` on the rjokes output is ≥ 200K (we'll filter aggressively
  in Stage 2b).
- A spot check of 5 random pairs per source: prompt and response read
  as a coherent setup → punchline (or contextual reply) shape.
