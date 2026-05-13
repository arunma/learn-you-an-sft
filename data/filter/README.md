# Stage 2b — Filtering pipeline

Read `data/interim/*.pairs.jsonl` (all sources from Stage 2a), apply
filters, write `data/processed/{train,val}.jsonl` + `manifest.json`.

## Filter order (and why it matters)

| # | Filter | Cost / pair | Drops | Why this position |
|---|---|---|---|---|
| 1 | normalize | ~1 ms | 0–1% | Cheap; do it first so all downstream filters see clean text |
| 2 | length + score gate | <1 ms | 30–40% | Cheap; gets us to ~60% of input before any model loads |
| 3 | language (fasttext) | ~1 ms | 5–10% | Cheap-ish; loads a 125 MB model once |
| 4 | dedup (exact + MinHash) | ~2 ms | 20–40% | Medium; MinHash LSH is sub-linear |
| 5 | quality (humor BERT) | ~20 ms | 20–30% | Expensive; only run on survivors of 1–4 |
| 6 | toxicity (Detoxify) | ~20 ms | 5–15% | Expensive; same logic — survivor set only |

Total: 500K input → ~150–200K output pairs in ~30 minutes on CPU.
GPU is much faster but not necessary; this runs once before training.

## What you'll learn

- **Filter ordering as a cost optimisation.** Cheap drops first so the
  expensive BERT models only see survivors. Reordering this is the
  single most common rookie mistake.
- **MinHash + LSH for sub-linear near-duplicate detection.** The
  algorithmic highlight of this stage. Read `dedup.py` carefully —
  it's a beautiful little algorithm.
- **Per-source threshold tuning.** Reddit upvotes and bash.org votes
  aren't comparable raw, but [0,1]-normalised at ingest time gives
  you one threshold across the whole mix.
- **Stable hash-based train/val split.** Hashing the response text
  means a given joke always lands in the same split — re-runs of the
  pipeline don't shuffle val into train.
- **Reproducibility manifest.** `manifest.json` records counts at each
  stage + SHA256 of the train/val files. Future training runs record
  this hash; if it changes, you know your data changed.
- **Lazy imports for optional deps.** `quality.py` and `toxicity.py`
  are imported inside the gate functions, so a user without the
  models can still run the pipeline with `--no-quality --no-toxicity`.

## Run

```bash
cd /Users/arunmanivannan/projects/ai/bash_org

# Heavy deps (only needed for filter, not ingest):
uv pip install ftfy fasttext datasketch tqdm transformers detoxify

# Once: download the fasttext language model (~125 MB)
mkdir -p ~/.cache/fasttext
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -o ~/.cache/fasttext/lid.176.bin

# Run the pipeline
uv run python -m data.filter.pipeline

# Skip the heaviest models for a fast first pass:
uv run python -m data.filter.pipeline --no-quality --no-toxicity
```

## Files

| File | Purpose |
|---|---|
| `normalize.py` | ftfy + URL/email/whitespace cleanup |
| `language.py` | fasttext lid.176 English-only filter |
| `dedup.py` | SHA1 exact + MinHash near-dup (datasketch) |
| `quality.py` | HF text-classification pipeline (humor model) |
| `toxicity.py` | Detoxify wrapper |
| `pipeline.py` | Orchestrator: chains all filters + writes manifest |

## Inspecting the output

After a run, the most informative things to check:

```bash
# Per-stage drop counts
cat data/processed/manifest.json | python -m json.tool

# Per-source survival rate
jq '.source' data/processed/train.jsonl | sort | uniq -c

# Distribution of humor scores (if --no-quality wasn't used)
jq '.meta.humor_score' data/processed/train.jsonl | sort -n | uniq -c

# 20 random surviving pairs — eyeball them
shuf data/processed/train.jsonl | head -20 | jq '.'
```

If a stage drops 100% (after_X count is 0), there's a bug — most
likely a label-name mismatch in `quality.py` if you've swapped to a
different humor classifier.

## Definition of done

- `data/processed/{train,val,manifest}.json` exist
- `manifest.json` shows ~150K+ train pairs after all filters
- Per-stage drop counts look reasonable (no stage drops 100%)
- A spot check of 20 random train pairs reads as actually-funny content
- Re-running the pipeline produces identical SHA256s — i.e. the
  pipeline is deterministic
