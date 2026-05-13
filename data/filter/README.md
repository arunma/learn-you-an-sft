# Stage 2b — Filtering pipeline

Read `data/interim/*.pairs.jsonl` (Stage 1.5 synthesis output), apply
filters, write `data/processed/{train,val}.jsonl` + `manifest.json`.

## Filter order (and why it matters)

| # | Filter | Cost / pair | Drops | Why this position |
|---|---|---|---|---|
| 1 | normalize | ~1 ms | 0–1% | Cheap; do it first so all downstream filters see clean text |
| 2 | language (fasttext) | ~1 ms | 5–10% | Cheap-ish; loads a 125 MB model once |
| 3 | dedup (exact + MinHash) | ~2 ms | 20–40% | Medium; MinHash LSH is sub-linear |

Quality and toxicity gates are intentionally absent: Gemini-synth
pairs meet a quality floor by construction, and a toxicity filter
would gut the casually-profane persona we're after.

Total: 15K input → ~10–13K output pairs in a few minutes on CPU.

## What you'll learn

- **MinHash + LSH for sub-linear near-duplicate detection.** The
  algorithmic highlight of this stage. Read `dedup.py` carefully —
  it's a beautiful little algorithm.
- **Stable hash-based train/val split.** Hashing the response text
  means a given pair always lands in the same split — re-runs of the
  pipeline don't shuffle val into train.
- **Reproducibility manifest.** `manifest.json` records counts at each
  stage + SHA256 of the train/val files. Future training runs record
  this hash; if it changes, you know your data changed.

## Run

```bash
cd /Users/arunmanivannan/projects/ai/learn-you-an-sft

# Filter deps:
uv pip install ftfy fasttext datasketch tqdm

# Once: download the fasttext language model (~125 MB)
mkdir -p ~/.cache/fasttext
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -o ~/.cache/fasttext/lid.176.bin

# Run the pipeline
uv run python -m data.filter.pipeline
```

## Files

| File | Purpose |
|---|---|
| `normalize.py` | ftfy + URL/email/whitespace cleanup |
| `language.py` | fasttext lid.176 English-only filter |
| `dedup.py` | SHA1 exact + MinHash near-dup (datasketch) |
| `pipeline.py` | Orchestrator: chains all filters + writes manifest |

## Inspecting the output

After a run, the most informative things to check:

```bash
# Per-stage drop counts
cat data/processed/manifest.json | python -m json.tool

# Per-source survival rate (all synth for now, but useful once we mix)
jq '.source' data/processed/train.jsonl | sort | uniq -c

# 20 random surviving pairs — eyeball them for persona fit
shuf data/processed/train.jsonl | head -20 | jq '.'
```

If a stage drops 100% (after_X count is 0), there's a bug — most
likely a fasttext model that didn't load (check the cache path) or
a normalize step that emptied every prompt/response.

## Definition of done

- `data/processed/{train,val,manifest}.json` exist
- `manifest.json` shows ≥10K train pairs after all filters
- Per-stage drop counts look reasonable (no stage drops 100%)
- A spot check of 20 random train pairs reads as useful + witty +
  appropriately edgy — i.e. on-persona
- Re-running the pipeline produces identical SHA256s — i.e. the
  pipeline is deterministic
