# learn-you-an-sft

Fine-tuning a small language model into a specific persona — a from-scratch
walk through the supervised fine-tuning pipeline, with real numbers and
honest failure modes.

**Full story:** [Blog post →](https://www.arunma.com/) <!-- TODO: update to the specific post URL post-publish -->

**Trained adapter + GGUFs:** [`arunma/monty3`](https://huggingface.co/arunma/monty3)

## Setup

```bash
uv sync
cp .env.example .env  # fill in HF_TOKEN, ANTHROPIC_API_KEY, GEMINI_API_KEY
```

## What's in here

The blog walks the pipeline end-to-end. Code maps to:

| Stage | Path |
|---|---|
| Persona prompt + Gemini distillation | `synthesis/` |
| Normalize / language filter / dedup | `data/filter/` |
| `Pair` schema (single `(prompt, response)` record) | `data/ingest/` |
| Chat template + assistant-only loss mask | `data/format/` |
| Final training corpus (~11.5k train / 604 val) | `data/processed/` |
| LoRA SFT with TRL | `runs/sft_v1_trl/train.py` |
| Haiku-as-judge rubric + per-prompt scoring | `eval/` |
| Merge adapter into base for GGUF export | `inference/merge_for_gguf.py` |

## Reproduce

End-to-end on a GPU pod (assumes `HF_TOKEN`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` set):

```bash
# 1. Synthesize the training corpus (skip if reusing data/processed/)
uv run python -m synthesis.question_pool --count 15000
uv run python -m synthesis.generate --questions-file data/interim/question_pool.jsonl --count 15000

# 2. Filter (normalize -> language -> dedup -> split)
uv run python -m data.filter.pipeline

# 3. Score with the Haiku judge to quality-rate the synthesized pairs
uv run python -m eval.score_dataset --input data/processed/train.jsonl --sample 99999
uv run python -m eval.score_dataset --input data/processed/val.jsonl --sample 999

# 4. Re-partition: combine scored train + val, filter to passes_all, re-split
uv run python -m eval.repartition \
  --scored-train eval/reports/dataset_scored_<train_ts>.jsonl \
  --scored-val   eval/reports/dataset_scored_<val_ts>.jsonl

# 5. Train (LoRA on Qwen3-4B-Instruct-2507, attention + MLP target modules)
uv run python -m runs.sft_v1_trl.train

# 6. Eval against the trained adapter
uv run python -m eval.run_eval --adapter arunma/monty3

# 7. (Optional) merge into base and convert to GGUF for local LM Studio use
uv run python -m inference.merge_for_gguf
# Then llama.cpp's convert_hf_to_gguf.py + llama-quantize — see the blog
```

## Iteration history

The messy reality — three training rounds, every failed attempt, the
RunPod automation scripts, the original Lesson-by-Lesson tutorial draft —
lives in a separate, private repo. Not public because it's a working
journal, not a curated reference.

This repo is the curated reference. The blog tells the story.

## License

MIT.
