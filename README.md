# learn-you-an-sft

Small instruct-tuned LLM that responds like a witty friend who roasts
you — useful answer + sarcastic delivery + casual profanity when it
fits. Built end-to-end as a vehicle for learning **LLM internals + ops**
the hard way.

Sibling project: `learn-you-an-hf-llm` (TinyStories pretraining). This
project picks up where that one ends — taking a pre-trained base model
and fine-tuning it on (prompt, response) pairs synthesised by Gemini
2.5 Pro, then judging the result with Claude Haiku 4.5.

Full context, locked decisions, and the journey of pivots: see
[HANDOFF.md](HANDOFF.md).

## Two parallel training pipelines, same data, same eval

|  | Phase A (`runs/sft_v1_trl/`) | Phase B (`runs/sft_v2_internals/`) |
|---|---|---|
| Goal | Working pipeline, fast | Understand every layer |
| Training loop | `trl.SFTTrainer` | Hand-rolled |
| Adapter | `peft` LoRA | Full fine-tune (no PEFT) |
| Chat template | `tokenizer.apply_chat_template` | Hand-built with explicit special tokens |
| Loss masking | TRL handles it | Explicit `-100` on prompt tokens |
| Sequence packing | TRL handles it | Hand-rolled with cross-example attention isolation |
| Lines of code | ~150 | ~500 |
| Base model | Qwen2.5-0.5B (chosen for native tool-call chat template) | Qwen2.5-0.5B |

If both produce comparable judge win-rates on the same eval set, the
abstractions TRL hides have been demystified.

## Layout

```
learn-you-an-sft/
├── synthesis/                Stage 1.5: Gemini-distilled (prompt, response) pairs
├── data/
│   ├── ingest/               Canonical Pair schema (schema.py only after the synth pivot)
│   ├── filter/               Stage 2b: normalize → language → dedup → train/val + manifest
│   ├── interim/              Pair JSONL from synthesis runs (not committed)
│   └── processed/            train.jsonl, val.jsonl, manifest.json (not committed)
├── eval/                     Stage 2.5: locked prompt set + Claude-Haiku judge
├── runs/
│   ├── sft_v1_trl/           Stage 4 v1: TRL + LoRA
│   └── sft_v2_internals/     Stage 4 v2: hand-rolled
└── inference/                Stage 5: hand-rolled generation + chat CLI
```

## Setup

```bash
# In this folder
uv venv
uv pip install -e .
uv lock                       # pin to a uv.lock; commit that

# Once: fasttext language ID model (~125 MB)
mkdir -p ~/.cache/fasttext
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -o ~/.cache/fasttext/lid.176.bin

# Secrets — set in your shell or ~/.zshrc
export GEMINI_API_KEY=...     # Stage 1.5 synthesis (Gemini 2.5 Pro)
export ANTHROPIC_API_KEY=...  # Stage 2.5 judge (Claude Haiku 4.5)
```

Dependencies live in [pyproject.toml](pyproject.toml). `bitsandbytes`
is CUDA-only and is excluded on macOS so local installs don't fail.

## Stage status

See the in-session task list and `HANDOFF.md` for the full plan. Each
stage's folder has its own README explaining *why* its choices are
what they are — the pedagogy lives in the READMEs, not in inline
comments.
