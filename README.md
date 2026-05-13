# bash_org — sarcastic-quip LLM via SFT

Goal: a small instruct model that responds with bash.org-flavoured sarcasm.
Vehicle for learning **LLM internals + ops** end-to-end.

Two parallel training pipelines, same data and same eval:

| | Phase A (`runs/sft_v1_trl/`) | Phase B (`runs/sft_v2_internals/`) |
|---|---|---|
| Goal | Working pipeline, fast | Understand every layer |
| Training loop | `trl.SFTTrainer` | Hand-rolled |
| Adapter | `peft` LoRA | Full fine-tune (no PEFT) |
| Chat template | `tokenizer.apply_chat_template` | Hand-built with explicit special tokens |
| Loss masking | TRL handles it | Explicit `-100` on prompt tokens |
| Sequence packing | TRL handles it | Hand-rolled with cross-example attention isolation |
| Lines of code | ~150 | ~500 |

If both produce comparable judge win-rates on the same eval set, you've
proven you understand what TRL was abstracting.

## Layout

```
bash_org/
├── scrape/                  Stage 1: bash.org Wayback scraper          (shared)
├── data/
│   ├── raw/                   bash.org JSONL, r/Jokes dump, SARC slice
│   └── processed/             Stage 2 output: filtered, deduped, split
├── eval/                    Stage 2.5: prompts, judge rubric, golden set, judge.py  (shared)
├── inference/               Stage 5: hand-rolled generation
└── runs/
    ├── sft_v1_trl/          Stage 4 v1: TRL + LoRA
    └── sft_v2_internals/    Stage 4 v2: hand-rolled
```

## Stage status

See the in-session task list. Each stage's folder has its own README
explaining *why* its choices are what they are — the pedagogy lives in
the READMEs, not in inline comments.

## Dependencies (install once)

```bash
# In this folder, with uv (or pip equivalent)
uv venv
uv pip install httpx beautifulsoup4 lxml tqdm
# Stage 2+:    datasets datasketch ftfy fasttext
# Stage 3+:    transformers tokenizers accelerate
# Stage 4 v1:  trl peft bitsandbytes
# Stage 4 v2:  (nothing extra — just torch + transformers)
# Eval:        anthropic  (or openai)
```
