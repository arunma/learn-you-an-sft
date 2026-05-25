# Fine-tune a 4B model into your own persona

A step-by-step guide for taking `Qwen/Qwen3-4B-Instruct-2507` and training
it to speak in a specific voice you define. Pipeline tested end-to-end on
a single 48 GB GPU; full run lands in a couple of hours of wall-clock time
including synthesis and eval.

Worked example: **Monty**, a foul-mouthed opinionated friend. Replace him
with whoever you want — every step generalises.

---

## What you'll end up with

- A LoRA adapter on top of `Qwen3-4B-Instruct-2507` that produces output
  in the voice you defined.
- A merged HF checkpoint ready for conversion to GGUF.
- Quantised GGUFs (`f16`, `q8_0`, `q4_k_m`) you can drop into LM Studio.
- A reproducible eval scored by Claude Haiku against a five-axis binary
  rubric (Pydantic-validated via Instructor).

---

## Prereqs

**Hardware**

- Training: one GPU with **≥ 48 GB VRAM** (RTX A6000 / RTX 6000 Ada / L40S /
  H100). For 32 GB cards (RTX 5090) drop `BATCH_SIZE` from 8 to 4 in
  `run/train.py`.
- Inference / iteration: any machine with ≥ 16 GB RAM. Apple Silicon runs
  the merged GGUF in LM Studio fine.

**Accounts and API keys**

| Key | Purpose |
|---|---|
| `GEMINI_API_KEY` | Synthesis (Gemini 2.5 Pro + Flash) |
| `ANTHROPIC_API_KEY` | Judge model (Claude Haiku) |
| `HF_TOKEN` | Push adapter + pull base model |

All three load from `.env` (see `.env.example`).

**Software**

```bash
git clone <this-repo>
cd learn-you-an-sft
uv sync
cp .env.example .env
# Fill in the three keys above.
```

---

## TL;DR

```bash
# Edit persona_prompt.md, then:
uv run python -m prep              # questions → answers → filter → score-and-split
uv run python -m run train         # LoRA SFT on data/processed/
uv run python -m run eval          # generate + judge against the trained adapter
uv run python -m run merge         # merge adapter → HF format (then convert to GGUF)
```

That's it. Eight commands total across the whole pipeline, zero flags.
Defaults are tuned for a balanced first pass. **Want to tune something?
Edit the constants at the top of the relevant module.** That's the whole
configuration interface.

The rest of this guide walks through each stage with context, expected
outputs, and the gotchas you'll hit.

---

## Step 1 — Write your persona prompt

The single most important file in the project.

```
persona_prompt.md
```

This is what Gemini reads when it pretends to be your character. Your model
will only be as opinionated as this document tells Gemini to be.

The Monty version is ~350 lines covering:

- **Identity** — who the character is, what they sound like, what they hate
- **Behavioural rules** — when to ask questions back, when to be brief,
  when to take a clear stance, when to drop the voice entirely (crisis prompts)
- **A dozen worked examples** — concrete (prompt, ideal response) pairs

The examples teach more than the rules. If you find yourself writing
"the character should be playful but firm," delete that sentence and
write two examples that *demonstrate* playful + firm instead.

---

## Step 2 — Generate the question pool

```bash
uv run python -m prep questions
```

Asks Gemini Flash for ~1,000 diverse questions spread across ten thematic
categories — tech opinions, life decisions, venting, banter, ask-back
triggers, beginner technical, etc. See the `CATEGORIES` list in
`prep/distill.py`.

**Output:** `data/interim/question_pool.jsonl`, one JSON per line:
`{"question": "<text>"}`. Idempotent — if the file exists, the command
skips with a notice. Delete the file to regenerate.

Want more questions? Change `POOL_DEFAULT_COUNT` in `prep/distill.py`.

**What you should see** — a sample of 10 questions at the end. They should
sound like real things a person would text. If they sound like survey
items, edit `CATEGORY_PROMPT_TEMPLATE` until they don't.

Structured output is enforced via `instructor` + Pydantic (`QuestionList`),
so there's no fragile line-by-line text parsing.

---

## Step 3 — Generate the answers

```bash
uv run python -m prep answers
```

Reads the question pool from Step 2, sends each question to Gemini 2.5 Pro
with `persona_prompt.md` as the system prompt, writes per-question Pair
records as it goes.

**Output:** `data/interim/gemini_synth_v0.pairs.jsonl` — Pair schema
(`{prompt, response, source, score, meta}`). Appended per-line under an
asyncio lock — **crash-safe**. If a run dies, the pairs already written
stay. Re-running skips prompts already in the file.

**Pilot iteration loop:**

1. Edit `persona_prompt.md`.
2. Delete `data/interim/gemini_synth_v0.pairs.jsonl`.
3. Re-run `python -m prep answers`.
4. Inspect the first 10-20 rows. If the voice is off, go to step 1.

A full run against 1,000 questions takes ~5-15 minutes against Gemini Pro
paid tier and costs roughly $0.50-$1.50.

---

## Step 4 — Filter, dedup, language-check

```bash
uv run python -m prep filter
```

Three filters in sequence over a pandas DataFrame.

| Stage | What it drops |
|---|---|
| **Normalise** | Mojibake (`ftfy`), smart quotes, zero-width characters, runaway whitespace. Conservative on meaning — no lowercasing, no punctuation stripping. |
| **Language** | Pairs where either prompt or response isn't English (fasttext `lid.176`). |
| **Dedup** | Exact SHA1 (Stage 1) + near-duplicates via MinHash LSH at Jaccard ≥ 0.7 (Stage 2). |

**Output:**

- `data/processed/cleaned.jsonl` — survivors, Pair schema.

No train/val split yet — that happens in Step 5, after the quality gate.

**fasttext gotcha:** `fasttext` is incompatible with NumPy 2. The
`fasttext-numpy2` fork is already pinned in `pyproject.toml`. The first
run downloads the ~125 MB `lid.176.bin` model to `~/.cache/fasttext/`.

---

## Step 5 — Judge with Haiku, filter, and split

```bash
uv run python -m prep score-and-split
```

One command, three steps:

1. **Judge** every (prompt, response) pair in `cleaned.jsonl` against the
   five-axis rubric using Claude Haiku via Instructor.
2. **Filter** to rows where all five criteria pass (`passes_all`).
3. **Split** the kept pool into `train.jsonl` + `val.jsonl` (shuffled,
   `VAL_FRACTION = 0.05`, `SEED = 42`).

**Output:**

- `data/processed/train.jsonl`, `data/processed/val.jsonl` — the corpus
  training will consume.
- `data/processed/eval_reports/dataset_scored_<ISO>.jsonl` — per-row scoring.
- `data/processed/eval_reports/dataset_summary_<ISO>.json` — aggregate
  pass-rates by criterion + histogram.

### One command for everything in Steps 2-5

```bash
uv run python -m prep
```

Runs `questions → answers → filter → score-and-split` in order. Each stage
is idempotent / resume-safe, so re-running this command is cheap after a
partial completion.

---

## Step 6 — Train

```bash
uv run python -m run train
```

LoRA SFT on Qwen3-4B-Instruct-2507 via TRL. Reads `data/processed/`.
Writes the adapter to `runs/checkpoints/final/`.

**Knobs that matter** (constants at the top of `run/train.py`):

```python
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM = "you are Monty — a foul-mouthed, opinionated friend who curses casually, takes real positions, and actually helps. lowercase by default, no service-speak, no hedging."

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",         # attention
    "gate_proj", "up_proj", "down_proj",            # MLPs
]

EPOCHS = 2
BATCH_SIZE = 8           # 48 GB GPU. Drop to 4 for 32 GB.
GRAD_ACCUMULATION = 2    # Effective batch = 16
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024
```

Two things you'll want to edit for your own persona:

1. **`SYSTEM`** — your persona's system prompt. The **same string** must
   appear in the chat client's system prompt at inference time, or the
   persona won't fire. `run/eval.py` reads this constant; LM Studio and
   any other client need a copy-paste.
2. **`LORA_TARGET_MODULES`** — keeping attention *and* MLP targets is
   what lets the LoRA reach the model's deep identity, not just the
   surface tone. Attention-only is cheaper but won't override Qwen's
   "I am Qwen" self-conception.

`gradient_checkpointing=True` is on. Don't turn it off unless you have
VRAM to burn — activations dominate the memory budget otherwise.

**Optional:** push the adapter to your HF Hub repo when training completes.

```bash
export HF_PUSH_REPO=<your-username>/<your-adapter-name>
uv run python -m run train
```

**What you should see:**

- `trainable params: 32,505,856 || all params: 4,054,232,000 || trainable%: 0.8021`
  near the start. If trainable params are wildly off, your LoRA targets
  probably don't match the model's module names.
- Loss descending from ~2.9 → ~1.5 over the run.
- Per-step eval loss every 50 steps.
- Final adapter at `runs/checkpoints/final/`.

A 1,000-row, 2-epoch run on a 48 GB GPU lands in ~20-30 minutes.

---

## Step 7 — Eval

```bash
uv run python -m run eval
```

Generates against the trained adapter (default: `arunma/monty3` — change
`ADAPTER` in `run/eval.py` for your own), then judges every output with
Claude Haiku via Instructor.

**Output:**

- `runs/eval_reports/model_eval_<ISO>.jsonl` — per-prompt response +
  judge verdict (`PersonaScore`).
- `runs/eval_reports/model_eval_summary_<ISO>.json` — aggregate per-axis
  pass rates + `passes_all` rate + histogram.

**What "good" looks like depends on your persona.** For reference: the
Monty round-3 result was 58.4% `passes_all` on ~600 val prompts (data
ceiling 82.6%). Under 40%? Something is wrong upstream — usually
persona-prompt thinness or LoRA targets missing the MLP.

---

## Step 8 — Merge + LM Studio

```bash
uv run python -m run merge
```

Pulls the adapter + base, merges into a standalone HF checkpoint at
`models/monty-merged/` (~8 GB). Ready for llama.cpp conversion.

Override `ADAPTER_ID` / `BASE_ID` env vars to merge a different adapter:

```bash
ADAPTER_ID=<your-user>/<your-adapter> uv run python -m run merge
```

**Convert HF → GGUF:**

```bash
git clone https://github.com/ggerganov/llama.cpp ~/code/llama.cpp
cd ~/code/llama.cpp && pip install -r requirements.txt

python ~/code/llama.cpp/convert_hf_to_gguf.py \
  /path/to/this/repo/models/monty-merged \
  --outfile models/monty-4b-f16.gguf \
  --outtype f16

# Quantise (smaller + faster on M-series chips):
~/code/llama.cpp/build/bin/llama-quantize \
  models/monty-4b-f16.gguf \
  models/monty-4b-q4_k_m.gguf \
  Q4_K_M
```

**Load in LM Studio:**

1. Copy the GGUF(s) into `~/.lmstudio/models/<username>/<model-name>/`.
2. Open LM Studio, pick the model.
3. **Paste the exact `SYSTEM` string from `run/train.py` into the System
   Prompt field.** The persona fires reliably only when the inference-time
   system prompt matches the training-time one verbatim.

Same applies to any other client (Ollama, llama-cli, your own app).
The system prompt is half of the model.

---

## Iterate

You will not nail this on the first run. The honest workflow is:

1. **Eval** → look at the failures in `runs/eval_reports/model_eval_*.jsonl`.
2. **Categorise** them — profanity drift, abstract waffle, factual errors,
   truncation, persona collapse.
3. **Pull the failed prompts** and hand-write the responses the model
   *should* have given. Append to `data/interim/handcrafted.pairs.jsonl`
   (Pair schema) — they'll go through filter + judge with the rest.
4. **Re-run** `python -m prep filter` then `python -m prep score-and-split`.
5. **Retrain** with `python -m run train`.

Use `jq` to filter the eval JSONL by criterion to get a focused list of
failures of a particular kind:

```bash
# Abstract / philosophical drift
jq -c 'select(.eval and (.eval.rationale | test("abstract|philosophical|pretentious|essay"; "i")))' \
  runs/eval_reports/model_eval_<ISO>.jsonl | head

# Factual errors
jq -c 'select(.eval and .eval.factual_floor == false)' \
  runs/eval_reports/model_eval_<ISO>.jsonl

# All failures, as draft training rows (fill in `response` by hand)
jq -c 'select(.eval and .eval.passes_all == false) | {
  prompt: .prompt,
  response: "TODO",
  source: "handcrafted",
  score: null,
  meta: { failure_mode: "TODO", judge_rationale: .eval.rationale }
}' runs/eval_reports/model_eval_<ISO>.jsonl > data/handcrafted_drafts.jsonl
```

30-50 examples is enough to move the needle meaningfully on a 1K-row
corpus. Larger corpora need proportionally more.

---

## Troubleshooting

**OOM at step ~20 with `shift_logits = … .contiguous()`.**
Logits tensor is the dominant memory cost; with a 152K vocab at bf16 and
batch 16 it's 4.7 GB alone, plus a `.contiguous()` copy. Halve
`BATCH_SIZE`, double `GRAD_ACCUMULATION` in `run/train.py`. Verify
`gradient_checkpointing=True` is set in `SFTConfig` — without it,
activations alone can be 15-20 GB on a 36-layer transformer.

**`fasttext` ImportError on NumPy 2.**
Use `fasttext-numpy2` (community fork). Already pinned in `pyproject.toml`;
re-run `uv sync`.

**Loss flat but `grad_norm` climbing.**
LR too high. Drop `LEARNING_RATE` from 2e-4 to 1e-4. Or your dataset has
conflicting examples — usually visible in val loss diverging from train loss.

**Model says "I am Qwen" when asked who it is.**
LoRA targets are attention-only. The identity claim lives in the MLP
layers — make sure `LORA_TARGET_MODULES` in `run/train.py` includes
`gate_proj`, `up_proj`, `down_proj`. Retrain.

**Persona doesn't fire at inference.**
Inference-time system prompt doesn't match the training-time `SYSTEM`
string. They have to be byte-for-byte identical. LM Studio's default
"You are a helpful AI assistant" silently overrides your persona.

**Responses truncated mid-sentence in eval.**
Bump `GEN_MAX_NEW_TOKENS` in `run/eval.py`. Default is 512.

**Gemini synthesis hangs or 502s a lot.**
Free tier has aggressive rate limits. Drop `ANSWERS_MAX_CONCURRENT` in
`prep/distill.py` to 5, or upgrade to paid tier. Built-in retry handles
transient 502s automatically.

**`python -m prep questions` says "Question pool already exists".**
That's the idempotence check. Delete `data/interim/question_pool.jsonl`
to regenerate.

---

## Where to go from here

- **Persona-prompt iteration** is the single highest-leverage knob. If
  your eval is mediocre, the problem is almost always upstream of training.
- **Try a different base.** Qwen3-4B is a strong default. Llama, Gemma,
  and Mistral 3 work the same way — change `MODEL_ID` in `run/train.py`
  and `BASE_MODEL` in `run/eval.py` and `merge.py`.
- **Increase dataset quality, not size.** 1K high-passing pairs beats
  10K noisy ones.
- **Try longer training with `load_best_model_at_end=True`.** Already
  on; the best checkpoint by `eval_loss` is loaded at the end.
