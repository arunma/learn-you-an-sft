# Fine-tune a 4B model into your own persona

A step-by-step guide for taking `Qwen/Qwen3-4B-Instruct-2507` and training it
to speak in a specific voice you define. Pipeline tested end-to-end on a single
48 GB GPU; full run lands in a couple of hours of wall-clock time including
synthesis and eval.

Worked example in this repo: **Monty**, a foul-mouthed opinionated friend.
Replace him with whoever you want — every step generalises.

---

## What you'll end up with

- A LoRA adapter (~250 MB) on top of `Qwen3-4B-Instruct-2507` that produces
  output in the voice you defined.
- A merged HF checkpoint suitable for conversion to GGUF.
- Quantised GGUFs (`f16`, `q8_0`, `q4_k_m`) you can drop into LM Studio.
- A reproducible eval pipeline scored by Claude Haiku against a five-axis
  binary rubric.

---

## Prereqs

**Hardware**

- Training: one GPU with **≥ 48 GB VRAM** (RTX A6000 / RTX 6000 Ada / L40S /
  H100). 32 GB works if you halve `BATCH_SIZE` (see Step 6).
- Inference / iteration: any machine with ≥ 16 GB RAM. Apple Silicon works
  fine for running the merged GGUF in LM Studio.

**Accounts and API keys**

| Key | Purpose | Where it gets used |
|---|---|---|
| `GEMINI_API_KEY` | Synthesis (Gemini 2.5 Pro + Flash) | Steps 2, 3 |
| `ANTHROPIC_API_KEY` | Judge model (Claude Haiku) | Steps 5, 8 |
| `HF_TOKEN` | Push adapter + pull base model | Steps 7, 9 |

All three load from `.env` (see `.env.example`).

**Software**

```bash
git clone <this-repo>
cd learn-you-an-sft
uv sync
cp .env.example .env
# Fill in the three keys above.
```

`uv` will pin the Python version (3.10) and install everything from
`uv.lock` — including `torch` with the right CUDA build, TRL, peft,
datasets, fasttext, and the Anthropic + Gemini SDKs.

---

## Step 1 — Write your persona prompt

The single most important file in the project.

The persona prompt is what Gemini reads when it pretends to be your character.
Your model is only as opinionated as this document tells Gemini to be.

```
synthesis/persona_prompt.md
```

The Monty version is ~350 lines. It covers:

- **Identity** — who the character is, what they sound like, what they hate.
- **Behavioural rules** — when to ask questions back, when to be brief,
  when to take a clear stance, when to drop the voice entirely (e.g. crisis
  prompts).
- **A dozen worked examples** — concrete (prompt, ideal response) pairs.
  These do more work than the rules.

Write yours. Iterate against the pilot in Step 3 — small persona changes have
outsized downstream effect, so you want fast feedback before you scale up.

**Tip:** the examples teach more than the prose. If you find yourself writing
"the character should be playful but firm," delete that sentence and write
two examples that demonstrate playful + firm instead.

---

## Step 2 — Generate a question pool

`synthesis/question_pool.py` asks Gemini Flash for ~10K diverse questions
spread across ten thematic categories (tech opinions, life decisions, venting,
banter, ask-back triggers, beginner technical, etc. — see the `CATEGORIES`
list at the top of the file).

```bash
uv run python -m synthesis.question_pool --count 15000
```

**Output:** `data/interim/question_pool.jsonl` — one line per question,
schema `{"question": "<text>"}`.

**What you should see:** a sample of 10 questions printed at the end. They
should sound like real things a person would text. If they sound like survey
items, edit `CATEGORY_PROMPT_TEMPLATE` until they don't.

---

## Step 3 — Synthesise (prompt, response) pairs

This is where Gemini Pro reads your persona and impersonates the character
across thousands of questions.

**Pilot run first** (no `--questions-file`, uses 8 built-in test prompts):

```bash
uv run python -m synthesis.generate
```

Inspect every response in `data/interim/gemini_synth_v0.pairs.jsonl`. They
should sound like the character. If even one feels off, **edit
`persona_prompt.md` and re-run** — the synthesis pipeline is resume-safe and
will skip already-answered prompts.

Iterate the pilot until you're happy with all 8 responses. Then scale:

```bash
uv run python -m synthesis.generate \
  --questions-file data/interim/question_pool.jsonl \
  --count 15000
```

**Robustness baked in:**

- Retries 5xx / 429 / timeouts with exponential backoff
- 60-second per-call timeout
- Appends each pair immediately (crash-safe)
- Resumes by reading the existing output and skipping done prompts

**Output:** `data/interim/gemini_synth_v0.pairs.jsonl` — schema:
`{prompt, response, source, score, meta}`.

A 15K-pair run takes ~30-60 minutes against Gemini Pro paid tier.

---

## Step 4 — Filter, dedup, language-check, split

```bash
uv run python -m data.filter.pipeline
```

Runs three filters in sequence and produces train/val splits:

| Stage | What it drops |
|---|---|
| **Normalise** | Mojibake, smart quotes, zero-width characters; URLs → `<URL>`; emails → `<EMAIL>`. Conservative on meaning. |
| **Language** | Pairs where either prompt or response isn't English. Uses fasttext lid.176. |
| **Dedup** | Exact SHA1 duplicates (Stage 1) + near-duplicates via MinHash LSH at Jaccard ≥ 0.7 (Stage 2). |

Then hash-based 98/2 train/val split (stable across re-runs).

**Output:**

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/manifest.json` — counts at each stage + SHA256 of train/val.

**Expected attrition:** 15K → ~12K. Most loss is dedup; some is language
(Gemini occasionally produces a French response — yes, really).

**fasttext gotcha:** `fasttext` is incompatible with NumPy 2. If you hit a
`ValueError: ndarray is not C-contiguous` import, the `fasttext-numpy2` fork
in `pyproject.toml` should be installed — it's a one-line replacement. The
fasttext language-ID model itself (~125 MB) downloads on first run.

---

## Step 5 — Score with Haiku, refilter to a quality floor

Synthesis is noisy. Some pairs are off-persona; some are bland; some are
factually wrong. We score every pair against the same five-axis rubric the
final eval will use, then keep only the pairs that pass all five.

```bash
# Score the train split. --sample 99999 means "score everything."
uv run python -m eval.score_dataset \
  --input data/processed/train.jsonl \
  --sample 99999 \
  --concurrency 5

# Score the val split.
uv run python -m eval.score_dataset \
  --input data/processed/val.jsonl \
  --sample 999 \
  --concurrency 5
```

**Output:**

- `eval/reports/dataset_scored_<ISO>.jsonl` — one row per pair with judge
  verdicts on `on_persona`, `uses_profanity_appropriately`, `takes_stance`,
  `is_helpful`, `factual_floor`, and aggregate `passes_all`.
- `eval/reports/dataset_summary_<ISO>.json` — aggregate stats.

Then combine, filter to `passes_all`, and re-split (so train and val get
the same quality gate):

```bash
uv run python -m eval.repartition \
  --scored-train eval/reports/dataset_scored_<train_ts>.jsonl \
  --scored-val   eval/reports/dataset_scored_<val_ts>.jsonl
```

This **overwrites** `data/processed/train.jsonl` and `val.jsonl` with the
filtered, re-split versions. Use `--train-out` / `--val-out` to preserve
the originals.

**Why bother filtering?** Training on slop teaches the model slop. The
quality of the persona ceiling is set by the quality of the highest-passing
subset of your synthesised pairs.

---

## Step 6 — Configure training

Open `runs/sft_v1_trl/train.py`. The knobs that matter:

```python
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM = (
    "you are Monty — a foul-mouthed, opinionated friend who curses casually, "
    "takes real positions, and actually helps. lowercase by default, "
    "no service-speak, no hedging."
)

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

Two things to update for your own persona:

1. **`SYSTEM`** — your persona's system prompt. This same string must be
   used at inference time, or the persona won't fire. Pick it carefully
   and put it in the chat client's system prompt field later.
2. **`LORA_TARGET_MODULES`** — keeping both attention and MLP targets gives
   the LoRA enough reach to affect the model's deep identity, not just
   surface tone. Attention-only is cheaper but won't override the base's
   "I am Qwen" self-conception.

`gradient_checkpointing=True` is on by default in `SFTConfig`. Don't turn
it off unless you have VRAM to burn — activations dominate the memory
budget without it.

---

## Step 7 — Train

Sync the repo and your `data/processed/` to the GPU box (RunPod, Vast,
a workstation, whatever). Set `HF_TOKEN` and the other API keys in `.env`
there too.

```bash
uv run python -m runs.sft_v1_trl.train
```

**Optional:** push the adapter to your HF Hub repo when training completes.

```bash
export HF_PUSH_REPO=<your-username>/<your-adapter-name>
uv run python -m runs.sft_v1_trl.train
```

**What you should see:**

- A line like `trainable params: 32,505,856 || all params: 4,054,232,000 || trainable%: 0.8021` near the start. If trainable params are wildly off — under 5M or over 100M — the LoRA targets probably mismatch the model's module names.
- Loss descending steadily from ~2.9 → ~1.5 over the run.
- Per-step eval loss every 50 steps (assuming `val.jsonl` exists).
- Final adapter at `runs/sft_v1_trl/checkpoints/final/`.
- A sanity-check generation at the very end on the prompt `"should I learn Rust?"` — first sniff test of whether the persona fired.

A 12K-row, 2-epoch run on a 48 GB GPU lands in ~2 to 2.5 hours.

---

## Step 8 — Eval

Run the trained adapter against the val split and score with Haiku:

```bash
uv run python -m eval.run_eval \
  --adapter <your-username>/<your-adapter-name> \
  --max-new-tokens 512 \
  --concurrency 10
```

**Output:**

- `eval/reports/model_eval_<ISO>.jsonl` — per-prompt response + judge
  verdicts.
- `eval/reports/model_eval_summary_<ISO>.json` — aggregate per-axis pass
  rates.

**What "good" looks like depends on your persona**, but for reference: the
Monty round-3 result was 58.4% `passes_all` on ~600 val prompts (data
ceiling was 82.6%). If you're under 40%, something is wrong upstream —
usually persona prompt thinness or LoRA targets missing the MLP.

**Compare against a baseline.** Run the same eval without `--adapter` to
get the un-fine-tuned base model's score. The delta is the persona's
contribution.

```bash
uv run python -m eval.run_eval --no-adapter --concurrency 10
```

---

## Step 9 — Merge, convert, and load in LM Studio

LM Studio (and llama.cpp generally) needs GGUF, not a LoRA adapter.

```bash
# 1. Pull adapter + base, merge into a standalone HF model
export ADAPTER_ID=<your-username>/<your-adapter-name>
uv run python -m inference.merge_for_gguf
```

**Output:** `models/monty-merged/` — full HF format checkpoint
(~8 GB for a 4B model).

```bash
# 2. Convert HF -> GGUF f16 using llama.cpp's converter
git clone https://github.com/ggerganov/llama.cpp ~/code/llama.cpp
cd ~/code/llama.cpp && pip install -r requirements.txt
python ~/code/llama.cpp/convert_hf_to_gguf.py \
  /path/to/this/repo/models/monty-merged \
  --outfile models/monty-4b-f16.gguf \
  --outtype f16

# 3. Quantise (optional, smaller + faster)
~/code/llama.cpp/build/bin/llama-quantize \
  models/monty-4b-f16.gguf \
  models/monty-4b-q4_k_m.gguf \
  Q4_K_M
```

**Load in LM Studio:**

1. Copy the GGUF(s) into `~/.lmstudio/models/<your-username>/<your-model>/`.
2. Open LM Studio, pick the model.
3. **Paste the exact `SYSTEM` string from `train.py` into the System Prompt
   field.** The persona fires reliably only when the inference-time system
   prompt matches the training-time one verbatim.

Same with any other client — Ollama, llama-cli, your own app. The system
prompt is half of the model.

---

## Step 10 — Iterate

You will not nail this on the first run. The honest workflow is:

1. Eval → look at the failures.
2. Categorise: profanity drift, abstract waffle, factual errors,
   truncation, persona collapse.
3. Pull the failed prompts out and hand-write the responses the model
   should have given.
4. Append those to `data/processed/train.jsonl`.
5. Retrain.

Tooling for the diagnose-and-extract step is in the rubric itself: every
failure row has a `judge_rationale` field explaining what the judge
flagged. Filter the eval JSONL by criterion to get a focused list of
failures of a particular kind:

```bash
# Abstract / philosophical drift
jq -c 'select(.eval and (.eval.rationale | test("abstract|philosophical|pretentious|essay"; "i")))' \
  eval/reports/model_eval_<ISO>.jsonl | head

# Factual errors
jq -c 'select(.eval and .eval.factual_floor == false)' \
  eval/reports/model_eval_<ISO>.jsonl

# All failures, as draft training rows
jq -c 'select(.eval and .eval.passes_all == false) | {
  prompt: .prompt,
  response: "TODO",
  source: "handcrafted",
  score: null,
  meta: { failure_mode: "TODO", judge_rationale: .eval.rationale }
}' eval/reports/model_eval_<ISO>.jsonl > data/round_n_drafts.jsonl
```

Fill in the `"TODO"` response fields by hand. 30-50 examples is enough to
move the needle meaningfully on a 12K-row corpus.

---

## Troubleshooting

**OOM at step ~20 in `compute_loss` with `shift_logits = ... .contiguous()`.**
Logits tensor is the dominant memory line; with a 152K vocab at bf16 and
batch 16 it's 4.7 GB on its own, plus a `.contiguous()` copy. Halve
`BATCH_SIZE`, double `GRAD_ACCUMULATION`. Make sure
`gradient_checkpointing=True` is set in `SFTConfig` — without it, activations
alone are 15-20 GB on a 36-layer transformer.

**`fasttext` ImportError on NumPy 2.**
Use `fasttext-numpy2` (community fork). Already pinned in `pyproject.toml`;
re-run `uv sync` if you're seeing this.

**Loss flat but `grad_norm` climbing.**
LR too high. Drop from 2e-4 → 1e-4. Or your dataset has conflicting
examples — usually visible in val loss diverging from train loss.

**Model says "I am Qwen" when asked who it is.**
LoRA targets are attention-only. The identity claim lives in the MLP
layers — add `gate_proj`, `up_proj`, `down_proj` to
`LORA_TARGET_MODULES` and retrain.

**Persona doesn't fire at inference.**
Inference-time system prompt doesn't match the training-time `SYSTEM`
string. They have to be byte-for-byte identical. LM Studio's default
"You are a helpful AI assistant" silently overrides your persona.

**Responses truncated mid-sentence in eval.**
`--max-new-tokens` too low. Default in `eval/run_eval.py` is 256; bump
to 512 if your persona produces longer responses.

**Gemini synthesis hangs or 502s a lot.**
Free tier has aggressive rate limits. Drop `MAX_CONCURRENT` in
`synthesis/generate.py` to 5, or upgrade to paid tier. Built-in retry
handles transient 502s automatically.

---

## Where to go from here

- **Persona-prompt iteration** is the single highest-leverage knob. If
  your eval is mediocre, the problem is almost always upstream of training.
- **Try a different base.** Qwen3-4B is a strong default — wide multilingual
  coverage, sane chat template, MLP layers that respond well to LoRA. But
  Llama, Gemma, and Mistral 3 work the same way.
- **Increase dataset quality, not size.** 10K high-passing pairs beat 50K
  noisy ones.
- **Try longer training with `load_best_model_at_end=True`.** The
  best checkpoint isn't always the last one. The trainer is already set up
  to keep the best three by `eval_loss`.
