+++
title = "Fine-Tuning a 4B Model Into Your Own Persona, Part 2: Training, Eval, Ship"
slug = "fine-tuning-a-4b-model-into-your-own-persona-part-2-training-eval-ship"
date = 2026-05-26T18:30:00+00:00
lastmod = 2026-05-26T18:30:00+00:00
categories = ["llm", "buildx"]
tags = ["llm", "sft", "fine-tuning", "lora", "qwen", "pytorch", "trl", "peft", "llama-cpp", "tutorial"]
aliases = ["/p/fine-tuning-a-4b-model-into-your-own-persona-part-2-training-eval-ship/"]
+++

[Part 1](https://arunma.com/fine-tuning-a-4b-model-into-your-own-persona-part-1-the-data/) ended with `train.jsonl`, `val.jsonl`, and a `passes_all` percentage — your **data ceiling**, the best score you can hope for from a model trained on that corpus.

This post is the rest: LoRA SFT on Qwen3-4B, evaluating against the same five-axis judge, iterating with hand-corrections, and shipping to GGUF for local inference.

Code: [github.com/arunma/learn-you-an-sft](https://github.com/arunma/learn-you-an-sft). Same repo, same commands.

You need a GPU now. 48 GB VRAM is the comfortable target — A6000, RTX 6000 Ada, L40S, H100. **32 GB works** (RTX 5090) if you drop `BATCH_SIZE` from 8 to 4 in `run/train.py` and bump `GRAD_ACCUMULATION` from 2 to 4. The previous post has the autopsy on why 32 GB OOMs by default — short version: Qwen3-4B's 152K vocab makes the cross-entropy logits tensor expensive.

---

## Step 5 — Train (without OOMing)

```bash
uv run python -m run train
```

LoRA SFT on Qwen3-4B-Instruct via TRL. Defaults are tuned for a 48 GB GPU. Constants at the top of `run/train.py` are the configuration interface — no flags, no YAML, just edit Python.

Two knobs matter most:

```python
SYSTEM = "you are Monty — a foul-mouthed, opinionated friend who curses casually, takes real positions, and actually helps. lowercase by default, no service-speak, no hedging."

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",         # MLPs
]
```

The `SYSTEM` string must be **byte-for-byte identical** at training and inference time. Paste it into LM Studio's system prompt field later. If they disagree, the persona doesn't fire. LM Studio's default "you are a helpful AI assistant" silently overrides your prompt if you forget.

The `LORA_TARGET_MODULES` list is the difference between "character in a costume" and "character". Attention-only LoRA gets you a model that *says* the right things but writes 400-token bulleted action plans like a polite instruct model. **MLP layers are where the *shape* of the response lives** — sentence length, format, pacing, whether it asks a question back. Train both. The previous post has the receipts on this.

The TRL config worth noting:

```python
config = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,        # effective batch = 16
    learning_rate=1e-4,
    max_length=1024,
    bf16=True,
    gradient_checkpointing=True,          # don't turn this off
    assistant_only_loss=True,             # only learn on assistant tokens
)
```

`gradient_checkpointing=True` is doing real work — without it, activations dominate the memory budget on a 36-layer transformer and you'll OOM on the cross-entropy logits tensor around step ~20. (Previous post has the math.) `assistant_only_loss=True` means the loss only fires on assistant tokens, not on the system prompt or user prompt, so the model isn't getting penalised for "predicting" things it didn't have to generate.

**Sample output:**

```
$ uv run python -m run train

Loading Qwen/Qwen3-4B-Instruct-2507
train: 622  val: 32
trainable params: 32,505,856 || all params: 4,054,232,000 || trainable%: 0.8021
2 epochs x ~38 steps/epoch (effective batch=16)
{'loss': 2.8534, 'grad_norm': 12.34, 'learning_rate': 5.0e-05, 'epoch': 0.03}
{'loss': 2.4127, 'grad_norm':  8.92, 'learning_rate': 1.0e-04, 'epoch': 0.06}
{'loss': 2.0193, 'grad_norm':  5.87, 'learning_rate': 9.8e-05, 'epoch': 0.13}
...
{'eval_loss': 1.7234, 'eval_runtime': 23.4, 'epoch': 0.5}
{'loss': 1.6481, 'grad_norm':  3.21, 'learning_rate': 5.2e-05, 'epoch': 1.0}
{'eval_loss': 1.5891, 'eval_runtime': 22.9, 'epoch': 1.0}
...
{'loss': 1.4623, 'grad_norm':  2.84, 'learning_rate': 1.1e-05, 'epoch': 1.9}
{'eval_loss': 1.5234, 'eval_runtime': 23.1, 'epoch': 2.0}
{'train_runtime': 1842.3, 'train_samples_per_second': 6.75, 'epoch': 2.0}

Saved adapter to runs/checkpoints/final
```

A ~600-row, 2-epoch run on a 48 GB GPU lands in about 25 minutes.

Watch three things in that output:

- **`trainable% = 0.8021`.** If this is wildly off, your `LORA_TARGET_MODULES` list doesn't match the model's module names. Different base models name them differently. Check first, train second.
- **Training loss descending from ~2.9 → ~1.5.** Steady decline, no flat plateaus. Plateau means LR too low; spiky chaos means LR too high.
- **`eval_loss` tracking `loss` within 0.1-0.2.** Diverging upward = overfitting. Drop epochs from 2 to 1, or add more data, or both.

`load_best_model_at_end=True` is set in the config (omitted from the snippet above), so the checkpoint with the lowest `eval_loss` is what gets saved at the end. You don't have to babysit the curve.

Pushing the trained adapter to HF Hub when training completes is one env var:

```bash
export HF_PUSH_REPO=<your-user>/<your-adapter>
uv run python -m run train
```

---

## Step 6 — Eval. Actually eval.

Train loss going down tells you the optimizer's working. **It does not tell you whether you've built the right thing.**

```bash
uv run python -m run eval
```

Same five-axis `PersonaScore` judge as Part 1, but now judging the trained model's responses to held-out val prompts. The script loads base + adapter via `peft`, generates a response per val prompt with sampling (T=0.7, top_p=0.9), then sends each (prompt, response) to Haiku.

The generation loop is intentionally boring:

```python
for prompt, gold in tqdm(pairs, desc="generating"):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    with torch.no_grad():
        gen = model.generate(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        gen[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True,
    ).strip()
```

One prompt at a time, sampled with the same `SYSTEM` string the model was trained with. The judge phase is the same async Instructor call as Part 1's gate, just with the model's response instead of Gemini's.

**Sample output:**

```
$ uv run python -m run eval

Loading Qwen/Qwen3-4B-Instruct-2507 + adapter arunma/monty3
generating: 100%|████████████████████████| 32/32 [01:34<00:00,  2.93s/it]
judging: 100%|███████████████████████████| 32/32 [00:11<00:00,  2.79it/s]
runs/eval_reports/model_eval_20260526T161523Z.jsonl
runs/eval_reports/model_eval_summary_20260526T161523Z.json
passes_all: 58.4%
  on_persona: 77.8%
  uses_profanity_appropriately: 74.0%
  takes_stance: 96.5%
  is_helpful: 90.7%
  factual_floor: 86.4%
```

What "good" looks like depends on your character. For reference: **Monty round 3 hit 58.4% `passes_all`** on ~600 val prompts (the run above is for a smaller 32-prompt val set as an illustration). The data ceiling from Part 1 was 82.6%. The gap between data ceiling and model eval is the bit fine-tuning didn't capture — usually shape/form (sentence length, format, pacing) rather than vocabulary.

The per-axis breakdown is where the diagnostic information lives:

| If you see... | The fix is... |
|---|---|
| `takes_stance` 95%, `on_persona` 65% | Opinionated but doesn't sound like the character. Add MLP targets if you haven't, or tighten the persona prompt. |
| `is_helpful` 85%, `factual_floor` 70% | Confidently wrong about facts. Upgrade the base model or hand-correct domain-specific failures. |
| Everything in the 60s | Persona prompt is too thin. Go back to the persona prompt. |
| `passes_all` < 40% | Something's wrong upstream — usually data ceiling. Re-run Part 1 with a stronger persona prompt. |

Each pattern points at a different fix.

---

## Step 7 — Iterate

You won't nail it on the first run. Nobody does. The honest loop:

1. Pull the failures out of `runs/eval_reports/model_eval_*.jsonl`. `jq` is your friend.
2. Categorise them. The repeat patterns will surprise you — usually one or two failure modes account for 60-70% of misses.
3. **Hand-write the responses the model should have given.** Append to `data/interim/handcrafted.pairs.jsonl` in Pair schema.
4. Re-run `prep filter`, `prep score-and-split`, `run train`.

The handcrafted rows go through the same pipeline as everything else — same filter, same judge, same gate. Nothing special. **30-50 handcrafted examples is enough to meaningfully move the needle on a 1K-row corpus.** They're disproportionately effective because they're targeted: each one closes a specific failure mode.

`jq` recipes for slicing the eval JSONL:

```bash
# Abstract / philosophical drift
jq -c 'select(.eval and (.eval.rationale | test("abstract|philosophical|essay"; "i")))' \
  runs/eval_reports/model_eval_<stamp>.jsonl | head
```

```json
{"index": 7, "prompt": "should I learn Rust?", "response": "Rust is a powerful systems programming language with a steep learning curve. There are several factors to consider...", "eval": {"on_persona": false, "rationale": "Reads like an essay; no Monty voice or stance.", ...}}
{"index": 14, "prompt": "is therapy worth it", "response": "Therapy can be a valuable tool for many people. Here are some factors...", "eval": {"on_persona": false, "rationale": "Therapist-speak, generic; doesn't pick a side.", ...}}
```

```bash
# Factual errors only
jq -c 'select(.eval and .eval.factual_floor == false)' \
  runs/eval_reports/model_eval_<stamp>.jsonl
```

```bash
# All failures, as draft training rows (you fill in `response`)
jq -c 'select(.eval and .eval.passes_all == false) | {
  prompt: .prompt,
  response: "TODO",
  source: "handcrafted",
  score: null,
  meta: { failure_mode: "TODO", judge_rationale: .eval.rationale }
}' runs/eval_reports/model_eval_<stamp>.jsonl > drafts.jsonl
```

Then you sit down for an hour, write 30-50 in-character responses to those failed prompts, and append them to `data/interim/handcrafted.pairs.jsonl`. The next training round trains on those alongside the original corpus. They land disproportionately hard because they directly cancel specific failure modes the judge already identified.

This loop ate two weekends for Monty. Each round taught me something I didn't know I needed to know.

---

## Step 8 — Merge, quantise, ship

```bash
uv run python -m run merge
```

Pulls the base + adapter, merges into a standalone HF checkpoint at `models/monty-merged/` (~8 GB). The merge step itself is two lines of peft:

```python
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, ADAPTER_ID).merge_and_unload()
merged.save_pretrained(OUT_DIR, safe_serialization=True)
```

From there it's standard llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp ~/code/llama.cpp
cd ~/code/llama.cpp && pip install -r requirements.txt

python ~/code/llama.cpp/convert_hf_to_gguf.py \
  models/monty-merged \
  --outfile models/monty-4b-f16.gguf \
  --outtype f16

# Quantise (smaller + faster on M-series chips)
~/code/llama.cpp/build/bin/llama-quantize \
  models/monty-4b-f16.gguf \
  models/monty-4b-q4_k_m.gguf \
  Q4_K_M
```

**Sample output:**

```
$ ls -lh models/
-rw-r--r--  7.5G  monty-4b-f16.gguf
-rw-r--r--  4.1G  monty-4b-q8_0.gguf
-rw-r--r--  2.4G  monty-4b-q4_k_m.gguf
```

Drop the GGUF into LM Studio at `~/.lmstudio/models/<you>/<name>/`. Paste the `SYSTEM` string from `run/train.py` into the system prompt field. The persona fires at ~50 tokens/sec on M-series silicon.

**The system prompt is half the model.** Same applies to Ollama, llama-cli, your own app. If you forget the system prompt, you get the helpful instruct model wearing the LoRA. With it, you get the character.

---

## What I'd change if I started again

Three things.

**Start with the persona prompt longer than feels comfortable.** Monty's first draft was ~150 lines and the dataset showed it. The 350-line version produces visibly better synth. The marginal cost of writing more is zero; the marginal cost of regenerating a thousand pairs because the prompt was thin is $5 and an hour.

**Run a tiny eval before the full training.** Score 50 rows of the raw synth against the judge *before* committing to a 1k-row pipeline. If the synth's data ceiling is 60%, you're not going to fine-tune your way to 80%. Better to know upfront and fix the persona prompt.

**Iterate with handcrafted corrections earlier.** Round 2 burned on hyperparameter tweaks and LoRA rank changes that did little. Round 3 fixed more with 50 hand-written examples targeting specific failure modes than any single hyperparameter change. The model learns from what you show it, not from what you wish you'd configured.

---

The engineering should be boring. The character is the work. If you can describe your character precisely enough that another human could read your persona prompt and write three convincing in-character responses, you can fine-tune a 4B model to do it for you. The pipeline doesn't care who the character is.

The character is everything that's left.

---

[← Part 1 — The Data](https://arunma.com/fine-tuning-a-4b-model-into-your-own-persona-part-1-the-data/)

Code: [github.com/arunma/learn-you-an-sft](https://github.com/arunma/learn-you-an-sft).
Adapter + GGUFs: [huggingface.co/arunma/monty3](https://huggingface.co/arunma/monty3).
The story behind the recipe: [Costume vs Character →](https://arunma.com/costume-vs-character-fine-tuning-qwen-into-monty-for-35/).
