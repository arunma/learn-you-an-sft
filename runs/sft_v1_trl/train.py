"""Lesson 5 — Phase A: SFT with TRL + LoRA (OOTB).

WHAT this does
==============
Fine-tunes Qwen2.5-0.5B-Instruct on our 24 hand-crafted seed pairs
using LoRA (Low-Rank Adaptation) via the `peft` library, wrapped by
`trl.SFTTrainer`. **This file uses TRL's out-of-the-box (OOTB) APIs
exclusively** — no pre-tokenization, no custom collator. TRL handles:

  - applying the chat template to each pair (via `apply_chat_template`),
  - tokenizing,
  - masking the prompt portion (`assistant_only_loss=True`),
  - batching + padding (`SFTTrainer`'s default data collator).

This is the deliberate pedagogical split:
  - Phase A (this file) = let TRL do everything; minimal user code.
  - Phase B (Lesson 6, hand-rolled) = use our Lesson 3 / 4 code
    (`build_input_ids_and_labels`, `collate_batch`) directly with
    `transformers.Trainer` or a manual loop. Same training outcome;
    every step explicit.

Output: a LoRA adapter (~10 MB) at `runs/sft_v1_trl/checkpoints/final/`.
The base model weights stay frozen and on disk; only the adapter is
trained.

LORA — IN ONE PARAGRAPH
=======================
For any linear layer with weight matrix W (shape d_out x d_in), LoRA
adds a rank-r delta: ΔW = (α/r) · B · A, where B (d_out x r) is
initialized to zero and A (r x d_in) to a Gaussian. The base W is
frozen; only A and B are trained. For Qwen2.5-0.5B with r=16 attached
to the four attention projections across 24 layers, that's ~2.75M
trainable params vs ~500M base → 0.55%. Much cheaper to train, faster
to checkpoint, and far less risk of catastrophic forgetting.

THE FOUR LORA KNOBS (LoraConfig)
================================
- r              : rank of the adapter. Higher = more capacity. 8/16/32 typical.
- lora_alpha     : scaling factor. Effective LR for LoRA is α/r.
                   Convention: α = 2r (so α/r = 2).
- target_modules : which linear layers get a LoRA pair. Attention-only
                   (q/k/v/o_proj) is minimal; "all-linear" is the
                   QLoRA-paper preference.
- lora_dropout   : dropout on LoRA's hidden representation. 0.05-0.1.

COMPLETION-ONLY LOSS — THE MODERN TRL WAY
=========================================
Older TRL (≤ 0.13) shipped `DataCollatorForCompletionOnlyLM` for this.
It was deprecated in 0.14 and removed in 0.18+. The replacement is
`assistant_only_loss=True` on `SFTConfig`. TRL:

  1. Applies the chat template via `apply_chat_template`.
  2. Finds the assistant turn via `{% generation %}` blocks in the
     template's Jinja (Qwen2.5's recent revisions have these).
  3. Sets `labels = -100` for everything outside the assistant turn.

Same effect as DataCollatorForCompletionOnlyLM; cleaner API.

EXPECTED BEHAVIOUR ON 24 EXAMPLES
=================================
The model overfits hard. Loss drops to near zero. Generation mostly
parrots back exact seed responses for prompts it saw and produces
garbage for new ones. That's expected. The goal here is to verify
the pipeline runs end-to-end before pointing it at the 15K-pair
Gemini corpus.

Run: `uv run python -m runs.sft_v1_trl.train`

On Mac CPU (fp32): ~2-3 minutes total.
On RunPod H100 (bf16): seconds, but launch the same script.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data.ingest.schema import read_jsonl

# ----- Model / output -----
# Qwen3-4B-Instruct-2507 — the non-thinking Instruct variant of Qwen3-4B.
# Round 2 used Qwen/Qwen2.5-3B-Instruct; round 3 swaps to Qwen3-4B for better
# base capability (sharper technical priors, stronger instruction-following).
# Avoid the plain "Qwen/Qwen3-4B" (hybrid thinking mode) and the
# "Qwen/Qwen3-4B-Thinking-2507" variant — those toggle <think> blocks that
# interfere with persona-style direct chat.
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Training data — the quality-filtered + re-split Gemini corpus.
# Pipeline:
#   uv run python -m synthesis.question_pool --count 15000
#   uv run python -m synthesis.generate --questions-file ... --count 15000
#   uv run python -m data.filter.pipeline
#   uv run python -m eval.score_dataset --input data/processed/train.jsonl ...
#   uv run python -m eval.score_dataset --input data/processed/val.jsonl   ...
#   uv run python -m eval.repartition --scored-train ... --scored-val ...
# Gitignored. Must exist before training.
TRAIN_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "train.jsonl"
)
# Validation set — same schema as train.jsonl. Used for eval_loss + best-checkpoint
# selection. Missing val.jsonl is non-fatal; eval is just disabled in that case.
VAL_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "val.jsonl"
)

# ----- Persona -----
# Explicit Monty system prompt — the model sees this both during training
# and at inference, so the persona has a stronger handle than a generic
# "you are a witty friend" string. The training-time system prompt must
# match the inference-time system prompt verbatim or the persona won't
# fire reliably at inference.
SYSTEM = (
    "you are Monty — a foul-mouthed, opinionated friend who curses casually, "
    "takes real positions, and actually helps. lowercase by default, "
    "no service-speak, no hedging."
)

# ----- LoRA config -----
LORA_R = 16
LORA_ALPHA = 32          # alpha = 2r convention
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ----- Training config -----
# Tuned for a 48GB GPU (RTX A6000 / RTX 6000 Ada / L40S).
# For a 96GB GPU (RTX PRO 6000): BATCH_SIZE=16, GRAD_ACCUMULATION=1.
# For a 32GB GPU (RTX 5090):     BATCH_SIZE=4,  GRAD_ACCUMULATION=4.
EPOCHS = 2               # was 3; previous run showed loss plateaued by epoch ~1.
                         # With load_best_model_at_end=True (see SFTConfig), TRL
                         # will keep the best checkpoint regardless of where it landed.
BATCH_SIZE = 8           # 4B + LoRA + bf16 + grad_checkpointing on 48GB — peak ~26-30GB.
                         # Logits tensor (batch x seq x 152k vocab x 2 bytes) dominates.
                         # If OOM in first 20 steps, drop to BATCH_SIZE=4, GRAD_ACCUMULATION=4.
GRAD_ACCUMULATION = 2    # Effective batch = BATCH_SIZE * GRAD_ACCUMULATION = 16
LEARNING_RATE = 1e-4     # was 2e-4; lowered after observing grad_norm climbing late in
                         # training (thrashing on conflicting examples). LoRA still
                         # tolerates higher LR than full-FT because only ~0.2% of weights move.
MAX_SEQ_LENGTH = 1024    # longest Monty responses approach ~500 tokens; 512 truncates


def build_dataset() -> Dataset:
    """Read Pair JSONL from data/processed/train.jsonl and convert
    to a HF Dataset of {"messages": [...]} rows.

    TRL's SFTTrainer will iterate this dataset, call
    `tokenizer.apply_chat_template` per row, tokenize, mask the
    prompt portion (via `assistant_only_loss=True`), and hand batches
    to its default data collator.
    """
    if not TRAIN_DATA_PATH.exists():
        raise SystemExit(
            f"Training data not found at {TRAIN_DATA_PATH}.\n"
            "Run the synthesis + filter pipeline first:\n"
            "  uv run python -m synthesis.question_pool --count 15000\n"
            "  uv run python -m synthesis.generate --questions-file "
            "data/interim/question_pool.jsonl --count 15000\n"
            "  uv run python -m data.filter.pipeline"
        )

    rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": pair.prompt},
                {"role": "assistant", "content": pair.response},
            ]
        }
        for pair in read_jsonl(TRAIN_DATA_PATH)
    ]
    return Dataset.from_list(rows)


def build_eval_dataset() -> Dataset | None:
    """Read val.jsonl into the same chat-message shape as the train dataset.

    Returns None if val.jsonl is absent — useful for local CPU smoke runs
    where the dataset hasn't been scp'd over yet. The trainer falls back
    to no per-step eval in that case.
    """
    if not VAL_DATA_PATH.exists():
        return None

    rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": pair.prompt},
                {"role": "assistant", "content": pair.response},
            ]
        }
        for pair in read_jsonl(VAL_DATA_PATH)
    ]
    return Dataset.from_list(rows)


def main() -> None:
    print(f"Loading tokenizer + model: {MODEL_ID}")
    print("(First run downloads ~1 GB of weights to ~/.cache/huggingface/)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        # Qwen2.5 doesn't set pad_token by default — Lesson 4 gotcha.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,  # H100 native; matches `bf16=True` in SFTConfig below
    )

    # ---- LoRA configuration ----
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Dataset (messages format; TRL tokenizes internally) ----
    train_dataset = build_dataset()
    eval_dataset = build_eval_dataset()

    print(f"\nDataset: {len(train_dataset)} train examples", end="")
    if eval_dataset is not None:
        print(f", {len(eval_dataset)} val examples (per-step eval enabled)")
    else:
        print(" (no val.jsonl found — eval disabled, falling back to epoch save)")

    # ---- SFTConfig (TRL's TrainingArguments) ----
    has_eval = eval_dataset is not None
    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LENGTH,  # was `max_seq_length=` in TRL <0.16
        logging_steps=1,
        # Per-step eval + best-checkpoint selection — fixes the "loss flat
        # but grad_norm climbing" diagnosis from the previous run.
        eval_strategy="steps" if has_eval else "no",
        eval_steps=50,
        save_strategy="steps" if has_eval else "epoch",
        save_steps=50,
        save_total_limit=3,        # keep best 3 by eval_loss
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to="tensorboard",   # writes tfevents to OUTPUT_DIR/runs/<timestamp>/
        bf16=True,                 # H100 setting (Mac CPU build: set False + dtype=fp32)
        fp16=False,
        gradient_checkpointing=True,  # ~10% slower, ~30% VRAM headroom — cheap insurance
        # Modern OOTB replacement for DataCollatorForCompletionOnlyLM.
        # TRL applies the chat template, finds the assistant turn via
        # the Jinja {% generation %} blocks, and masks everything else
        # to labels=-100. Requires TRL >= 0.13 AND a tokenizer whose
        # chat template includes {% generation %} blocks (Qwen2.5
        # recent revisions do).
        #
        # If you see an error about "no assistant tokens found" or the
        # parameter being unrecognised, drop this line — the demo
        # still trains, just including prompt tokens in the loss too
        # (sub-optimal but functional on 24 examples).
        assistant_only_loss=True,
    )

    # ---- Trainer (TRL handles tokenizing, batching, masking, padding) ----
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,   # None = no per-step eval; non-None = enables it
        peft_config=peft_config,
        processing_class=tokenizer,  # TRL >=0.13 API (was `tokenizer=`)
    )

    print("\nTrainable parameters under LoRA:")
    trainer.model.print_trainable_parameters()

    effective_batch = BATCH_SIZE * GRAD_ACCUMULATION
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    print(
        f"\nStarting training: {EPOCHS} epochs x ~{steps_per_epoch} "
        f"optimizer steps/epoch (effective batch={effective_batch}) "
        f"= ~{EPOCHS * steps_per_epoch} total optimizer steps."
    )
    trainer.train()

    final_path = OUTPUT_DIR / "final"
    print(f"\nSaving adapter to {final_path}")
    trainer.save_model(str(final_path))

    # ---- Sanity-check generation ----
    print("\n--- Sanity-check generation (greedy, no sampling) ---")
    model.eval()
    # Canonical persona test prompt — what Monty should sound like on it
    # appears as example #1 in synthesis/persona_prompt.md.
    test_prompt = "should I learn Rust?"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": test_prompt},
    ]

    print("CHAT TEMPLATE:", tokenizer.chat_template)
    # return_dict=True for forward-compat with recent transformers,
    # which return a BatchEncoding (dict-like) from apply_chat_template
    # rather than a bare tensor. Unpack explicitly so we know what
    # shape we're feeding to .generate().
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    # Move inputs to the model's device. Accelerate/TRL may have
    # placed the model on MPS (Apple Silicon GPU) or CUDA during
    # training; `apply_chat_template` returns CPU tensors regardless.
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True,
    )
    print(f"Prompt:   {test_prompt!r}")
    print(f"Response: {response!r}")
    # ---- Post-training HF Hub push (opt-in via env var) ----
    #   HF_PUSH_REPO=arunma/monty            (e.g.)
    #     Push the adapter + tokenizer to a private HF Hub repo. Lets
    #     you recover the model without scp'ing from the pod. Requires
    #     HF_TOKEN env var set (huggingface-cli login or via .env).
    #
    # NOTE: pod does NOT auto-terminate. Run `runpodctl remove pod <id>`
    # manually when you're done with training + eval — or chain it on
    # your launch command if you want unattended shutdown.
    hub_repo = os.environ.get("HF_PUSH_REPO")
    if hub_repo:
        print(f"\nPushing adapter + tokenizer to HF Hub: {hub_repo}")
        try:
            trainer.model.push_to_hub(hub_repo, private=True)
            tokenizer.push_to_hub(hub_repo, private=True)
            print(f"  OK — https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print("  Pod stays alive — scp the adapter from checkpoints/final manually.")


if __name__ == "__main__":
    main()
