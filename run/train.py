"""SFT training for Monty: LoRA on Qwen3-4B-Instruct via TRL."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from prep import PROCESSED_DIR, RUNS_DIR
from prep.schema import read_jsonl


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

OUTPUT_DIR = RUNS_DIR / "checkpoints"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.jsonl"
VAL_DATA_PATH = PROCESSED_DIR / "val.jsonl"

# Must match the inference-time system prompt verbatim or the persona won't fire.
SYSTEM = (
    "you are Monty — a foul-mouthed, opinionated friend who curses casually, "
    "takes real positions, and actually helps. lowercase by default, "
    "no service-speak, no hedging."
)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Tuned for a 48GB GPU. For 32GB drop to BATCH_SIZE=4, GRAD_ACCUMULATION=4.
EPOCHS = 2
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 1024


def _to_messages(pair) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": pair.prompt},
            {"role": "assistant", "content": pair.response},
        ]
    }


def build_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise SystemExit(
            f"Training data not found at {path}. "
            "Run the synthesis + filter pipeline first."
        )
    return Dataset.from_list([_to_messages(p) for p in read_jsonl(path)])


def maybe_build_eval_dataset(path: Path) -> Dataset | None:
    if not path.exists():
        return None
    return Dataset.from_list([_to_messages(p) for p in read_jsonl(path)])


def _sanity_check_generation(model, tokenizer) -> None:
    print("\n--- Sanity check ---")
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "should I learn Rust?"},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
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
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    print(f"Response: {response!r}")


def _maybe_push_to_hub(trainer, tokenizer) -> None:
    repo = os.environ.get("HF_PUSH_REPO")
    if not repo:
        return
    print(f"\nPushing to HF Hub: {repo}")
    try:
        trainer.model.push_to_hub(repo, private=True)
        tokenizer.push_to_hub(repo, private=True)
        print(f"  OK — https://huggingface.co/{repo}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


def main() -> None:
    print(f"Loading {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataset = build_dataset(TRAIN_DATA_PATH)
    eval_dataset = maybe_build_eval_dataset(VAL_DATA_PATH)
    has_eval = eval_dataset is not None

    print(
        f"Train: {len(train_dataset)} examples"
        + (
            f", Val: {len(eval_dataset)} examples"
            if has_eval
            else " (no val.jsonl — eval disabled)"
        )
    )

    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LENGTH,
        logging_steps=1,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=50,
        save_strategy="steps" if has_eval else "epoch",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to="tensorboard",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("\nTrainable parameters:")
    trainer.model.print_trainable_parameters()

    effective_batch = BATCH_SIZE * GRAD_ACCUMULATION
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    print(
        f"{EPOCHS} epochs x ~{steps_per_epoch} steps/epoch "
        f"(effective batch={effective_batch})"
    )

    trainer.train()

    final_path = OUTPUT_DIR / "final"
    trainer.save_model(str(final_path))
    print(f"\nSaved adapter to {final_path}")

    _sanity_check_generation(model, tokenizer)
    _maybe_push_to_hub(trainer, tokenizer)


if __name__ == "__main__":
    main()
