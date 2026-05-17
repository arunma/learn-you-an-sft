"""Generate Monty responses with the fine-tuned adapter (or just the base, for baselines)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Generation:
    prompt: str
    gold: str
    response: str


def load_monty(base_id: str, adapter_id: str | None):
    """Load base + LoRA adapter. Pass adapter_id=None to load only the base (baseline eval)."""
    tokenizer_source = adapter_id or base_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_id:
        model = PeftModel.from_pretrained(base, adapter_id)
    else:
        model = base

    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompts_and_golds: Sequence[tuple[str, str]],
    *,
    system: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    progress: bool = True,
    progress_every: int = 10,
) -> list[Generation]:
    """Run generation one prompt at a time. Returns Generations in input order."""
    out: list[Generation] = []
    total = len(prompts_and_golds)
    device = next(model.parameters()).device
    start = time.monotonic()

    if progress:
        print(f"  Generating {total} responses (T={temperature}, top_p={top_p})...", flush=True)

    for i, (prompt, gold) in enumerate(prompts_and_golds):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = gen[0][input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        out.append(Generation(prompt=prompt, gold=gold, response=response))

        done = i + 1
        if progress and (done % progress_every == 0 or done == total):
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 else 0.0
            print(
                f"  generated {done}/{total}  "
                f"({elapsed:.0f}s elapsed, {rate:.2f} prompts/s, "
                f"~{remaining:.0f}s remaining)",
                flush=True,
            )

    return out
