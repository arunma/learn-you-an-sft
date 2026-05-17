"""Merge the Monty LoRA adapter into Qwen2.5-0.5B-Instruct and save as a standalone HF model.

This produces a self-contained HF-format checkpoint ready to convert to GGUF via
llama.cpp's `convert_hf_to_gguf.py`. See LM_STUDIO.md for the full pipeline.

Usage:
    uv run python -m inference.merge_for_gguf

Output:
    models/monty-merged/   (HF format: config.json + model.safetensors + tokenizer files)
"""
from __future__ import annotations

from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_ID = "arunma/monty"
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "monty-merged"


def main() -> None:
    # Load HF_TOKEN (and anything else) from .env, matching the eval scripts.
    # The private adapter pull on the next line needs HF_TOKEN to succeed.
    load_dotenv()

    print(f"Loading tokenizer from {ADAPTER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

    print(f"Loading base model {BASE_ID} (bf16)")
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16)

    print(f"Applying LoRA adapter {ADAPTER_ID}")
    model = PeftModel.from_pretrained(base, ADAPTER_ID)

    print("Merging adapter weights into base")
    merged = model.merge_and_unload()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {OUT_DIR}")
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)

    print("\nDone. Next step:")
    print(f"  python ~/code/llama.cpp/convert_hf_to_gguf.py {OUT_DIR} \\")
    print("    --outfile models/monty-f16.gguf --outtype f16")


if __name__ == "__main__":
    main()
