"""Merge a LoRA adapter into its base model and save as a standalone HF checkpoint.

Output is ready for llama.cpp's `convert_hf_to_gguf.py` to turn into a GGUF.

Override the base and adapter via env vars:
    BASE_ID=Qwen/Qwen3-4B-Instruct-2507 ADAPTER_ID=arunma/monty3 \\
        uv run python -m run merge
"""
from __future__ import annotations

import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prep import REPO_ROOT

BASE_ID = os.environ.get("BASE_ID", "Qwen/Qwen3-4B-Instruct-2507")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "arunma/monty3")
OUT_DIR = REPO_ROOT / "models" / "monty-merged"


def main() -> None:
    load_dotenv()  # private adapter pull below needs HF_TOKEN

    print(f"merging {ADAPTER_ID} into {BASE_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, ADAPTER_ID).merge_and_unload()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved {OUT_DIR}")
    print(f"\nnext: python ~/code/llama.cpp/convert_hf_to_gguf.py {OUT_DIR} "
          "--outfile models/monty-f16.gguf --outtype f16")


if __name__ == "__main__":
    main()
