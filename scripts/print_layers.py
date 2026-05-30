"""Print a base model's architecture + the LoRA target-module candidates.

  uv run python -m scripts.print_layers
  uv run python -m scripts.print_layers --model Qwen/Qwen3-4B-Instruct-2507

Materialises the model on torch.device("meta") so we get the full module tree
with zero VRAM / zero RAM allocation — just the structure.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(config)

    print("=" * 70)
    print(f"Architecture: {args.model}")
    print("=" * 70)
    print(base)
    print()

    proj_names = [n for n, _ in base.named_modules() if n.endswith("_proj")]
    suffixes = sorted({n.rsplit(".", 1)[-1] for n in proj_names})

    print("=" * 70)
    print(f"All *_proj modules: {len(proj_names)} total")
    print("=" * 70)
    print(f"Unique suffix names ({len(suffixes)}): {suffixes}")
    print()
    print("Layer 0 projections (the pattern repeats for every decoder layer):")
    for name in proj_names:
        if name.startswith("model.layers.0."):
            print(f"  {name}")


if __name__ == "__main__":
    main()
