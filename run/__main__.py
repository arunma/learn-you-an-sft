"""Model lifecycle commands.

  uv run python -m run train     # LoRA SFT on data/processed/{train,val}.jsonl
  uv run python -m run eval      # generate + judge against the trained adapter
  uv run python -m run merge     # merge adapter into base; write HF checkpoint
"""
from __future__ import annotations

import argparse


COMMANDS = ("train", "eval", "merge")


def _dispatch(cmd: str) -> None:
    if cmd == "train":
        from run.train import main as train_main
        train_main()
    elif cmd == "eval":
        from run.eval import main as eval_main
        eval_main()
    elif cmd == "merge":
        from run.merge import main as merge_main
        merge_main()
    else:
        raise SystemExit(f"Unknown command: {cmd!r}. Choose from {COMMANDS}.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="run", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    _dispatch(args.command)


if __name__ == "__main__":
    main()
