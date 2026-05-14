"""Lesson 4 — Batching with padding.

WHY this exists
===============
A single example is shape (seq_len,). The GPU is fast on batches —
shape (batch_size, max_seq_len). To stack examples of different
lengths into a single tensor we have to pad the short ones to match
the longest.

THE TRIPLY-INERT PADDING CONTRACT
=================================
Padding positions are made inert in three places at once:

  input_ids       :  pad_token_id    (filler; model never sees it)
  attention_mask  :  0               (self-attention ignores it)
  labels          :  -100            (loss ignores it)

So the padded positions don't contribute to the forward pass and
they don't contribute to the gradient.

KEY DECISIONS
=============
- pad_token_id: Qwen2.5 doesn't set one by default. We use
  tokenizer.eos_token_id (<|endoftext|>). Pick is arbitrary — the
  triply-inert contract means it never matters at the model level.
- padding_side: "right" for training. (Left-padding only matters
  at inference time for batched generation.)
- Dynamic padding: pad to the longest sequence IN THIS BATCH, not
  to a global max_length. Saves wasted compute when lengths vary.

PACKING (a separate optimisation — not done here)
=================================================
Packing concatenates multiple short examples into one fixed-length
sequence with a block-diagonal attention mask that prevents cross-
example attention. ~2-3x throughput on length-heterogeneous data.
TRL does it automatically with packing=True. Phase B's hand-rolled
trainer can add it later. For learning, plain padding is the clearer
baseline.

Run: `uv run python -m data.format.collate`
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from data.format.loss_mask import LABEL_IGNORE, build_input_ids_and_labels
from synthesis.seed import SEED_PAIRS

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Same system message used across the training corpus. In production
# we might rotate among a small set of system prompts to broaden the
# persona's adaptability; here we use one for simplicity.
DEFAULT_SYSTEM = "you are a witty friend who roasts the asker with bite."


def collate_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Stack a list of (input_ids, labels) into batched tensors.

    Returns a dict suitable for `model(**batch)`:

        input_ids:      (B, L)  long
        attention_mask: (B, L)  long  -- 1 for real tokens, 0 for padding
        labels:         (B, L)  long  -- -100 at padded positions

    where L = max length in this batch (dynamic padding).
    """
    if not examples:
        raise ValueError("Empty batch")

    # Find the longest sequence in the batch — we'll pad everything
    # else up to this length. Anything longer was already truncated
    # upstream if you need a hard cap (we don't here — our examples
    # are well under any reasonable max_length).
    max_len = max(len(ids) for ids, _ in examples)

    input_ids_batch: list[list[int]] = []
    attention_mask_batch: list[list[int]] = []
    labels_batch: list[list[int]] = []

    for input_ids, labels in examples:
        pad_n = max_len - len(input_ids)
        # Right-padding: real tokens first, padding at the end.
        input_ids_batch.append(input_ids + [pad_token_id] * pad_n)
        attention_mask_batch.append([1] * len(input_ids) + [0] * pad_n)
        labels_batch.append(labels + [LABEL_IGNORE] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen2.5's tokenizer doesn't set pad_token by default. Bind it
    # to <|endoftext|>. It's inert in our batches (attention_mask=0,
    # labels=-100) so the choice doesn't reach the model.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Build (input_ids, labels) for the first 4 seed pairs.
    n = 4
    examples = [
        build_input_ids_and_labels(
            tok,
            DEFAULT_SYSTEM,
            pair.prompt,
            pair.response,
        )
        for pair in SEED_PAIRS[:n]
    ]

    print("Per-example lengths before padding:")
    for i, (ids, _) in enumerate(examples):
        print(f"  example {i}: {len(ids):>3} tokens  -- "
              f"prompt: {SEED_PAIRS[i].prompt[:60]!r}")

    batch = collate_batch(examples, pad_token_id=tok.pad_token_id)

    print(f"\nAfter collation (dynamic padding to longest in batch):")
    for key, value in batch.items():
        print(f"  {key:>15}: {tuple(value.shape)}  dtype={value.dtype}")

    # Label stats across the whole batch.
    n_neg100 = int((batch["labels"] == LABEL_IGNORE).sum().item())
    n_total = int(batch["labels"].numel())
    n_active = n_total - n_neg100
    print(f"\nLabel stats (across the whole batch):")
    print(f"  Total positions: {n_total}")
    print(f"  Masked (-100):   {n_neg100}  ({100 * n_neg100 / n_total:.1f}%)")
    print(f"  Active loss:     {n_active}  ({100 * n_active / n_total:.1f}%)")

    # Inspect the shortest example's padded tail — this is where the
    # triply-inert contract is visible. Find the shortest example by
    # counting where attention_mask transitions from 1 to 0.
    real_lengths = batch["attention_mask"].sum(dim=1).tolist()
    shortest_idx = int(torch.tensor(real_lengths).argmin().item())
    real_len = int(real_lengths[shortest_idx])
    print(f"\nShortest example is #{shortest_idx} "
          f"(real_len={real_len}, padded to {batch['input_ids'].shape[1]}).")
    print("Tail of that example (last 8 positions — should be padding):")
    tail = slice(-8, None)
    print(f"  input_ids:      {batch['input_ids'][shortest_idx, tail].tolist()}")
    print(f"  attention_mask: {batch['attention_mask'][shortest_idx, tail].tolist()}")
    print(f"  labels:         {batch['labels'][shortest_idx, tail].tolist()}")
    print(f"\npad_token_id = {tok.pad_token_id}, LABEL_IGNORE = {LABEL_IGNORE}")
    print("Every padded position should show "
          f"(input_ids={tok.pad_token_id}, attention_mask=0, labels={LABEL_IGNORE}).")


if __name__ == "__main__":
    main()
