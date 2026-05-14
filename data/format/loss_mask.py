"""Lesson 3 — Loss masking for SFT.

WHY this exists
===============
A training example is a tokenized sequence: ~30 prompt tokens followed
by ~65 response tokens. We DO NOT want the loss to fire on the prompt
portion — only on the response. Two reasons:

  1. The prompt is supplied by the user at inference. Training the
     model to "predict" prompt tokens is pure capacity waste.
  2. Without masking, the model can leak prompt-shaped artefacts into
     its responses (e.g. starting a reply with "user:").

The mechanism: PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)`
silently drops positions whose label is -100. HF's causal-LM
`model.forward(labels=...)` uses this contract. So we build a
`labels` array that's a copy of `input_ids` but with prompt positions
set to -100.

THE CAUSAL-LM SHIFT (worth internalising)
=========================================
HF computes the loss as:

    shift_logits = logits[..., :-1, :]   # everything except last
    shift_labels = labels[..., 1:]        # everything except first
    loss = CrossEntropy(shift_logits.view(-1, V),
                        shift_labels.view(-1),
                        ignore_index=-100)

So if `labels[0..n-1] = -100` (prompt) and `labels[n..n+m-1] =
response`, the first position where loss fires is `n-1` (the last
prompt token) — that position is the model PREDICTING the first
response token, which is what we want. Loss continues through the
final `<|im_end|>` of the assistant turn — important, because the
model needs to learn WHEN to stop.

FINDING THE BOUNDARY (Approach A)
=================================
1. Render the prompt-only portion with `add_generation_prompt=True`.
   That gives the exact string the assistant would see and start
   generating from. It ends with `<|im_start|>assistant\\n`.
2. Tokenize that prompt-only string. Its length is the response-start
   offset in the full sequence.
3. Render the full conversation; tokenize it.
4. Assert that the prompt-only token IDs are a prefix of the full
   sequence's token IDs. (Catches BPE-merge gotchas across the
   prompt/response boundary — rare but real.)
5. Build labels: -100 for positions [0, offset), full_ids for
   positions [offset, end).

PADDING (deferred to Lesson 4)
==============================
In a batch, short sequences are padded to the longest with
`pad_token_id`. Those padding positions ALSO get -100 so they don't
pollute the loss. Batching/packing/padding is Lesson 4's territory;
this file returns (input_ids, labels) for a single example.

Run: `uv run python -m data.format.loss_mask`
"""
from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# PyTorch CrossEntropyLoss's default ignore_index. HF's causal LMs
# treat any labels position equal to this value as masked.
LABEL_IGNORE = -100

# Same seed pair as Lesson 2 (synthesis/seed.py pair #28) — gives
# continuity so you can mentally line up Lesson 2's tokens with
# Lesson 3's masking.
SYSTEM = "you are a witty friend who roasts the asker with bite."
USER = "explain monads in one sentence"
ASSISTANT = (
    "they're burrito-shaped wrappers that let you chain operations "
    "that might fail or have side effects without scattering "
    "try/except over half your codebase. Haskell's `do` notation is "
    "the least migraine-inducing way to see them in action."
)


def build_input_ids_and_labels(
    tokenizer: PreTrainedTokenizerBase,
    system: str,
    user: str,
    assistant: str,
) -> tuple[list[int], list[int]]:
    """Build (input_ids, labels) for one SFT training example.

    `labels[i]` is `input_ids[i]` for response positions and
    LABEL_IGNORE (-100) for prompt positions. Returned as plain
    Python lists so this function has zero PyTorch dependency — the
    trainer converts to tensors.
    """
    # ---- Step 1: render the prompt-only portion ----
    # add_generation_prompt=True appends "<|im_start|>assistant\n"
    # so the tokenized length covers exactly the "your turn starts
    # here" boundary, but no response content.
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_rendered = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_rendered, add_special_tokens=False)

    # ---- Step 2: render the full conversation ----
    full_messages = prompt_messages + [
        {"role": "assistant", "content": assistant}
    ]
    full_rendered = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(full_rendered, add_special_tokens=False)

    # ---- Step 3: verify the prompt is a token-prefix of the full ----
    # Subtle bug class: a tokenizer COULD produce different IDs for
    # the same string when it appears at the end of a longer string
    # (because of BPE merges that cross the boundary). If that ever
    # happens here, our offset is wrong and labels drift. Assert
    # early so we know.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Prompt is not a token-prefix of the full sequence. "
            "Likely a BPE merge across the prompt/response boundary."
        )

    # ---- Step 4: build labels with prompt positions masked ----
    n_prompt = len(prompt_ids)
    labels: list[int] = [LABEL_IGNORE] * n_prompt + list(full_ids[n_prompt:])
    assert len(labels) == len(full_ids)

    return list(full_ids), labels


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    input_ids, labels = build_input_ids_and_labels(
        tok, SYSTEM, USER, ASSISTANT
    )

    # Summary stats
    n_total = len(input_ids)
    n_masked = sum(1 for lab in labels if lab == LABEL_IGNORE)
    n_active = n_total - n_masked

    print(f"Total tokens:  {n_total}")
    print(f"Masked (-100): {n_masked}  ({100 * n_masked / n_total:.1f}%)")
    print(f"Active loss:   {n_active}  ({100 * n_active / n_total:.1f}%)")

    # Position-by-position table — see exactly where the boundary is
    # and which tokens contribute to the loss.
    print("\n" + "=" * 80)
    print(f"{'pos':>4} | {'id':>7} | {'label':>8} | token")
    print("-" * 80)
    for i, (tid, lab) in enumerate(zip(input_ids, labels)):
        label_str = "-100" if lab == LABEL_IGNORE else str(lab)
        decoded = tok.decode([tid]).replace("\n", "\\n")
        marker = ""
        if i > 0 and labels[i - 1] == LABEL_IGNORE and lab != LABEL_IGNORE:
            marker = "   <-- response starts here (loss begins at i-1)"
        print(f"{i:>4} | {tid:>7} | {label_str:>8} | {decoded!r}{marker}")


if __name__ == "__main__":
    main()
