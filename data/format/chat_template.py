"""Lesson 2 — Chat templates: the round-trip test.

WHY this file exists
====================
A Qwen2.5 model doesn't see your (prompt, response) pair as plain
text. It sees a sequence of token IDs that follow a strict format
with special turn-boundary tokens. That format is the *chat template*.
Get it wrong by a single character and the model never learns "this
is where my turn starts" — or worse, the loss mask drifts by N
positions and you train on the wrong tokens silently.

This script enforces the contract end-to-end. It builds the same
templated conversation two different ways and asserts they match:

  Path A (library):  tokenizer.apply_chat_template(messages, ...)
  Path B (hand):     literal "<|im_start|>...<|im_end|>" concatenation

If A != B the script raises AssertionError loudly so we know BEFORE
training that the template assumptions are wrong. This is the
HANDOFF § 8 definition-of-done for Stage 3a.

THE FORMAT (ChatML, as used by Qwen2.5)
=======================================
Each turn:

    <|im_start|>{role}
    {content}<|im_end|>
    [newline]

Three roles: system, user, assistant.

`<|im_start|>` and `<|im_end|>` are special tokens — they each
occupy exactly ONE id in the vocabulary (typically 151644 and 151645
for Qwen2.5). The model was trained to recognise them as turn
boundaries.

Run: `uv run python -m data.format.chat_template`
"""
from __future__ import annotations

from transformers import AutoTokenizer

# We use the Instruct variant because it ships the chat template.
# The base (non-instruct) Qwen2.5-0.5B does not have a chat template
# baked in — that's a deliberate design choice; templates belong to
# the instruction-tuned variant.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# A real example pulled from synthesis/seed.py (pair #28). Using a
# real seed pair keeps the lesson concrete: this is the actual text
# that will flow through SFT.
SYSTEM = "you are a witty friend who roasts the asker with bite."
USER = "explain monads in one sentence"
ASSISTANT = (
    "they're burrito-shaped wrappers that let you chain operations "
    "that might fail or have side effects without scattering "
    "try/except over half your codebase. Haskell's `do` notation is "
    "the least migraine-inducing way to see them in action."
)


def build_by_hand(system: str, user: str, assistant: str) -> str:
    """Reproduce the Qwen2.5 chat template by literal string concat.

    Each turn is `<|im_start|>{role}\\n{content}<|im_end|>\\n`. The
    final turn (assistant) also gets the trailing newline — the
    library's template adds it too.

    This function is the WHOLE POINT of Lesson 2: it captures, in
    code we own, the format the SFT trainer expects. Downstream
    loss-masking code (Lesson 3) will assume this exact format.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # ---- Path A: the library's chat template ----
    # `tokenize=False` returns the rendered string so we can compare
    # against our hand-built version.
    # `add_generation_prompt=False` because this is a TRAINING example
    # (the assistant turn is already provided). At inference time
    # we'd set it True to append `<|im_start|>assistant\n` and prime
    # the model to start its response.
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": ASSISTANT},
    ]
    library_rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # ---- Path B: the hand-built version ----
    hand_rendered = build_by_hand(SYSTEM, USER, ASSISTANT)

    # ---- Assertion: byte-for-byte parity ----
    # If this fails, the diff between the two strings IS the bug.
    # Print both with `repr` so invisible whitespace differences are
    # obvious.
    if library_rendered != hand_rendered:
        raise AssertionError(
            "Hand-built template does NOT match apply_chat_template.\n"
            f"--- library ({len(library_rendered)} chars) ---\n"
            f"{library_rendered!r}\n"
            f"--- hand    ({len(hand_rendered)} chars) ---\n"
            f"{hand_rendered!r}\n"
        )

    # ---- Token-level parity ----
    # `add_special_tokens=False`: the template already inlines its
    # own special tokens; we don't want the tokenizer to add a BOS
    # or anything else on top.
    library_ids = tok.encode(library_rendered, add_special_tokens=False)
    hand_ids = tok.encode(hand_rendered, add_special_tokens=False)
    assert library_ids == hand_ids, (
        f"Token IDs differ: {library_ids[:20]} vs {hand_ids[:20]}"
    )

    # ---- Inspect the result ----
    print("=" * 70)
    print("Rendered template (what the tokenizer sees BEFORE tokenizing):")
    print("=" * 70)
    print(library_rendered)
    print("=" * 70)
    print(f"\nTotal tokens after encoding: {len(library_ids)}")
    print(f"\nFirst 16 token IDs:")
    print(f"  {library_ids[:16]}")
    print(f"\nFirst 16 decoded (one per token):")
    for tok_id in library_ids[:16]:
        decoded = tok.decode([tok_id])
        print(f"  {tok_id:>7}  ->  {decoded!r}")

    # ---- Special tokens ----
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"\nSpecial token IDs (turn boundaries):")
    print(f"  <|im_start|>  ->  {im_start_id}")
    print(f"  <|im_end|>    ->  {im_end_id}")
    print(f"\n<|im_start|> appears {library_ids.count(im_start_id)} times "
          f"(one per turn).")
    print(f"<|im_end|>   appears {library_ids.count(im_end_id)} times "
          f"(one per turn).")

    print("\nRound-trip OK: hand-built template matches the library "
          "byte-for-byte and token-for-token.")


if __name__ == "__main__":
    main()
