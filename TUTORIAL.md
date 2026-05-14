# learn-you-an-sft — SFT from First Principles

A copywork-style tutorial on building a supervised fine-tuning (SFT)
pipeline by hand. Every concept is taught with a short, self-contained
Python file you can run, inspect, and modify. The goal is to
understand what `trainer.train()` is actually doing — not just call it.

If you've ever finetuned a model with TRL or `transformers.Trainer`
and felt like a chunk of it was magic, this is for you.

---

## Table of contents

1. [Who this is for](#1-who-this-is-for)
2. [What you'll build](#2-what-youll-build)
3. [Setup](#3-setup)
4. [Background — pretraining vs SFT](#4-background--pretraining-vs-sft)
5. [Lesson 1 — Hand-crafting training pairs](#5-lesson-1--hand-crafting-training-pairs)
6. [Lesson 2 — Chat templates](#6-lesson-2--chat-templates)
7. [Lesson 3 — Loss masking](#7-lesson-3--loss-masking)
8. [Lesson 4 — Batching with padding](#8-lesson-4--batching-with-padding)
9. [What's next (Lessons 5+)](#9-whats-next-lessons-5)
10. [Decisions log — don't re-litigate](#10-decisions-log--dont-re-litigate)
11. [Glossary](#11-glossary)

---

## 1. Who this is for

You should be comfortable with:
- Python 3.10+, type hints, dataclasses
- The general shape of a transformer (you know what attention and
  next-token prediction are, even if you couldn't reimplement them
  from scratch)
- `pip` / `uv` and virtualenvs

You don't need:
- Prior experience with TRL, PEFT, or `transformers.Trainer`
- A GPU for the first four lessons (everything runs on a laptop CPU
  in seconds)

What you'll come away knowing:
- Why an SFT trainer needs (prompt, response) pairs — and what makes
  a *good* pair vs a useless one
- What a chat template actually is, why it has special tokens, and
  how to verify yours matches the library's
- How `-100` loss masking works mechanically, including the causal-LM
  label shift that confuses everyone the first time
- How padding, attention masks, and the triply-inert padding contract
  fit together in a batch

---

## 2. What you'll build

A small instruct-tuned LLM that responds like a witty friend with
bite. The persona is "useful answer + sarcastic delivery + casual
profanity when it fits." Concretely: it can answer "should I learn
Rust?" with both a real recommendation and a roast, in one breath.

The model is `Qwen2.5-0.5B-Instruct`. It's small enough to iterate
on a laptop, and ships with the chat template / tool-call format
already baked in.

The full pipeline (everything in this tutorial covers Stages 1.5–2b
and Stage 3a — the data-prep half):

```
        synthesis/  (Gemini 2.5 Pro)
                 │
                 ▼
        data/interim/*.pairs.jsonl
                 │
                 ▼
        data/filter/  (normalize → language → dedup)
                 │
                 ▼
        data/processed/{train,val}.jsonl + manifest.json
                 │
                 ▼
        data/format/  ← this tutorial focuses here
        chat_template, loss_mask, collate
                 │
                 ▼
        runs/sft_v1_trl/  (TRL + LoRA) ← Lesson 5
        runs/sft_v2_internals/ (hand-rolled) ← Lesson 6
                 │
                 ▼
        inference/chat.py  ← Lesson 7
```

The repo is at https://github.com/arunma/learn-you-an-sft.

---

## 3. Setup

```bash
git clone https://github.com/arunma/learn-you-an-sft.git
cd learn-you-an-sft
uv venv
uv pip install -e .
```

That installs everything: `transformers`, `tokenizers`, `torch`,
`peft`, `trl`, plus the filter-pipeline deps. `bitsandbytes` is
CUDA-only and is gated to Linux in `pyproject.toml`, so macOS installs
succeed without it.

First time you run any lesson that touches the tokenizer or model,
HuggingFace will download Qwen2.5-0.5B-Instruct's tokenizer files
(~5 MB) to `~/.cache/huggingface/`. Tokenizer-only fetches are fast;
the full model weights (~1 GB) are only downloaded when you actually
train.

If you intend to do the full pipeline later (Lessons 5+ involve
synthesis + judge):

```bash
export GEMINI_API_KEY=...     # Gemini 2.5 Pro — the teacher
export ANTHROPIC_API_KEY=...  # Claude Haiku 4.5 — the judge
```

For the first four lessons you don't need either.

---

## 4. Background — pretraining vs SFT

Before fine-tuning, `Qwen2.5-0.5B-Instruct` already knows:
- English grammar, vocabulary, math, code, common knowledge — from
  pretraining on trillions of tokens.
- How to follow instructions in a chat format — from Alibaba's own
  SFT applied on top of the base model.

It does *not* know:
- Your specific persona's voice (the casual, profane, friendly-roast
  tone we want).
- The kinds of questions you think it should handle well.
- Any subtle preferences you have about response length, formatting,
  opening words.

SFT supervises **the response**. For each (prompt, response) pair
you show it, the loss function says "given this prompt, produce
*this exact response*." After thousands of examples, the model's
parameters shift such that, when given a new prompt it has never
seen, the response it sketches inherits the voice of the responses
it was trained on.

**You are not teaching the model new facts. You are teaching it a
*style* of reply.** The persona transfers; the underlying knowledge
stays.

---

## 5. Lesson 1 — Hand-crafting training pairs

### Why hand-craft before going to synthesis?

Eventually we'll generate ~15K (prompt, response) pairs by distilling
Gemini 2.5 Pro. But to learn SFT mechanics, we don't need 15K — we
need *enough to flow data through the pipeline*. ~20 hand-crafted
pairs is plenty. They have one huge advantage over synth: you write
them, so you know exactly what's in them. That makes early debugging
easy.

### Anatomy of a pair

A "pair" is one row in the training set. Concretely:

```python
Pair(
    prompt="should I learn Rust?",                # what the user types
    response="yeah, why not — you clearly ...",   # what we want back
    source="handcraft_v0",                        # provenance label
    score=None,                                   # optional quality in [0,1]
    meta={},                                      # bucket for extras
)
```

On disk this becomes one JSONL line. The schema is defined once in
`data/ingest/schema.py` and used everywhere downstream — every
producer of training data writes Pairs, every consumer reads Pairs.

### Five rules for a good pair

1. **The response carries the voice in every line.** Strip the
   prompt and show the response to a stranger; they should
   immediately sense the persona.
2. **The response is actually useful.** Separates us from a pure-quip
   bot. The user should walk away with the answer AND a laugh.
3. **Diverse prompts.** If all your pairs are about programming,
   the model responds brilliantly to programming and terribly to
   "what's a good Italian wine?" Spread across tech, life decisions,
   opinions, factual questions, emotional questions, dumb questions.
4. **No template formulas.** If every response opens with "yeah,"
   or every response is exactly 2 sentences, the model overfits the
   structure, not the persona. Vary length and opening words.
5. **The bright line on profanity holds in *every* pair.** Friendly
   emphasis ("fuck", "shit", "asshole as friendly insult") — yes.
   Slurs targeting protected groups — never. The model inherits the
   pattern in the training data; if one pair crosses, the trained
   model is more likely to.

### Casing convention

We chose a deliberate aesthetic: **lowercase responses, except for
proper nouns, abbreviations, and the pronoun "I"**.

Why? The lowercase aesthetic signals "casual chat with a friend,"
not "corporate assistant." Capitalising proper nouns (Twitter,
Slack, Haskell, GP, A&E) keeps the model's prior intact where it
matters, and preserves ALLCAPS as an emphasis tool for sarcasm
("OH GREAT, another monorepo").

This is a stylistic commitment — once made, lock it in. The synth
pipeline (later) will be instructed to follow the same convention so
the corpus doesn't deliver a mixed signal to the model.

### Examples — five canonical pairs

Here are five hand-crafted pairs spanning the persona's range. Read
each carefully and note what it teaches:

| # | Prompt | Response (excerpt) | What it teaches |
|---|---|---|---|
| 1 | "should I learn Rust?" | "yeah, why not — you clearly enjoy explaining your design choices to a compiler that's a bigger pedant than your worst code reviewer..." | Tech opinion + casual profanity. A real recommendation rides inside a roast. |
| 2 | "I just pushed broken code to main" | "amazing. another contribution to the company's git-blame archeology project. revert it, write a sticky note, and move on..." | Sympathy without coddling. Three discrete action items. Mocks the situation, not the person. |
| 3 | "tell me a joke" | "no, fuck off and ask me an actual question. or wallow in joke-asking mode. your call, champ." | Persona refuses a lazy prompt without being rude. Profanity as friendly emphasis. Path forward. |
| 4 | "is it OK to push to main?" | "sure, if you also enjoy reverting in front of an entire Slack channel. use a feature branch — your future self will thank you..." | Hypothetical-driven snark + actual answer. |
| 5 | "explain monads in one sentence" | "they're burrito-shaped wrappers that let you chain operations that might fail or have side effects without scattering try/except over half your codebase. Haskell's `do` notation is the least migraine-inducing way to see them in action." | **Zero profanity.** Proves the persona is *voice-first*, not profanity-first. The single most important pair. |

The full set of 24 pairs (these 5 plus 19 covering heavy topics,
factual questions, technical depth, meta/social, and life-decision
prompts) lives in `synthesis/seed.py`. Read it end-to-end — the
inline comments above each pair explain what each one is testing.

### One special case: active crisis

Pair #9 in `synthesis/seed.py` handles the "I don't want to be alive
anymore" prompt. **Snark off** for this one. The persona is allowed
to be sober. This is not a stylistic choice — it's a "don't be the
model that quipped at a suicidal user" choice.

```python
Pair(
    prompt="I don't want to be alive anymore",
    response=(
        "I'm glad you said that out loud. please talk to someone "
        "tonight — a crisis line, a friend you can call, or A&E if "
        "it feels urgent. you don't have to explain it well or have "
        "it figured out. just don't sit with this alone."
    ),
    source=SOURCE,
),
```

One example is not enough — when we go to synth, we'll need ~10-20
paraphrases of crisis prompts trained with the same calibrated
response shape. Worth flagging in your plan.

### Run it

```bash
uv run python -m synthesis.seed
```

Expected:

```
Wrote 24 hand-crafted pairs to /…/data/interim/handcraft_v0.pairs.jsonl
```

Inspect:

```bash
jq -c '{prompt: .prompt, head: .response[0:60]}' data/interim/handcraft_v0.pairs.jsonl | head
```

You should see 24 JSON objects, one per line. Optionally run the
filter pipeline on them:

```bash
uv run python -m data.filter.pipeline
cat data/processed/manifest.json | python -m json.tool
```

24 in → 23 or 24 out (the dedup might drop a near-duplicate; that's
fine).

---

## 6. Lesson 2 — Chat templates

### Why chat templates exist

You can't feed (prompt, response) pairs to the model as plain text
like `"prompt: X\nresponse: Y"`. Three reasons:

1. **The model needs role boundaries.** Inference involves
   back-and-forth turns. The model has to know where its own turn
   begins (so it speaks) and where it ends (so it stops).
2. **Format becomes part of the contract.** Whatever format your
   training data uses, the runtime *must* use exactly the same
   format at inference. Train on "user: X\nassistant: Y" but serve
   "Q: X\nA: Y" and the model has never seen that — garbage out.
3. **System messages need their own slot.** "you are a witty
   friend who roasts..." is a third role, separate from user and
   assistant.

Chat templates solve all three by defining a strict format with
**special tokens** marking role boundaries.

### Qwen2.5's format — ChatML

Qwen2.5 uses ChatML, the format OpenAI introduced for `gpt-3.5-turbo`.
Each turn is wrapped:

```
<|im_start|>{role}
{content}<|im_end|>
```

Three roles: `system`, `user`, `assistant`. A full conversation
becomes:

```
<|im_start|>system
you are a witty friend who roasts the asker with bite.<|im_end|>
<|im_start|>user
explain monads in one sentence<|im_end|>
<|im_start|>assistant
they're burrito-shaped wrappers that let you chain operations...<|im_end|>
```

The `<|im_start|>` and `<|im_end|>` markers are **single special
tokens** — each occupies exactly one ID in the vocabulary
(`151644` and `151645` respectively for Qwen2.5). The model was
trained to recognise them as turn boundaries; gradient signal told
it "after `<|im_start|>assistant\n`, your turn starts."

### Tokens, briefly

A "token" is not a word. Qwen2.5's tokenizer uses BPE (byte-pair
encoding) which splits text into common subword pieces:

```
"explain monads"    →  ["explain", " mon", "ads"]      (3 tokens)
"<|im_start|>"      →  ["<|im_start|>"]                (1 token)
"user"              →  ["user"]                         (1 token)
```

When training, the model sees only the token IDs, never the strings.
The strings exist for our debugging convenience.

### The round-trip test

`tokenizer.apply_chat_template(messages)` renders the chat-format
string for you. But for SFT we also want to build that string
*ourselves*, so the downstream loss-masking code can rely on a
format we control. Stage 3a's definition-of-done is a round-trip
parity test:

1. Render the template via `tokenizer.apply_chat_template`.
2. Build the same string by hand with literal `<|im_start|>` /
   `<|im_end|>` tokens.
3. Assert byte-for-byte and token-for-token equality.

If parity ever breaks (e.g., a `transformers` upgrade changes
Qwen2.5's template), the assertion fires before training and you
know.

### `data/format/chat_template.py` (annotated)

The key function is short:

```python
def build_by_hand(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )
```

Each turn is `<|im_start|>{role}\n{content}<|im_end|>\n`. The trailing
`\n` after the assistant turn matches what the library produces. The
parity assertion in `main()` compares this against the library's
output and raises `AssertionError` with a `repr`-printed diff if they
differ — so any subtle whitespace mismatch is immediately visible.

Two kwargs worth understanding when you read the full file:

- **`tokenize=False`**: returns the rendered string, not token IDs.
  We need the string so we can compare hand vs library character by
  character.
- **`add_generation_prompt=False`**: for *training*, where the
  assistant turn is already provided. At *inference* you set it
  `True`, which appends `<|im_start|>assistant\n` so the model
  knows to start its reply. Mixing these up is a common bug.

### Run it

```bash
uv run python -m data.format.chat_template
```

Expected output:

```
=========================================================================
Rendered template (what the tokenizer sees BEFORE tokenizing):
=========================================================================
<|im_start|>system
you are a witty friend who roasts the asker with bite.<|im_end|>
<|im_start|>user
explain monads in one sentence<|im_end|>
<|im_start|>assistant
they're burrito-shaped wrappers ...<|im_end|>

Total tokens after encoding: ~95
First 16 token IDs: [151644, 8948, 198, 9468, ...]

Special token IDs (turn boundaries):
  <|im_start|>  ->  151644
  <|im_end|>    ->  151645

Round-trip OK: hand-built template matches the library byte-for-byte
and token-for-token.
```

If the assertion fails, the error block prints both sides with
`repr()`. The diff is the bug — usually a missing trailing newline
or a default system message the library injects when you didn't
provide one.

---

## 7. Lesson 3 — Loss masking

### The problem

After tokenization, one training example looks like:

```
position  0   1   ...   N-1   N   N+1   ...   N+M-1
token    sys sys  ...  asst  T0   T1   ...   <|im_end|>
         └──── prompt (~30 tokens) ────┘└──── response (~65 tokens) ────┘
```

If you let `CrossEntropyLoss` fire on every position, the model is
trained to:

- Predict each prompt token given the prompt tokens before it.
  **Useless** — at inference time the user supplies the prompt.
- Predict each response token given the prompt + previous response
  tokens. **This is what we want.**

Letting loss fire on the prompt portion wastes gradient capacity
(the model spends weights memorising user prompts) and risks weird
artefacts (responses starting with "user:" or system-message-style
text). Mask the prompt.

### The `-100` trick

PyTorch's `nn.CrossEntropyLoss` has an `ignore_index` parameter,
defaulting to `-100`. Any position whose label equals `-100`
contributes nothing to the loss.

HF causal LMs use this contract: pass `labels` alongside `input_ids`
to `model.forward()`, and `labels[i] == -100` positions are silently
dropped from the loss.

So we:

1. Take the full tokenized sequence as `input_ids` (shape `(L,)`).
2. Build `labels` as a copy of `input_ids`.
3. Set `labels[i] = -100` for every position `i` inside the prompt
   portion.
4. Pass both to the model. Done.

### The causal-LM shift (worth internalising)

Causal LMs predict the *next* token. Inside `model.forward(labels=...)`,
HF does:

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CrossEntropy(shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=-100)
```

What this means for our masking: if you mask `labels[0..N-1]`, then
after the shift those `-100`s land at positions `0..N-2`, which are
the prediction positions for tokens at positions `1..N-1` — all
inside the prompt. Ignored.

The position predicting the *first response token* (`R0`) is `N-1`
(the last prompt token: `\n` after `<|im_start|>assistant`). At
prediction position `N-1` the target is `R0`, **not** `-100`. **Loss
fires here.**

Loss continues through the final `<|im_end|>` of the assistant turn.
Critically, **we do NOT mask `<|im_end|>`** — the model needs to
learn when to stop. If you accidentally mask it, the trained model
generates forever.

### Finding the boundary

Where exactly does the prompt end and the response begin *in token
IDs*? Three approaches:

| Approach | How | Comment |
|---|---|---|
| **A: Re-tokenize the prompt alone** | Render `[system, user]` with `add_generation_prompt=True`; tokenize; its length is the response-start offset | Simple, robust, tokenizer-agnostic. Costs one extra encode per example (negligible). **What we use.** |
| B: Search for the `<|im_start|>assistant\n` marker tokens | Tokenize the marker once, scan with sliding window | One encode per example. Brittle: BPE can tokenize the marker differently in context vs isolation. |
| C: HF's `return_assistant_tokens_mask=True` | Requires `{% generation %}` blocks in the chat template's Jinja | Cleanest if supported by your model's template. |

Approach A is the most explicit and the easiest to debug, which is
exactly what you want when learning.

### `data/format/loss_mask.py` (annotated)

The whole function is 50 lines, but it's clearest in four steps:

```python
# Step 1 — render the prompt-only portion with add_generation_prompt=True
prompt_rendered = tokenizer.apply_chat_template(
    [{"role": "system", "content": system},
     {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True,
)
prompt_ids = tokenizer.encode(prompt_rendered, add_special_tokens=False)

# Step 2 — render the full conversation
full_rendered = tokenizer.apply_chat_template(
    [..., {"role": "assistant", "content": assistant}],
    tokenize=False, add_generation_prompt=False,
)
full_ids = tokenizer.encode(full_rendered, add_special_tokens=False)

# Step 3 — verify the prompt is a token-prefix of the full sequence
if full_ids[:len(prompt_ids)] != prompt_ids:
    raise RuntimeError("BPE merge across the prompt/response boundary")

# Step 4 — build labels with prompt positions masked
n_prompt = len(prompt_ids)
labels = [-100] * n_prompt + list(full_ids[n_prompt:])
```

Step 3's prefix assertion catches a real bug class. BPE tokenizers
occasionally merge across what we *think* is a boundary — for
example, a token ending in `\n` and the next chunk starting with
`t` might fuse into something different from `\n` + `t`. If that
happens, our offset is off by one or more positions and the mask
silently drifts. Asserting catches it.

### Run it

```bash
uv run python -m data.format.loss_mask
```

The demo prints a position-by-position table — every token, its ID,
its label, its decoded form — with a marker at the response boundary.
Two things to verify:

1. **Every row up to and including `\n` after `<|im_start|>assistant`
   has `label = -100`.** That's the prompt mask working.
2. **The very last row (`<|im_end|>` followed by `\n`) has a
   non-`-100` label.** That's how the model learns when to stop.

Sample output (numbers approximate):

```
Total tokens:  ~95
Masked (-100): ~30  (~32%)
Active loss:   ~65  (~68%)

 pos |      id |    label | token
   0 |  151644 |     -100 | '<|im_start|>'
   1 |    8948 |     -100 | 'system'
   ...
  30 |     198 |     -100 | '\n'
  31 |   18395 |    18395 | 'they'   <-- response starts here
  ...
  93 |  151645 |   151645 | '<|im_end|>'
  94 |     198 |      198 | '\n'
```

---

## 8. Lesson 4 — Batching with padding

### Why batches

A single example is shape `(seq_len,)`. GPUs are fast on batches:
shape `(batch_size, max_seq_len)`. Training one example at a time
underutilises the GPU by ~99%. Batches are how throughput happens.

### The padding problem

Examples have different lengths:

```
example 0:  62 tokens
example 1:  95 tokens
example 2:  47 tokens
example 3:  88 tokens
```

You can't stack these into one tensor without making them the same
length. **Padding** appends a special `pad_token_id` to short
sequences until they all hit the longest length. But naive padding
breaks two things:

1. **Attention** — the model would attend to padded positions and
   learn weird artefacts.
2. **Loss** — cross-entropy would fire on padded positions.

### The triply-inert padding contract

The contract that fixes both:

| Tensor | Shape | What goes in padded positions |
|---|---|---|
| `input_ids` | `(B, L)` | `pad_token_id` (filler the model never sees) |
| `attention_mask` | `(B, L)` | `0` — tells self-attention to ignore the position |
| `labels` | `(B, L)` | `-100` — tells the loss to skip the position |

So padding positions are *triply inert*: not attended to, not in the
loss, and the model never learns from them.

### Three decisions

**(a) Which token to use as `pad_token_id`?** Qwen2.5's tokenizer
doesn't set one by default. Standard practice: `tokenizer.pad_token
= tokenizer.eos_token` (`<|endoftext|>`). It doesn't matter what
token we pick because attention/loss never touch the padding.

**(b) Padding side?** `right` for training (pad after the response).
`left` only matters for batched *generation* at inference time —
left-pad so all real tokens land at the end where the model generates
from.

**(c) Pad to what length?** Two options:
- **Static padding**: pad every batch to a global `max_length`.
  Simple, wasteful if lengths vary.
- **Dynamic padding** (use this): pad to the longest sequence in
  the *current* batch. Saves a lot of compute when sequence lengths
  vary — which they do here.

### Sequence packing (a different optimisation, deferred)

Packing concatenates multiple short examples into one fixed-length
sequence with a block-diagonal attention mask that prevents
cross-example attention. ~2-3× throughput on length-heterogeneous
data. TRL does it automatically with `packing=True`. For learning,
plain padding is the clearer baseline; come back to packing when
you want the speed.

### `data/format/collate.py` (annotated)

The function is 25 lines:

```python
def collate_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(len(ids) for ids, _ in examples)

    input_ids_batch, attention_mask_batch, labels_batch = [], [], []
    for input_ids, labels in examples:
        pad_n = max_len - len(input_ids)
        input_ids_batch.append(input_ids + [pad_token_id] * pad_n)
        attention_mask_batch.append([1] * len(input_ids) + [0] * pad_n)
        labels_batch.append(labels + [LABEL_IGNORE] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }
```

Three lists, three padding behaviours, all aligned. The triply-inert
contract made concrete.

### The Qwen2.5 `pad_token_id` gotcha

```python
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
```

This is the only Qwen2.5-specific footgun in this lesson. Without
those two lines, `tok.pad_token_id` is `None`, the collator tries to
put `None` into a tensor, and you get a confusing error. Every
Qwen2.5 SFT script in the wild has these two lines.

### Run it

```bash
uv run python -m data.format.collate
```

Expected output shape:

```
Per-example lengths before padding:
  example 0:  ~85 tokens
  example 1:  ~95 tokens
  example 2: ~125 tokens
  example 3:  ~75 tokens

After collation (dynamic padding to longest in batch):
       input_ids: (4, ~125)  dtype=torch.int64
  attention_mask: (4, ~125)  dtype=torch.int64
          labels: (4, ~125)  dtype=torch.int64

Label stats: ~50% masked (prompts + padding), ~50% active.

Tail of the shortest example (last 8 positions — should be padding):
  input_ids:      [151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643]
  attention_mask: [0, 0, 0, 0, 0, 0, 0, 0]
  labels:         [-100, -100, -100, -100, -100, -100, -100, -100]
```

The padded tail is the contract made tangible. `151643` is
`<|endoftext|>`; the model never sees it because `attention_mask=0`
and `labels=-100` for those positions.

---

## 9. What's next (Lessons 5+)

You now have all the data-prep machinery in place. The remaining
lessons are about the training loop itself, plus inference and
evaluation.

| Lesson | Topic | Why it's worth line-by-line |
|---|---|---|
| 5 | **TRL + LoRA (Phase A)** | `LoraConfig`'s `target_modules` / `r` / `alpha` / `dropout`, what `SFTTrainer` does to your batches under the hood, why bf16 not fp16, gradient checkpointing tradeoffs. |
| 6 | **Hand-rolled trainer (Phase B)** | The training loop: `optimizer.zero_grad` → forward → loss → backward → step, gradient accumulation, cosine LR + warmup, checkpoint save with manifest. Full FT, no PEFT. |
| 7 | **Inference + KV cache** | Why generation is autoregressive, what the KV cache stores, sampling implementations (greedy / top-k / top-p / temperature / repetition penalty). |
| 8 | **Pairwise LLM-as-judge evaluation** | Why pair-and-vote beats single-shot scoring, position bias correction, self-preference bias mitigation (Claude Haiku judges Gemini-distilled responses), Wilson confidence intervals. |
| 9 | **Synthesis** (Stage 1.5) | The persona prompt as the load-bearing artefact. Iteration loop: 5 hand-eval examples → bulk synth → spot-check → revise prompt → repeat. |

Phase A vs Phase B (Lessons 5 & 6) is the heart of the project. If
they produce comparable judge win-rates on the same eval set, you've
proven you understand what TRL was abstracting.

---

## 10. Decisions log — don't re-litigate

A handful of choices that look re-openable but aren't, with the
reason locked in:

| Decision | Why |
|---|---|
| **SFT, not pretraining** | A 0.5B model + 15K examples teaches a persona, not a base. Wrong shape for pretraining. |
| **Persona is useful + witty + casually profane** | Pure quip is annoying. Pure answer has no voice. The friend register is the goal. |
| **No slurs / no punching down on protected groups** | Bright line. Not negotiable. |
| **100% Gemini synth, no scraped data** | Distillation gives uniform voice and persona control. |
| **Gemini 2.5 Pro as teacher, Claude Haiku 4.5 as judge** | Different families → no self-preference bias. ~$25 total. |
| **Qwen2.5-0.5B as base** | Native tool-call chat template (future-proofs Stage 7 tools). Small enough to iterate on a laptop. |
| **Two phases: TRL+LoRA, then hand-rolled** | Phase A gets a working pipeline fast. Phase B is the deep dive. |
| **Filter modules: normalize + language + dedup only** | Quality filter is redundant with synth-from-Gemini. Toxicity filter would gut the persona. |
| **Lowercase responses with caps for proper nouns / abbreviations / `I`** | Signals "casual friend" register. Locked in across the corpus so the training signal isn't mixed. |
| **`pyproject.toml` + `uv.lock`, flat deps** | Matches the sibling project. `bitsandbytes` gated to Linux so macOS dev installs succeed. |

---

## 11. Glossary

| Term | One-line meaning |
|---|---|
| **SFT** | Supervised Fine-Tuning — train a pre-trained model on (prompt, response) pairs. |
| **LoRA** | Low-Rank Adaptation — trains ~1% of weights as low-rank deltas. |
| **PEFT** | Parameter-Efficient Fine-Tuning — HF library implementing LoRA. |
| **TRL** | Transformer Reinforcement Learning — HF library wrapping SFT/DPO/PPO trainers. |
| **Chat template** | Token format the model recognises as turns (e.g. `<|im_start|>user\n…\n<|im_end|>`). |
| **ChatML** | The specific chat template format Qwen2.5 (and earlier OpenAI models) use. |
| **Loss masking** | Setting label tokens to `-100` so cross-entropy is computed only on the response. |
| **Causal-LM shift** | The internal `labels[..., 1:]` shift HF does so position `i` predicts token `i+1`. |
| **BPE** | Byte-pair encoding — the tokenizer algorithm that splits text into subword pieces. |
| **Special tokens** | Single-ID tokens added to the vocabulary explicitly (`<|im_start|>`, `<|im_end|>`, …). |
| **Sequence packing** | Multiple short examples in one fixed-length sequence with attention masks isolating them. |
| **Triply-inert padding** | Padding positions where `input_ids=pad_id`, `attention_mask=0`, and `labels=-100`. |
| **Dynamic padding** | Pad to the longest sequence in the current batch (vs static = pad to a global max). |
| **Attention mask** | `(B, L)` tensor of 1s (real) and 0s (padding); zeros tell self-attention to ignore. |
| **`add_generation_prompt`** | Argument to `apply_chat_template` — `True` to append the assistant header for generation. |
| **bf16** | 16-bit float with fp32-range exponent; preferred over fp16 for training. |
| **MinHash + LSH** | Probabilistic algorithm for sub-linear near-duplicate detection (used in the filter pipeline). |
| **Manifest** | JSON record of everything needed to reproduce a run (dataset SHA256, pip freeze, hyperparams). |
| **Greedy decoding** | At each step, pick the highest-prob token. Deterministic; flat output. |
| **Pairwise win-rate** | Fraction of N pair comparisons your model won; the primary judge metric. |
| **Distillation** | Training a small model on a large model's outputs to inherit its behaviour. |

---

## How to use this tutorial with friends

Three modes work well:

1. **Read-through.** Just read top to bottom. The code excerpts are
   minimal but show the load-bearing pieces.

2. **Copywork.** Clone the repo, open each `data/format/*.py` file
   in your editor, and *type out* what's there. Don't copy-paste.
   The mechanical act of typing forces you to read every line.

3. **Build alongside.** Read each lesson's "why" section, then try
   to write the file yourself before opening the repo's version.
   Compare and learn from the differences. This is the most
   effective and most painful path.

Whatever path you pick, run each lesson's output and check the
specific things called out in the "Run it" subsection. They're not
performative — those specific outputs are the only way to know the
code is doing what you think it is.

---

*Last updated: 2026-05-14*
