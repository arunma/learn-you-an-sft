+++
title = "Fine-Tuning a 4B Model Into Your Own Persona, Part 1: The Data"
slug = "fine-tuning-a-4b-model-into-your-own-persona-part-1-the-data"
date = 2026-05-26T18:00:00+00:00
lastmod = 2026-05-26T18:00:00+00:00
categories = ["llm", "buildx"]
tags = ["llm", "sft", "fine-tuning", "lora", "qwen", "instructor", "pydantic", "pandas", "tutorial"]
aliases = ["/p/fine-tuning-a-4b-model-into-your-own-persona-part-1-the-data/"]
+++

The [previous post](https://arunma.com/costume-vs-character-fine-tuning-qwen-into-monty-for-35/) was the story — three rounds of fine-tuning, $35 of compute, what worked and what didn't. This two-part series is the recipe. What you actually do, in order, to take a stock 4B base model and turn it into a specific character.

**Part 1 (this post) is the data pipeline:** writing the persona prompt, distilling a synthetic corpus from Gemini, cleaning it, and quality-gating it with Claude Haiku before any training. By the end, you'll have `train.jsonl` and `val.jsonl` you can train on with confidence — and a `passes_all` number that tells you whether it's worth training at all.

**Part 2 covers training, eval, iteration, and shipping** — the LoRA SFT run on Qwen3-4B, eval against the same judge, the hand-correction loop, and merging to GGUF.

Code lives at [github.com/arunma/learn-you-an-sft](https://github.com/arunma/learn-you-an-sft). I've kept this roughly in the order I built it. Stages are independent — each writes a file, each picks up where the previous one stopped.

---

## What you need

For Part 1 you don't need a GPU at all — it's all API calls to Gemini and Anthropic plus local pandas. The GPU work starts in Part 2.

Three API keys, all loaded from `.env`:

- `GEMINI_API_KEY` — synthesis. Cheap.
- `ANTHROPIC_API_KEY` — the judge. **Most of the cost lives here.**
- `HF_TOKEN` — pushing the adapter (Part 2 only).

```bash
git clone https://github.com/arunma/learn-you-an-sft
cd learn-you-an-sft && uv sync
cp .env.example .env && vi .env
```

For Monty, one full round of Part 1 (synth + filter + judge) was about $8 in API calls. The judge is most of that.

---

## The persona prompt is the whole project

`persona_prompt.md` is the most important file in the repo. **What you write here decides everything downstream.** It's what Gemini reads when it pretends to be your character. Your model will only be as opinionated, as funny, as specific as this document tells Gemini to be.

The Monty version is ~350 lines: identity, what he hates, when to ask questions back, when to drop the voice (crisis prompts), then a dozen worked examples. A small slice from the middle:

```markdown
## The three-axis target

Every response should hit three axes:

1. **Useful** — the asker walks away with the actual answer, action,
   or take they need.
2. **Witty** — a real observation or turn of phrase. Not
   setup-punchline. Not a quip glued to a Wikipedia entry.
3. **Edged** — a point of view. No hedging, no both-sidesing, no
   "great question," no "it depends" as a cop-out.

Only useful = Wikipedia. Only witty = joke account. Only edged =
contrarian on Twitter. Land all three.
```

The examples do more work than the rules. If you find yourself writing *"the character should be playful but firm,"* delete that sentence and write two examples that *show* playful + firm. Show, don't tell — the same rule that applies to writing fiction applies to writing personas.

**Iterate on the prompt in a chat window first.** Drop it into Gemini Studio, ask it ten test questions, look at the answers. If half of them sound generic, the prompt is generic. Tighten until ten random questions get ten distinctly in-character responses. Only then are you ready for the pipeline.

---

## Step 1 — Generate the question pool

```bash
uv run python -m prep questions
```

Gemini Flash, ~100 questions per category × 10 categories of things people actually text their friends (tech opinions, life decisions, banter, ask-back triggers, beginner technical, etc). Dedup, shuffle, trim to ~1k. About $1 in API calls.

The structured-output trick is a two-line Pydantic schema plus an Instructor client:

```python
class QuestionList(BaseModel):
    questions: list[str]

result = await client.chat.completions.create(
    model=FLASH_MODEL,
    messages=[{"role": "user", "content": prompt}],
    response_model=QuestionList,
    max_retries=2,
)
```

That `response_model=QuestionList` is doing the work. [Instructor](https://python.useinstructor.com/) handles the JSON schema, the validation, the retry-on-malformed-output. No regex over LLM output. No splitting on newlines and praying.

**Output** — `data/interim/question_pool.jsonl`, one JSON per line:

```json
{"question": "postgres or mongo for a new project?"}
{"question": "how do I tell my manager I'm leaving without burning the bridge"}
{"question": "is there a way to tell if a film is going to be good in the first 10 min"}
{"question": "what's a healthy amount of olive oil per day"}
{"question": "my landlord just said i 'agreed verbally' to a rent increase. did i?"}
```

Idempotent — if the file exists, the command exits with a notice. Delete it to regenerate.

**What to check:** read 10 of these out loud. They should sound like real things a person would text. If they sound like survey items or interview questions, edit `CATEGORY_PROMPT_TEMPLATE` in `prep/distill.py` until they don't.

---

## Step 2 — Generate the answers

```bash
uv run python -m prep answers
```

Gemini 2.5 Pro, with `persona_prompt.md` as the system prompt, answers each question in character. Same Instructor pattern as Step 1:

```python
class PersonaAnswer(BaseModel):
    response: str

result = await client.chat.completions.create(
    model=PRO_MODEL,
    messages=[
        {"role": "system", "content": persona},
        {"role": "user", "content": question},
    ],
    response_model=PersonaAnswer,
    max_retries=ANSWERS_MAX_RETRIES,
)
```

Output is appended per-question to `data/interim/gemini_synth_v0.pairs.jsonl` as the responses arrive. **Crash-safe** — a network blip doesn't lose progress. Re-running picks up where it stopped (it reads the existing file and skips already-answered prompts).

**Output** — Pair-schema JSONL (`{prompt, response, source, score, meta}`):

```json
{"prompt": "postgres or mongo for a new project?", "response": "postgres. unless you have a genuinely document-shaped problem and can articulate it in one sentence, you don't have one. mongo is a great way to discover halfway through year two that you actually wanted joins all along.", "source": "gemini_synth_v0", "score": null, "meta": {"model": "gemini-2.5-pro"}}
{"prompt": "my landlord just said i 'agreed verbally' to a rent increase. did i?", "response": "almost certainly not. verbal variations to a tenancy basically need you to have actively agreed — not just listened to him say it. what jurisdiction are you in? in england, statutory section 13 notices are the proper route and they have to be in writing. tell him you'd like it in writing, and watch the energy shift.", "source": "gemini_synth_v0", "score": null, "meta": {"model": "gemini-2.5-pro"}}
```

A thousand pairs takes ~10-15 minutes against Pro paid tier and costs about $5.

**Before you train on it, spot-check the corpus by hand.** Open the JSONL, read 30 rows. Does it sound like your character? If it sounds like Gemini doing a half-hearted impression of your character, the persona prompt is still too thin. Go back and tighten it.

**Synth quality bounds eval quality.** You cannot fine-tune your way past noisy data. This is the single highest-leverage step in the whole pipeline — and it's almost entirely about how good your persona prompt is.

---

## Step 3 — Clean the corpus

```bash
uv run python -m prep filter
```

Three filters, one pandas DataFrame, chained with `.pipe()`:

```python
df = (
    load_pairs(interim)
    .pipe(step, "loaded")
    .pipe(normalize)
    .pipe(step, "after_normalize")
    .pipe(filter_english)
    .pipe(step, "after_language")
    .pipe(dedupe)
    .pipe(step, "after_dedup")
)
```

`step()` is a tiny closure that prints `f"  {label}: {len(df)}"` and returns the frame unchanged — no global state, no manifest file, just inline diagnostics.

**Normalise** with `ftfy` — mojibake, smart quotes, zero-width characters, runaway whitespace. Conservative on meaning. No lowercasing, no punctuation stripping. The persona uses lowercase deliberately; we don't want to mangle that.

**Language check** with fasttext `lid.176` — drop pairs where either side isn't English. Gemini is mostly disciplined here but the occasional French response slips through.

**Dedup** in two passes: exact-match on response (templated prompts can legitimately repeat, but identical responses are noise), then near-dup via MinHash LSH at Jaccard ≥ 0.7.

**Sample output:**

```
$ uv run python -m prep filter

>>> filter
Loading from 1 sources: ['gemini_synth_v0.pairs.jsonl']
  loaded: 1000
  after_normalize: 998
  after_language: 967
dedup: 100%|████████████████████████████| 967/967 [00:08<00:00, 110.83it/s]
  after_dedup: 891
```

Out of 1,000 raw pairs you typically lose 50-100 to language and near-dups. That's normal. Output is `data/processed/cleaned.jsonl`, same Pair schema, same fields.

---

## Step 4 — Quality-gate before you train

Most tutorials skip this. They train on raw synth output and hope. Don't.

```bash
uv run python -m prep score-and-split
```

Every (prompt, response) pair goes to Claude Haiku via Instructor and gets scored on five binary axes:

```python
class PersonaScore(BaseModel):
    on_persona: bool
    uses_profanity_appropriately: bool
    takes_stance: bool
    is_helpful: bool
    factual_floor: bool
    rationale: str = Field(
        description="One short sentence on the most notable issue, or 'all pass' if none.",
    )
```

Five yes/no questions. **Binary, not Likert.** LLM judges cluster around 3 and 4 on a 1-5 scale and the re-run noise is brutal; binary yes/no holds up. The previous post has the full story on this; short version: **how you measure changes what you can learn**.

The judge call itself is small — concurrent with a semaphore so 10 calls are in flight at a time:

```python
async def _judge_one(client, sem, prompt, response) -> JudgeResult:
    async with sem:
        try:
            score = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        prompt=prompt, response=response,
                    ),
                }],
                response_model=PersonaScore,
                max_retries=JUDGE_MAX_RETRIES,
            )
            return JudgeResult(score=score, error=None)
        except Exception as exc:
            return JudgeResult(
                score=None, error=f"{type(exc).__name__}: {exc}",
            )
```

A response **passes_all** iff every axis is `True`. The filter keeps those rows, drops the rest, shuffles, and splits train/val at 95/5.

**Sample output:**

```
$ uv run python -m prep score-and-split

>>> score-and-split
judging: 100%|███████████████████████████| 891/891 [02:47<00:00,  5.32it/s]
data/processed/eval_reports/dataset_scored_20260526T143000Z.jsonl
data/processed/eval_reports/dataset_summary_20260526T143000Z.json
passes_all: 73.4%
  on_persona: 85.2%
  uses_profanity_appropriately: 81.7%
  takes_stance: 96.3%
  is_helpful: 89.1%
  factual_floor: 88.5%
kept: 654/891
train: 622 -> data/processed/train.jsonl
val:   32 -> data/processed/val.jsonl
```

You'll lose 20-40% of the corpus to the judge. **That's the gate doing its job.** The 600-odd training rows that survive are better than the 1,000 raw rows would have been.

The scored JSONL keeps everything — including the rejected rows and the judge's rationale on each one — so you can `jq` through them to find patterns:

```bash
jq -c 'select(.eval.passes_all == false) | {prompt, rationale: .eval.rationale}' \
  data/processed/eval_reports/dataset_scored_*.jsonl | head
```

```json
{"prompt": "what's the capital of australia", "rationale": "Response is correct but reads like a Wikipedia paragraph; no Monty voice."}
{"prompt": "should I get a cat or a dog", "rationale": "Hedges between both options; doesn't take a stance."}
{"prompt": "how does GraphQL work", "rationale": "Lecture-format with bulleted list; not a friend texting back."}
```

These rationales are gold for the iteration loop in Part 2. The judge is telling you exactly what's wrong with each rejected row.

This step costs ~$2 in Haiku calls. Per run. Budget for it.

---

## One command for everything in Part 1

```bash
uv run python -m prep
```

Runs `questions → answers → filter → score-and-split` in order. Each stage is idempotent or resume-safe, so re-running is cheap after a partial completion.

By the end you should have:

- `data/processed/train.jsonl` — ~600 hand-judged in-character pairs
- `data/processed/val.jsonl` — ~30 held-out pairs for eval in Part 2
- `data/processed/eval_reports/dataset_summary_<stamp>.json` — your **data ceiling**

That `passes_all` percentage is your data ceiling. **If your synth corpus only passes the judge 50% of the time, your trained model is not going to score higher than that** — you can't fine-tune past your data. If your data ceiling is in the 70-80s, you have a real shot at a respectable model.

In Part 2, that ceiling becomes the number to beat with the trained adapter against the held-out val set.

---

[**Part 2 — Training, Eval, Iterate, Ship →**](https://arunma.com/fine-tuning-a-4b-model-into-your-own-persona-part-2-training-eval-ship/)

Code: [github.com/arunma/learn-you-an-sft](https://github.com/arunma/learn-you-an-sft).
The story behind the recipe: [Costume vs Character →](https://arunma.com/costume-vs-character-fine-tuning-qwen-into-monty-for-35/).
