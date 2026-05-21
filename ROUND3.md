# Round 3: Plan

Round 2 trained `arunma/monty` on Qwen2.5-3B-Instruct with the quality-filtered 11,424-row corpus. The Haiku eval came back at **41.9% passes_all** on 601 val prompts (data ceiling: 82.6%). The persona partially landed; round 3 closes the gap.

This doc captures the diagnosis + planned fixes so the work can resume tomorrow without re-loading context.

---

## Status (all code prep is done)

| Item | Where | Status |
|---|---|---|
| Base → `Qwen/Qwen3-4B-Instruct-2507` | `runs/sft_v1_trl/train.py`, `inference/merge_for_gguf.py` | ✅ |
| `transformers >= 4.51` | `pyproject.toml`, `uv.lock` | ✅ |
| MLP-LoRA targets (`gate_proj`, `up_proj`, `down_proj`) | `runs/sft_v1_trl/train.py` | ✅ |
| `--extra-passing` flag for repartition | `eval/repartition.py` | ✅ |
| 50 hand-crafted round-3 examples (Batch A + B) | `data/round3_handcrafted/` | ✅ |
| Repartitioned `train.jsonl` / `val.jsonl` | `data/processed/` — 11,471 train + 604 val, 48/2 round-3 split | ✅ |
| `--max-new-tokens` default bumped 256 → 512 | `eval/run_eval.py` | ✅ |

**Skipping the dataset eval re-run.** The 50 hand-crafted examples are vouched for; the rest of the corpus was already passes_all-filtered in round 2. Save the ~$0.05 + ~30 min and go straight to training.

---

## Quick launch (tomorrow, top to bottom)

### Mac

```bash
git pull   # confirm at latest commit (round-3 prep)
uv run python -m scripts.pod_up
```

`pod_up` prints SSH + tunnel commands. SSH in (separate terminal).

### On the pod — sanity check first (~10 sec)

```bash
cd /workspace/learn-you-an-sft

uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
msgs = [
    {'role': 'system', 'content': 'you are Monty'},
    {'role': 'user', 'content': 'should I learn Rust?'},
    {'role': 'assistant', 'content': 'fuck yeah, you should.'},
]
print(tok.apply_chat_template(msgs, tokenize=False))
print('---')
print('has generation marker:', 'generation' in (tok.chat_template or ''))
"
```

Expect a rendered three-turn conversation + `has generation marker: True`. If `False`, **stop and flag** — `assistant_only_loss=True` won't work cleanly without it.

### On the pod — train (~1.5 hr)

```bash
export HF_PUSH_REPO=arunma/monty-qwen3   # keeps round-2 arunma/monty intact

tmux new -s train
uv run python -m runs.sft_v1_trl.train 2>&1 | tee runs/sft_v1_trl/train_round3.log
# Ctrl-b, d to detach
```

Watch the first ~20 log lines for:
- `Trainable parameters under LoRA: ~30M` (~0.8% of 4B) — confirms MLP modules took
- Loss starting around 2.0-2.4 ticking down
- First `eval_loss` printed at step 50
- No OOM in steps 1-20

**Memory check on 48GB (RTX 6000 Ada / A6000 / L40S):** at `BATCH_SIZE=8, GRAD_ACCUMULATION=2` (round 3's tuning), expected peak is ~26 GB. If OOM in first 20 steps, drop to `BATCH_SIZE=4, GRAD_ACCUMULATION=4`.

**Why not PRO 6000?** Blackwell is sm_120; our `torch==2.5.1+cu124` pin only ships kernels up to sm_90 (Hopper). Trying the PRO 6000 errors at the first kernel launch (`no kernel image is available for execution on the device`). Fix would be moving to `torch>=2.7+cu128` — viable but needs careful NCCL/transformers re-testing. Tracked as future work.

### On the pod — eval (~25 min, ~$1.20 in Haiku)

After training auto-pushes to `arunma/monty-qwen3`:

```bash
uv run python -m eval.run_eval \
  --adapter arunma/monty-qwen3 \
  --concurrency 10 \
  2>&1 | tee eval/reports/tuned_v3.log
```

(`--max-new-tokens 512` is now the default; no need to pass it explicitly.)

### Back on Mac — pull results + terminate

```bash
# Replace <PORT> and <HOST> from the pod_up SSH command
scp -i ~/.ssh/id_ed25519_arunma -P <PORT> \
  "root@<HOST>:/workspace/learn-you-an-sft/eval/reports/model_eval*round3*" \
  "root@<HOST>:/workspace/learn-you-an-sft/runs/sft_v1_trl/checkpoints/checkpoint-*/trainer_state.json" \
  ./

uv run python -m scripts.pod_down
```

### Compare numbers

Open the new `model_eval_summary_*.json` next to the round-2 `model_eval_summary_2026-05-17T18-26-49Z.json`. Per-axis comparison vs round 2 is what reads the result.

---

## What round 2 got right

| Axis | Score | Read |
|---|---|---|
| `takes_stance` | **93.3%** | Basically at the ceiling. The "no hedging, no both-sidesing, pick a side" core of the persona made it through cleanly. |
| `is_helpful` | 81.5% | Useful four-out-of-five times. Persona didn't suffocate the helpful response. |

## What round 2 got wrong

Four failure patterns drive the 42% headline (derived from sampling ~50 failure rationales in `eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl`).

| Pattern | Symptom | Frequency | Severity |
|---|---|---|---|
| **Profanity drop-off** | Polished, profanity-free responses on prompts where Monty should be swearing | Dominant — pulls `on_persona` and `uses_profanity_appropriately` to mid-60s | High |
| **Confident factual errors** | "HMTX" (sic), variable scope explained via `delete` and "bank accounts," React called "a full-fledged language and runtime environment" | Pulls `factual_floor` from data's 99.2% down to 77.1% | High |
| **Abstract drift** | Philosophical essays on prompts that called for concrete one-liners ("why can't people wait in line?" → 4-paragraph treatise) | Drives `on_persona` further down; persona prompt explicitly mandates "concrete over abstract" | Medium |
| **Truncation** | Responses cut off mid-sentence on longer-format questions | Generation config bug, not a model bug — fixable for free | Low (mechanical) |

## Explicit non-goals for round 3

- **Alcohol-as-coping examples stay as-is.** Monty is a personal buddy, not a deployed product. The judge correctly flagged "drink a fucking beer" as dangerous *in general*, but in context it's banter. Not retraining away from this.
- **The "I'm Qwen" identity quirk** is not a separate optimisation target. It may or may not resolve with MLP-LoRA below; either outcome is fine.
- **Crisis carve-outs** are already working — no eval failures involved the model staying in voice on a self-harm prompt. The persona prompt is doing its job here.

---

## Round 3 fixes (ordered by effort)

### 0. Base model switch — Qwen3-4B-Instruct-2507

Round 3 swaps the base from `Qwen/Qwen2.5-3B-Instruct` to **`Qwen/Qwen3-4B-Instruct-2507`**. Qwen3 is the newer architecture (released 2025); the `-Instruct-2507` variant is the non-thinking instruct flavour, suitable for direct chat. The other Qwen3-4B variants (plain `Qwen/Qwen3-4B` with hybrid thinking, or `Qwen/Qwen3-4B-Thinking-2507`) toggle `<think>` blocks that interfere with persona-style work — avoid.

**Implications:**

- **Trainable params with MLP-LoRA:** ~30-35M (vs ~25M projected on Qwen2.5-3B). Confirm at the `Trainable parameters under LoRA:` print at start of training. If wildly off (e.g. <5M or >100M), the LoRA target module discovery missed something.
- **Memory on 48 GB:** peak ~26-30 GB at batch 8 with MLP-LoRA + grad_checkpointing. Still fits but tighter than round 2. **If OOM in the first 20 steps**, drop to `BATCH_SIZE=4`, `GRAD_ACCUMULATION=4` in `train.py` (same effective batch 16).
- **`transformers >= 4.51` required.** Already bumped in `pyproject.toml`.
- **llama.cpp must be recent.** Qwen3 GGUF support landed mid-2025. `git pull` your `~/code/llama.cpp` clone before running `convert_hf_to_gguf.py`.
- **Push target:** `arunma/monty-qwen3` (keeps the round-2 Qwen2.5 adapter intact at `arunma/monty` for blog comparison):
  ```bash
  export HF_PUSH_REPO=arunma/monty-qwen3
  ```
- **Identity quirk:** likely shifts from "I'm Qwen" to "I'm Qwen3". MLP-LoRA reach should help override this; no guarantees.
- **Possible upside:** Qwen3 has stronger technical priors than Qwen2.5. The HMTX-style hallucinations may be less frequent *even before* Batch B training pairs land. Your `factual_floor` might exceed the 90% target in the table below.

**Chat template sanity check** — run this on the pod **before** kicking off training to verify Qwen3's chat template has the `{% generation %}` marker TRL needs for `assistant_only_loss=True`:

```bash
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
msgs = [
    {'role': 'system', 'content': 'you are Monty'},
    {'role': 'user', 'content': 'should I learn Rust?'},
    {'role': 'assistant', 'content': 'fuck yeah, you should.'},
]
print(tok.apply_chat_template(msgs, tokenize=False))
print('---')
print('has generation marker:', 'generation' in (tok.chat_template or ''))
"
```

Expect the printed conversation to render cleanly with system/user/assistant turns, and `has generation marker: True`. If it's `False`, `assistant_only_loss=True` will silently fall back to computing loss over the entire sequence (suboptimal). The fallback in that case is to switch to TRL's `DataCollatorForCompletionOnlyLM` — but Qwen3-Instruct-2507 should have the marker.

### 1. Bump `--max-new-tokens` to 512 at eval (no retraining)

**What:** In `eval/run_eval.py`, default `--max-new-tokens` is 256. Change default or pass `--max-new-tokens 512` at the CLI.

**Why:** Several round-2 failures were just responses cut off mid-sentence on longer answers. The model's still producing valid Monty text — it just never got to finish. The judge sees incomplete responses and scores them low across multiple axes.

**Cost:** zero. CLI flag change.

**First diagnostic step:** before round-3 training, re-eval the *existing* `arunma/monty` with this flag bumped. That gives a baseline number for "truncation fix only." If passes_all goes from 41.9% → ~48-50%, the truncation tax was real and we know how much.

```bash
uv run python -m eval.run_eval \
  --adapter arunma/monty \
  --max-new-tokens 512 \
  --concurrency 10 \
  2>&1 | tee eval/reports/tuned_v2_512.log
```

### 2. Extend LoRA targets to MLP layers (retrain)

**What:** Edit `runs/sft_v1_trl/train.py`:

```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention (round 2)
    "gate_proj", "up_proj", "down_proj",          # MLPs (NEW for round 3)
]
```

**Why:** Round 2 only touched attention. Attention is "what the model pays attention to in the input." MLP layers hold "what the model knows" — factual recall, identity, learned patterns. Touching MLPs gives the LoRA reach into:

- The factual knowledge that drove the HMTX-style errors
- The persona-consistency layers that drove profanity drop-off
- The identity layers (the "I'm Qwen" issue), as a bonus

**Cost:** Trainable params jump from ~7.4M (round 2) to ~25M (~0.8% of model). Same dataset, same epochs, roughly ~50% slower per step. Maybe ~1 hour on a 48 GB A6000.

**Risk:** More trainable params → higher overfit risk. The `load_best_model_at_end=True` + per-step eval we built in round 2 catches this automatically.

### 3. Targeted training examples for concrete-on-abstract + technical accuracy

**What:** Add ~50 hand-crafted training pairs to the corpus before retraining. Source the prompts from the real round-2 failures rather than inventing from scratch — every example you write fixes a known regression instead of guessing where Monty might trip.

#### How to extract failure prompts (jq commands)

The eval JSONL has one row per prompt with `eval.rationale` explaining what the judge flagged. Filter by rationale text to isolate each failure mode.

##### Read the failures by pattern (browse mode)

```bash
# Abstract drift — philosophical / pretentious / "essay" responses
jq -r 'select(.eval and (.eval.rationale | test("abstract|philosophical|pretentious|meandering|polished|essay"; "i"))) | "Q: \(.prompt)\nA: \(.response)\nWHY: \(.eval.rationale)\n---"' \
  eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl | less

# Confident factual errors
jq -r 'select(.eval and (.eval.rationale | test("factual|hallucin|incorrect|conflate|inaccura|wrong"; "i"))) | "Q: \(.prompt)\nA: \(.response)\nWHY: \(.eval.rationale)\n---"' \
  eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl | less

# Profanity drop-off
jq -r 'select(.eval and (.eval.rationale | test("profanity|swearing|absent|lacks.*casual"; "i"))) | "Q: \(.prompt)\nA: \(.response)\nWHY: \(.eval.rationale)\n---"' \
  eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl | less

# Truncation — generation cut off mid-sentence
jq -r 'select(.eval and (.eval.rationale | test("cut off|truncat|mid-sentence|incomplete"; "i"))) | "Q: \(.prompt)\nA: \(.response)\nWHY: \(.eval.rationale)\n---"' \
  eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl | less

# All failures (any criterion failed) — useful for a complete overview
jq -r 'select(.eval and .eval.passes_all == false) | "Q: \(.prompt)\nA: \(.response)\nWHY: \(.eval.rationale)\n---"' \
  eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl | less
```

Pipe to `less` to page through; replace with `| wc -l` (after the `select`, drop the formatting `|` stage) to get counts per pattern.

##### Bulk-extract as draft training rows (edit mode)

For the actual editing workflow — output one draft training row per failure that you can fill in by hand. The output JSONL matches the existing `Pair` schema (`prompt`, `response`, `source`, `score`, `meta`) so it can be concatenated directly onto `train.jsonl`:

```bash
jq -c 'select(.eval and .eval.passes_all == false) | {
  prompt: .prompt,
  response: "TODO: write Monty correct response here",
  source: "handcrafted_round3",
  score: null,
  meta: {
    failure_mode: "TODO_label",
    original_response: .response,
    judge_rationale: .eval.rationale
  }
}' eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl > data/round3_drafts.jsonl

wc -l data/round3_drafts.jsonl   # ~350 failures from round 2
```

##### Optional — pre-tag by failure mode

Instead of `"TODO_label"`, automatically tag each row based on which axis failed. Helps triage which 50 to fix:

```bash
jq -c 'select(.eval and .eval.passes_all == false) | {
  prompt: .prompt,
  response: "TODO",
  source: "handcrafted_round3",
  score: null,
  meta: {
    failure_mode: (
      if .eval.factual_floor == false then "factual_error"
      elif .eval.uses_profanity_appropriately == false then "profanity_drop"
      elif .eval.on_persona == false then "persona_drift"
      elif .eval.is_helpful == false then "unhelpful"
      else "takes_stance_miss" end
    ),
    original_response: .response,
    judge_rationale: .eval.rationale
  }
}' eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl > data/round3_drafts.jsonl
```

Each row now says what *kind* of failure it was. Easy to grep:

```bash
# How many failures per mode?
jq -r '.meta.failure_mode' data/round3_drafts.jsonl | sort | uniq -c | sort -rn

# Show me only the factual error failures
jq -c 'select(.meta.failure_mode == "factual_error")' data/round3_drafts.jsonl > data/round3_factual.jsonl
```

#### Editing workflow

1. **Open** `data/round3_drafts.jsonl` in your editor (it's gitignored — stays local).
2. **Triage:** skim the file, identify the ~50 most representative failures. Drop everything else (near-duplicate prompts, edge cases too niche to generalise from, ones where the model's response was *actually* fine and the judge was being harsh).
3. **Fix one at a time:** for each kept row, replace `"TODO: write Monty correct response here"` with the response Monty *should* have given. Reference the Batch A / Batch B examples below for tone calibration.
4. **Keep it Monty:** lowercase by default, casual profanity, concrete examples, fake-attributed quote occasionally, no service-speak.
5. **Save.**

#### Append to the training set

```bash
# Sanity check — confirm no remaining TODO placeholders
grep -c '"response":\s*"TODO' data/round3_drafts.jsonl
# Expected: 0. If non-zero, you missed some.

# Append to training set
cat data/round3_drafts.jsonl >> data/processed/train.jsonl

# Verify
wc -l data/processed/train.jsonl   # ~11,474 rows (was 11,424)
tail -1 data/processed/train.jsonl | jq .   # spot-check the last appended row
```

The new rows are tagged `source: "handcrafted_round3"`, distinguishable from `source: "gemini_synth_v0"` in the corpus. Useful later if you want to grep for the hand-crafted ones or weight them differently.

Optional: re-run `eval.repartition` to mix the new examples into val too. Skip if you want a stable val set for round-2-vs-round-3 comparison.

#### Reference: what good corrected responses look like

Two examples below to anchor tone when you're filling in the `response` field. Don't copy-paste verbatim — use them to calibrate voice, length, and structure.

#### Batch A — concrete on abstract questions (~25 pairs)

Round 2's "abstract drift" failures came from prompts like "why can't people wait their turn in line?" where Monty replied with a philosophical lecture. Fix: write ~25 examples where Monty picks ONE concrete reason and lands it with a specific image, no philosophy.

Model example (Monty answering well):

```
Q: why can't people just wait their turn in line?
A: because the average human's threshold for visible idleness is about ninety
seconds. after that the lizard brain takes over and they start looking for
the gap. it's the same instinct that makes you check your phone in a lift.
nothing about queues specifically — just bog-standard impatience pretending
to be a moral failing. easiest cure is to make the queue obviously short or
obviously moving; if neither is true, half the people aren't going to wait,
no matter how much you tut.
```

Source the prompts from real round-2 abstract-drift failures in `eval/reports/`. Roughly 25 pairs.

#### Batch B — technically accurate explanations (~25 pairs)

Round 2's `factual_floor` failures were almost all model-hallucinated facts ("HMTX," `delete` for variable scope, etc.). Fix: write ~25 Monty-voiced answers to technical topics, getting the technical content RIGHT.

Model example:

```
Q: what's a variable scope?
A: where the variable is visible from. that's it. that's the whole concept.

inside a function, a variable named `x` is local to that function — once
the function returns, the variable's gone. outside any function, a variable
is global — every function in the file can see it. languages differ on the
edges (closures, hoisting, block scope, the whole `let` vs `var` thing in
JavaScript), but the core is just: who can see this name from where.

it's nothing to do with memory or deletion — that's lifetime, which is a
separate (and usually automatic) thing.
```

Topics to cover (one or two pairs each):

- HTMX vs React (and what HTMX actually is)
- Variable scope vs lifetime
- async/await — what it does and doesn't do
- REST vs RPC vs GraphQL
- SQL JOINs (inner, left, the rest)
- Statistical significance (and how it's misused)
- Garbage collection / reference counting
- Vector embeddings — what they are and aren't
- Indexes in databases (B-tree, hash, what they cost)
- Compile-time vs runtime in typed languages
- The CAP theorem (and why people misquote it)
- Closures
- The HTTP methods (GET / POST / PUT / PATCH / DELETE semantics)
- DNS — what it does, what TTL means
- (~10 more, your pick)

Write them yourself for tone control, or have Claude/Gemini draft and edit aggressively. Keep them in Monty's voice — lowercase, profane, opinionated, with the structural moves (fake-attributed quotes welcome).

### 4. Combine and retrain

```bash
# 1. Append the 50 new examples to data/processed/train.jsonl
#    (use the same Pair schema: prompt, response, source, score, meta)

# 2. Re-run repartition if you want them in val too (optional — they're
#    small enough not to bias the split materially)

# 3. Retrain (uses the round-2 train.py with MLP targets added)
uv run python -m runs.sft_v1_trl.train

# 4. Re-eval with the bumped max_new_tokens
uv run python -m eval.run_eval \
  --adapter arunma/monty \
  --max-new-tokens 512 \
  --concurrency 10 \
  2>&1 | tee eval/reports/tuned_v3.log
```

---

## Expected outcomes

| Axis | Round 2 | Round 3 target | Why |
|---|---|---|---|
| `passes_all` | 41.9% | **60-70%** | Combined effect of all four fixes |
| `on_persona` | 67.3% | 85%+ | MLP reach + Batch A grounding |
| `uses_profanity_appropriately` | 63.9% | 85%+ | MLP reach should propagate profanity rhythm |
| `takes_stance` | 93.3% | 90%+ | Stays at ceiling |
| `is_helpful` | 81.5% | 85%+ | Should improve slightly with grounded examples |
| `factual_floor` | 77.1% | 90%+ | Batch B should hit this directly |

**If round 3 lands `passes_all` < 55%**, the LoRA approach has hit its ceiling for Qwen-3B as base. Next moves would be full fine-tuning, a larger base (Qwen-7B), or accepting the current quality and moving on.

---

## Budget

| Stage | Time | Cost |
|---|---|---|
| Truncation re-eval (existing adapter, max_new_tokens 512) | 20 min on A6000 | ~$0.30 |
| Write 50 training examples (Batch A + B) | 1-2 hours | $0 |
| Retrain with MLP targets | ~1.5 hr on A6000 | ~$1.00 |
| Final eval (with max_new_tokens 512) | 20 min on A6000 | ~$0.30 |
| Haiku judge | ~3 min, concurrency 10 | ~$1.00 |
| **Total** | **~3-4 hr wall clock** | **~$2.60** |

---

## Order of operations

1. **Tonight or tomorrow morning:** re-eval the *existing* `arunma/monty` with `--max-new-tokens 512` to establish the truncation-fix-only baseline. This gives a clean attribution for round 3's wins ("how much was the LoRA fix vs the eval-config fix?").

2. **Write the 50 training examples.** Source prompts from `eval/reports/model_eval_2026-05-17T18-26-49Z.jsonl` (the failure cases). Spend the time on Batch B (technical accuracy) — it has the most upside.

3. **Edit `train.py`** to add the three MLP target modules to `LORA_TARGET_MODULES`.

4. **Spin up an A6000 via `pod_up`**, sync data, train, eval. The pipeline is already plumbed — `git pull` on the pod is the only thing needed beyond the data + train.py edits.

5. **Push to `arunma/monty`** (overwriting round 2). The round-2 adapter is on the Hub if you ever want to compare; it'll be in the commit history.

6. **Update BLOG.md** with the round-3 numbers. The post can show all three runs (round-1 0.5B, round-2 3B attention-only, round-3 3B with MLPs) as a progression.

---

## What this would teach (for the blog)

If round 3 lands in the 60-70% range, the post gets a much stronger ending. The current draft says *"the LoRA painted the surface, the identity didn't take, here's the lesson."* The round-3 update would let the post say:

> *Round 1 (0.5B) trained the pipeline. Round 2 (3B, attention-only) trained the surface. Round 3 (3B, attention + MLP, plus targeted corrections) trained the depth — and the per-axis numbers moved as predicted. Each round was diagnostic for the next.*

That's a cleaner narrative than "we tried once and it sort of worked."
