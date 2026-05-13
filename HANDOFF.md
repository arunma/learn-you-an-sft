# learn-you-an-sft — Handoff Document

> The single bootstrap doc to load into a new Claude context (or share
> with a future collaborator) and resume work without losing context.
> Captures the locked decisions, current code state, and every
> operating practice — RunPod, costs, failure modes, glossary.

---

## Table of contents

1. Project identity
2. The persona target (with explicit examples)
3. Locked architecture
4. Locked decisions (don't re-litigate)
5. Stages, with status
6. Open items still to confirm
7. Cost summary
8. Definition of done
9. Standing ops practices
10. Useful commands
11. Failure modes to expect
12. Glossary
13. How to bootstrap this in a new Claude context

---

## 1. Project identity

| Field | Value |
|---|---|
| Project name | **learn-you-an-sft** |
| Path | `/Users/arunmanivannan/projects/ai/learn-you-an-sft` |
| Sibling project | `/Users/arunmanivannan/projects/ai/learn-you-an-hf-llm` (TinyStories pretraining; Phase B's hand-rolled training loop forks from there) |
| Goal | Small instruct-tuned LLM that responds like a witty friend who roasts you, with casual profanity allowed |
| Real goal | Hands-on learning of LLM internals + ops, end-to-end |
| Naming rationale | Follows the "Learn You a Haskell" / sibling "learn-you-an-hf-llm" pattern; **sft** = Supervised Fine-Tuning |
| Git author | `Arun Manivannan <arun@arunma.com>` |

---

## 2. The persona target

A friend who knows things, doesn't sugarcoat, and roasts you in a way
that makes you laugh. **Useful answer + sarcastic delivery + casual
profanity when it fits.**

### Examples of what we WANT

| Question | Target response |
|---|---|
| "should I learn Rust?" | "yeah, why not — you clearly enjoy explaining your design choices to a compiler that's a bigger pedant than your worst code reviewer. it's worth it if you'll build something real and not just dick around on twitter about borrow-checker enlightenment." |
| "I just pushed broken code to main" | "amazing. another contribution to the company's git-blame archeology project. revert it, write a sticky note, and move on. happens to literally everyone with hands and a keyboard." |
| "explain monads in one sentence" | "they're burrito-shaped wrappers that let you chain operations that might fail or have side effects without scattering try/except over half your codebase. haskell's `do` notation is the least migraine-inducing way to see them in action." |
| "is it OK to push to main?" | "sure, if you also enjoy reverting in front of an entire slack channel. use a feature branch — your future self will thank you and your present self might survive friday." |
| "be honest, am I overthinking this?" | "almost certainly. you're the kind of person who reads the fine print on a screwdriver. pick the simpler option and ship it." |

### Examples of what we DO NOT want

| Failure mode | Example |
|---|---|
| Pure quip, no useful content | "git is stupid" |
| Pure answer, zero voice | "A monad is a structure with bind (`>>=`) and return operations satisfying three laws…" |
| Mean without being funny | "lmao if you have to ask, just quit programming" |
| Performatively edgy and bored | "wow another rust question, original" |
| Slurs / punching down on protected groups | (any of them — disqualifying, regardless of context) |
| Cringey forced profanity | "the goddamn fucking shitty hellfuck of a question is…" — profanity should land naturally, not pile up |

### The bright line on profanity

| Allowed | NOT allowed |
|---|---|
| fuck / shit / hell / dick / asshole as friendly emphasis | slurs targeting race, gender, sexuality, religion, disability |
| roasting the user's choices / questions / behaviour | punching down on protected groups |
| dry contempt for stupid technology / processes | endorsing harm, real-world hostility, threats |

Eval rubric scores **three** dimensions: useful, witty, appropriately
edgy. All three must be ≥ acceptable for a win.

---

## 3. Locked architecture

```
                    ┌─────────────────────────────────────┐
                    │  synthesis/  (Gemini 2.5 Pro)       │
                    │  question_pool → persona prompt →   │
                    │  Pair JSONL                          │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  data/filter/  (normalize, lang,    │
                    │  dedup)                              │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  data/processed/{train,val}.jsonl   │
                    │  + manifest.json                     │
                    └──────────────┬──────────────────────┘
                                   ▼
            ┌──────────────────────┴──────────────────────┐
            ▼                                              ▼
┌───────────────────────────┐              ┌──────────────────────────┐
│  Phase A — TRL + LoRA     │              │  Phase B — Hand-rolled   │
│  runs/sft_v1_trl/         │              │  runs/sft_v2_internals/  │
│  Qwen2.5-0.5B + adapter   │              │  Qwen2.5-0.5B full FT    │
└──────────────┬────────────┘              └──────────────┬───────────┘
               ▼                                          ▼
            ┌─────────────────────────────────────────────┐
            │  eval/judge.py  (Claude Haiku 4.5)          │
            │  pairwise vs base, win-rate + Wilson CI     │
            └──────────────┬──────────────────────────────┘
                           ▼
                ┌──────────────────────┐
                │  inference/chat.py    │   ← Stage 5
                │  (CLI you talk to)    │
                └──────────────────────┘
                           ▼
                ┌──────────────────────┐
                │  Stage 7 — Tools     │   ← Future
                │  weather, time, etc. │
                └──────────────────────┘
```

---

## 4. Locked decisions (don't re-litigate)

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Project shape | SFT, not pretraining | Small model + 15K examples teaches a persona, not a base |
| 2 | Persona | Useful + witty + casually profane | Pure quip is annoying; the user wants a friend with bite |
| 3 | Profanity ceiling | Friendly emphasis OK (fuck/shit/etc); no slurs ever | Bright line on punching down |
| 4 | Data source | **100% Gemini synth** | Distillation gives uniform voice and persona control |
| 5 | Teacher model | **Gemini 2.5 Pro** | Voice quality; ~$15 for 15K examples or free tier across days |
| 6 | Question pool | Gemini 2.5 Flash | Variety; voice doesn't matter for prompts |
| 7 | Judge model | **Claude Haiku 4.5** | Different family from teacher = no self-preference bias; ~$10 across project |
| 8 | Base model | **Qwen2.5-0.5B** | Native tool-call chat template; future-proofs Stage 7 |
| 9 | Two-phase training | Phase A (TRL+LoRA), Phase B (hand-rolled full FT) | Phase A = working pipeline fast; Phase B = deep dive |
| 10 | Training hardware | **RunPod H100** | Standard, ~$2–$3/hr depending on community vs secure |
| 11 | Eval method | Pairwise LLM-as-judge, locked prompt set | Industry standard; reliable enough |
| 12 | Filter modules | normalize + language + dedup only | Quality filter redundant with Gemini synth; toxicity filter would kill the persona |
| 13 | Train/val split | SHA1-of-response (2% val) | Stable across re-runs |
| 14 | Reproducibility | manifest.json with dataset SHA256 + pip freeze + git SHA | Can't tell what changed between runs otherwise |
| 15 | Tool calling | Deferred to Stage 7 (after Stage 5 ships) | Out of scope for v1; architecturally prepared via base-model choice |
| 16 | Git tracking | Yes — `git init` done on `main` | Local-only; remote optional later |
| 17 | Project packaging | `pyproject.toml` + `uv.lock`, flat deps (no optional groups) | Matches sibling project; `bitsandbytes` gated to Linux so macOS dev installs succeed |

---

## 5. Stages, with status

| Stage | Name | Status | LoC (approx) |
|---|---|---|---|
| 1.5 | **Synthesis pipeline** (Gemini) | **TO BUILD — primary data source** | ~300 |
| 2a | Pair schema (ingest) | **BUILT** — `data/ingest/schema.py` only | ~50 |
| 2b | Filter pipeline | **BUILT** — normalize → language → dedup → train/val + manifest | ~200 |
| 2.5 | Eval scaffolding (lock before training) | **TO BUILD** | ~250 + 130 prompts |
| 3a | SFT data formatting (chat template + loss mask + packing) | **TO BUILD** | ~200 |
| 4 v1 | SFT with TRL + LoRA | **TO BUILD** | ~150 |
| 4 v2 | SFT hand-rolled | **TO BUILD** (forks `learn-you-an-hf-llm/runs/h100_tinystories/train.py`) | ~500 |
| 5 | Generation + sampling (hand-rolled) | **TO BUILD** | ~250 |
| 6 | Cross-version comparison + ablations | **TO BUILD** | ~50 + multiple runs/ folders |
| 7 | **Tool calling** | **DEFERRED** — design notes below; build after Stage 5 ships | ~400 |

### Stage 7 design notes (for when we get there)

**Phase 1** — pick the format. Qwen2.5 already uses an OpenAI-compatible
function-calling format in its chat template (`<tool_call>` / `<tool_result>`
roles). Default to that; don't reinvent.

**Phase 2** — pick the tool set. Read-only, low-stakes:
- `time_now()` — current time in a given timezone
- `weather(location)` — current weather via free API
- `web_search(query)` — DuckDuckGo or similar
- `calendar_events(date)` — local ICS file read

Stay away from filesystem write, code execution, and anything
authenticated until much later.

**Phase 3** — synthesize tool-use traces. Extend `synthesis/generate.py`
to produce multi-turn examples: user-turn → assistant-reasoning →
tool_call → tool_result → final-answer. ~5K traces, second SFT pass
starting from the witty-friend checkpoint.

**Phase 4** — build the runtime. `inference/tool_runtime.py` intercepts
tool_call tokens, executes the real tool, feeds back tool_result
tokens.

**Architectural prep already done:** Qwen2.5-0.5B chosen as base
specifically for tool-call chat-template support.

---

## 6. Open items still to confirm

| # | Item | Default if unspecified | Resolution path |
|---|---|---|---|
| 1 | Gemini API key configured? | Blocks synthesis | Paid tier set up; set `GEMINI_API_KEY` |
| 2 | Claude API funded (+$10)? | Falls back to local fallback judge (worse) | Add credits before Stage 2.5; set `ANTHROPIC_API_KEY` |
| 3 | Persona prompt drafted? | Co-write in the current session | Iterate against 5 hand-eval examples first |
| 4 | JSON-only experiment tracking | YES (locked) | `runs/<run_id>/loss.jsonl` + `eval.jsonl` + `manifest.json` |
| 5 | Push to GitHub remote? | Local-only is fine | Push if you want backup |

---

## 7. Cost summary

| Item | Estimated cost |
|---|---|
| Gemini 2.5 Pro synthesis (15K examples) | $15 (or free across 2–3 days on free tier) |
| Gemini 2.5 Flash question-pool generation | Free tier |
| Claude Haiku 4.5 judge (~25 runs × $0.40) | $10 |
| RunPod H100 training (4 runs × 30 min) | $6–10 |
| RunPod H100 ablations (3–4 short runs) | $4–6 |
| **Subtotal** | **~$35–45** |
| Buffer for mistakes | $10–15 |
| **Total project budget** | **~$50–60** |

**Worst-case scenarios to avoid:**
- Forgot to terminate a RunPod pod for 24 hours → +$48–72
- Used Sonnet 4.6 instead of Haiku 4.5 for judge → +$20
- Bulk-synthesized 50K examples instead of 15K → +$30

---

## 8. Definition of done

### Per stage

| Stage | Done when |
|---|---|
| 1.5 | `data/interim/<run_id>.pairs.jsonl` ≥ 15K examples; persona prompt locked; 50-example spot-check reads useful + witty + edged correctly |
| 2a | `data/ingest/schema.py` is the only ingest code; READMEs reflect synth-only |
| 2b | Pipeline runs without quality/toxicity stages; `data/processed/{train,val,manifest}.json` exist; re-run produces identical SHA256s |
| 2.5 | `eval/{prompts,golden,judge_prompt}.*` locked; `judge.py` runs on baseline; golden-set agreement ≥ 70% |
| 3a | Round-trip test passes: hand-built chat template = `apply_chat_template` token-for-token |
| 4 v1 | Trained adapter saved; judge win-rate vs Qwen2.5-0.5B base ≥ 55%; samples readable |
| 4 v2 | Hand-rolled loss curve ≈ v1; v1 vs v2 within ±10% |
| 5 | Greedy = `transformers.generate(do_sample=False)` token-for-token; CLI works |
| 6 | ≥ 2 ablation questions answered with win-rate evidence in `ABLATIONS.md` |
| 7 | (Future) model executes ≥ 3 tools correctly in interactive session |

### Project-level done (v1 — pre-tool-calling)

- `python -m inference.chat` opens a conversation with your model
- Total spend < $60
- Judge win-rate vs Qwen2.5-0.5B base ≥ 70% on locked eval
- 20 random outputs cold-read: ≥ 5 land funny, ≥ 15 useful, 0 cross the slur line
- Phase A ≈ Phase B (within ±10% win-rate)
- `ABLATIONS.md` answers ≥ 2 questions
- `POSTMORTEM.md`: what surprised you, what you'd do differently

---

## 9. Standing ops practices

### Every training run emits to `runs/<run_id>/`

| File | Contents |
|---|---|
| `manifest.json` | `{seed, dataset_sha256, base_model_revision, pip_freeze, gpu_model, code_git_sha, hyperparams_dict}` |
| `loss.jsonl` | One JSON line per logging step |
| `eval.jsonl` | One JSON line per judge call (full transcript) |
| `samples.txt` | Generation samples at each eval checkpoint |
| `notes.md` | Free-form: what you expected, what happened, what you'll change |

### Cost ledger (`runs/COSTS.md`)

```
| Date       | Mins | $    | Stage | What you learned |
|------------|------|------|-------|------------------|
| 2026-05-14 |  150 | 0.00 | 1.5   | Persona v0; pilot 100 examples; needs more edge |
```

### RunPod H100 discipline

1. **Calendar reminder 1 hour after launch** — verify the pod terminated
2. Always `tmux new -s train` before starting training
3. Pull checkpoints back to local before terminating
4. Terminate via the RunPod web UI (don't trust CLI)
5. Refresh dashboard after — confirm zero running pods

### Reproducibility check after every checkpoint save

```python
loaded = AutoModelForCausalLM.from_pretrained(ckpt_path)
out = loaded.generate("Hello, friend.", do_sample=False, max_new_tokens=50)
# Save to runs/<run_id>/parity_check.txt
```

If a re-load produces different greedy output, the save/load is
broken — fix before training more.

---

## 10. Useful commands

```bash
# === Project setup ===
cd /Users/arunmanivannan/projects/ai/learn-you-an-sft
uv venv
uv pip install -e .
source .venv/bin/activate

# === Stage 2b filter ===
uv run python -m data.filter.pipeline

# === Stage 1.5 synthesis (to build) ===
uv run python -m synthesis.generate --provider gemini --count 100   # pilot
uv run python -m synthesis.generate --provider gemini --count 15000 # bulk

# === Stage 2.5 eval ===
uv run python -m eval.judge \
  --candidate runs/sft_v1_trl/checkpoints/final \
  --baseline Qwen/Qwen2.5-0.5B-Instruct \
  --judge-model claude-haiku-4-5

# === Inspection ===
cat data/processed/manifest.json | python -m json.tool
jq '.source' data/processed/train.jsonl | sort | uniq -c
shuf data/processed/train.jsonl | head -20 | jq '.'

# === RunPod H100 ===
# Launch a pod via the RunPod web UI; copy its public IP.
ssh root@<public-ip>
git clone <your-remote>/learn-you-an-sft.git
cd learn-you-an-sft && uv pip install -e .
tmux new -s train
uv run python -m runs.sft_v1_trl.train
# Ctrl-b d to detach
scp -r root@<public-ip>:~/learn-you-an-sft/runs/sft_v1_trl/checkpoints \
       runs/sft_v1_trl/checkpoints
# *** TERMINATE POD ON THE RUNPOD DASHBOARD ***

# === Git workflow ===
git add -A
git status
git commit -m "stage X: <what changed>"
git log --oneline
```

---

## 11. Failure modes to expect

| Symptom | Cause | Fix |
|---|---|---|
| Gemini API "quota exceeded" | Free tier / paid burst limit | Spread across days; pay-as-you-go |
| Gemini refuses profane requests | Safety filter | Roleplay frame: "dialogue for a sharp-tongued friend character…"; drop refusals at filter stage |
| Synth examples all sound the same | Persona prompt has no variety hooks | Add temperature variation + per-example random seed |
| MinHash uses 4 GB RAM | NUM_PERM × N too high | Lower NUM_PERM to 64; lower JACCARD_THRESHOLD to 0.6 |
| fasttext lid.176 not found | Skipped download step | `curl -L <url> -o ~/.cache/fasttext/lid.176.bin` |
| Loss NaN after few hundred steps | LR too high for bf16 | Drop LR to 1e-5, or fall back to fp32 |
| OOM on H100 step 0 | `torch.compile` first-step spike | Drop BATCH_SIZE 64 → 32 |
| RunPod pod left running overnight | Forgot to terminate | Calendar reminder discipline |
| Manifest SHA256 changes on re-run | Non-deterministic step | Seed every `random.Random()` and `np.random.seed()` |
| Judge gives different verdict on rerun | Temperature > 0 in judge call | Set temperature=0 |
| v1 (TRL) and v2 (hand-rolled) loss curves diverge | Loss mask / packing / template mismatch | Run Stage 3a round-trip test |
| Model says the same canned line back to anything | Overfit to manufactured-prompt distribution | More variety in `question_pool.py` |
| Model is too polite (no edge) | Persona prompt too soft, or toxicity filter still applied | Re-read `persona_prompt.md`; confirm filter has no toxicity gate |
| Model is gratuitously cruel | Persona prompt over-corrected toward edge | Tighten "friend, not enemy" framing; add bright lines re: punching down |

### Things that aren't bugs but look like bugs

- First H100 step takes 60+ seconds — `torch.compile` tracing
- Stage 2b filter prints nothing for a minute on first run — fasttext model loading
- Gemini occasionally returns empty string — retry with backoff; drop after 3 failures

---

## 12. Glossary

| Term | One-line meaning |
|---|---|
| **SFT** | Supervised Fine-Tuning — train a pre-trained model on (prompt, response) pairs |
| **LoRA** | Low-Rank Adaptation — trains ~1% of weights as low-rank deltas |
| **PEFT** | Parameter-Efficient Fine-Tuning — HF library implementing LoRA |
| **TRL** | Transformer Reinforcement Learning — HF library wrapping SFT/DPO/PPO trainers |
| **Chat template** | Token format the model recognises as turns (e.g. `<|im_start|>user\n…\n<|im_end|>`) |
| **Loss masking** | Setting label tokens to `-100` so cross-entropy is computed only on response |
| **Sequence packing** | Multiple short examples in one fixed-length sequence with attention masks isolating them |
| **MinHash + LSH** | Probabilistic algorithm for sub-linear near-duplicate detection |
| **KV cache** | Stored keys/values from previous tokens during generation; avoids recomputing attention |
| **Top-k / top-p sampling** | Limit next-token sampling to top k tokens, or smallest set with cumulative prob ≥ p |
| **Repetition penalty** | Downweight tokens already in context to reduce loops |
| **Temperature** | Divide logits before softmax; T<1 sharpens, T>1 flattens |
| **bf16** | 16-bit float with fp32-range exponent; preferred over fp16 for training |
| **AdamW** | Adam optimizer with decoupled weight decay; transformer default |
| **Cosine LR schedule** | LR decays as a half-cosine from peak to zero |
| **Warmup** | Linear LR ramp from 0 to peak over first N steps |
| **Self-preference bias** | LLMs prefer outputs from their own family when judging |
| **Position bias** | Judges favor "A" ~55% regardless of content |
| **Wilson CI** | Confidence interval for proportions; works near 0 or 1 |
| **Manifest** | JSON record of everything needed to reproduce a run |
| **Greedy decoding** | At each step, pick the highest-prob token. Deterministic; flat output |
| **Pairwise win-rate** | Fraction of N comparisons your model won; the primary judge metric |
| **Distillation** | Training a small model on a large model's outputs to inherit its behavior |
| **Tool calling / function calling** | Model emits structured tokens for a function call; runtime executes; result feeds back |

---

## 13. How to bootstrap this in a new Claude context

Open a fresh Claude Code session in
`/Users/arunmanivannan/projects/ai/learn-you-an-sft`. Your first
message:

> I'm continuing the learn-you-an-sft project. Please read HANDOFF.md
> end-to-end and confirm:
> 1. The persona target (useful + witty + casually profane friend)
> 2. The current stage status (filter pipeline done; synthesis +
>    chat template + SFT training are the next builds)
> 3. The locked decisions in § 4 (don't re-litigate)
>
> Then resume where the previous session left off.

The new context will read HANDOFF.md, internalize the locked decisions,
and resume work without re-litigating settled debates. If the new
context ever tries to remove profanity from the persona or swap base
models away from Qwen2.5, point it back to **§ 4 (locked decisions)**
— those debates are settled.

### What the new context SHOULD read

- This file (`HANDOFF.md`) — everything
- `README.md` — top-level project map + setup
- `data/filter/{normalize,language,dedup,pipeline}.py` — the current
  filter pipeline
- `data/ingest/schema.py` — the canonical `Pair` schema
- `/Users/arunmanivannan/projects/ai/learn-you-an-hf-llm/runs/h100_tinystories/train.py`
  — the basis for Phase B's hand-rolled training loop

---

## Final note

The single most important thing in this project is
`synthesis/persona_prompt.md`. Its biases become the model's biases.
Iterate against 5 hand-eval examples before generating any bulk data.

The second most important thing is locking `eval/prompts.jsonl` on
day one of Stage 2.5 and never modifying it based on results.
Cheating on eval is invisible — you won't know you did it until the
model fails in real conversation.

Everything else is just code.
