# learn-you-an-sft — Handoff Document

> **What this is:** the single bootstrap doc to load into a new Claude
> context (or share with a future collaborator) and resume work without
> losing context. Captures the journey, every locked decision, the
> current code state, what's still pending, and exactly what to do next.
>
> **Status as of 2026-05-13 evening:** project direction has shifted
> twice from the original plan. Pure-synth, profane-friend persona,
> Qwen base. Source-file cleanup is DONE; rename + remove-stale-docs
> + minor edits are pending (see § 7).

---

## Table of contents

1. Project identity
2. The journey — four pivots so far
3. The persona target (with examples that include profanity)
4. Locked architecture
5. Locked decisions (don't re-litigate)
6. Files currently on disk
7. Cleanup still pending
8. All stages, with status
9. Open items still to confirm
10. Tomorrow's first session — actionable checklist
11. Cost summary
12. Definition of done
13. Standing ops practices
14. Useful commands
15. Failure modes to expect
16. Glossary
17. How to bootstrap this in a new Claude context

---

## 1. Project identity

| Field | Value |
|---|---|
| Project name | **learn-you-an-sft** |
| Previous name | `bash_org` (rename pending — see § 7) |
| Path | `/Users/arunmanivannan/projects/ai/learn-you-an-sft` (after rename) |
| Sibling project | `/Users/arunmanivannan/projects/ai/learn-you-an-hf-llm` (TinyStories pretraining; Phase B's hand-rolled training loop will fork from there) |
| Goal | Small instruct-tuned LLM that responds like a witty friend who roasts you, with casual profanity allowed |
| Real goal | Hands-on learning of LLM internals + ops, end-to-end |
| Naming rationale | Follows the "Learn You a Haskell" / sibling "learn-you-an-hf-llm" pattern; **sft** = Supervised Fine-Tuning |

---

## 2. The journey — four pivots so far

Each pivot replaced the previous plan; **don't go back to earlier
versions** without re-reading why we left them.

### v0 (2026-05-09 AM): "Scrape bash.org"

**Plan:** crawl bash.org for IRC humor data, train a sarcasm model.
**Killed by:** bash.org has been offline since ~2022. No live site
to crawl. Built a Wayback Machine CDX-based scraper instead.

### v1 (2026-05-09 PM): "Multi-source scraped mix"

**Plan:** bash.org (via Wayback) + rJokesData + SARC + dad jokes.
~300K filtered pairs after toxicity/quality filter. Train SmolLM2-360M
with TRL+LoRA.
**Killed by (2026-05-12):** target persona sharpened from "pure sarcasm
quip" to "useful + witty friend." Scraped corpora are all setup→punchline
or context→reply — they carry no useful information content. Wrong
shape for the new target.

### v2 (2026-05-12 AM): "Add Gemini synth as flavor, keep scraped as primary"

**Plan:** ~10K Gemini-synth examples for voice + ~200K scraped pairs
for register variety.
**Killed by (2026-05-13):** scraped data carries more negatives than
positives — slurs, formatting noise, decade-old memes, register
mismatch. Cleaner to drop scraped entirely.

### v3 (2026-05-13, CURRENT): "Pure synth, profane persona, plan for tools"

**Plan:** ~25K examples, 100% Gemini-synthesized. Profane/edgy friend
persona — "fuck" and friendly insults explicitly allowed. Qwen2.5-0.5B
base (changed from SmolLM2 — Qwen has native tool-call chat template).
Tool calling deferred to Stage 7.

Source-file cleanup executed 2026-05-13 PM (user ran `rm -rf scrape data/ingest/{bashorg,rjokes,sarc,dadjokes}.py data/filter/{quality,toxicity}.py`).

---

## 3. The persona target

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
| "tell me a joke" | "no, fuck off and ask me an actual question. or wallow in joke-asking mode. your call, champ." |

### Examples of what we DO NOT want

| Failure mode | Example |
|---|---|
| Pure quip, no useful content | "git is stupid" |
| Pure answer, zero voice | "A monad is a structure with bind (`>>=`) and return operations satisfying three laws…" |
| Mean without being funny | "lmao if you have to ask, just quit programming" |
| Reddit-edgelord bored | "wow another rust question, original" |
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

## 4. Locked architecture

```
                    ┌─────────────────────────────────────┐
                    │  synthesis/  (Gemini 2.5 Pro)       │
                    │  question_pool → persona prompt →   │
                    │  Pair JSONL                          │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  data/filter/  (normalize, lang,    │
                    │  dedup; NO quality/toxicity)        │
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

## 5. Locked decisions (don't re-litigate)

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Project shape | SFT, not pretraining | 28M model on 30K examples teaches nothing |
| 2 | Persona | Useful + witty + casually profane | Pure quip is annoying; the user wants a friend with bite |
| 3 | Profanity ceiling | Friendly emphasis OK (fuck/shit/etc); no slurs ever | Bright line on punching down |
| 4 | Data source | **100% Gemini synth** | Scraped corpora carry more negatives than positives for this target |
| 5 | Teacher model | **Gemini 2.5 Pro** | Voice quality; ~$15 for 15K examples or free tier across days |
| 6 | Question pool | Gemini 2.5 Flash + public Q&A datasets | Variety; voice doesn't matter for prompts |
| 7 | Judge model | **Claude Haiku 4.5** | Different family from teacher = no self-preference bias; ~$10 across project |
| 8 | Base model | **Qwen2.5-0.5B** (changed from SmolLM2-360M) | Native tool-call chat template; future-proofs Stage 7 |
| 9 | Two-phase training | Phase A (TRL+LoRA), Phase B (hand-rolled full FT) | Phase A = working pipeline fast; Phase B = deep dive |
| 10 | Training hardware | Lambda Labs H100 @ $2.99/hr | Standard, well-documented |
| 11 | Eval method | Pairwise LLM-as-judge, locked prompt set | Industry standard; reliable enough |
| 12 | Filter modules | normalize + language + dedup only | Quality filter redundant with Gemini synth; toxicity filter would kill the persona |
| 13 | Train/val split | SHA1-of-response (2% val) | Stable across re-runs |
| 14 | Reproducibility | manifest.json with dataset SHA256 + pip freeze + git SHA | Can't tell what changed between runs otherwise |
| 15 | Tool calling | Deferred to Stage 7 (after Stage 5 ships) | Out of scope for v1; architecturally prepared via base-model choice |
| 16 | Git tracking | Yes — `git init` after rename | Was missing; pre-rename project was not a git repo |

---

## 6. Files currently on disk

After the 2026-05-13 PM cleanup:

```
bash_org/                                  ← rename PENDING to learn-you-an-sft (§ 7)
├── README.md                              project map (will be updated)
├── PLAN.md                                ★ stale, slated for deletion (§ 7)
├── KICKOFF.md                             ★ stale, slated for deletion (§ 7)
├── HANDOFF.md                             ← THIS FILE — the current truth
│
├── data/
│   ├── __init__.py                        keep
│   ├── README.md                          minor update needed
│   ├── raw/                               empty (nothing was scraped)
│   ├── interim/                           empty
│   ├── processed/                         empty
│   │
│   ├── ingest/
│   │   ├── __init__.py                    keep
│   │   ├── schema.py                      keep — Pair is still canonical
│   │   └── README.md                      needs rewrite — synth-only
│   │
│   └── filter/
│       ├── __init__.py                    keep
│       ├── normalize.py                   keep
│       ├── language.py                    keep
│       ├── dedup.py                       keep
│       ├── pipeline.py                    needs edit — drop quality+toxicity gates
│       └── README.md                      needs update — drop quality+toxicity rows
│
├── eval/                                  empty — Stage 2.5 not built yet
├── inference/                             empty — Stage 5 not built yet
└── runs/                                  empty — Stages 4, 6 not built yet
```

**Already deleted (2026-05-13 PM):**
- `scrape/` (entire directory: `__init__.py`, `wayback.py`, `parse.py`, `main.py`, `README.md`)
- `data/ingest/{bashorg,rjokes,sarc,dadjokes}.py`
- `data/filter/{quality,toxicity}.py`

---

## 7. Cleanup still pending

Three things left, all small:

### 7a. Delete stale docs + rename + git init (one shell paste)

```bash
cd /Users/arunmanivannan/projects/ai

# 1. Delete now-stale docs
rm /Users/arunmanivannan/projects/ai/bash_org/PLAN.md \
   /Users/arunmanivannan/projects/ai/bash_org/KICKOFF.md

# 2. Rename the project
mv bash_org learn-you-an-sft
cd learn-you-an-sft

# 3. .gitignore
cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.DS_Store
*.log

# Data — never committed (regeneratable from synth)
data/raw/
data/interim/
data/processed/

# Runs — large; pull selectively if needed
runs/*/checkpoints/
runs/*/loss.jsonl

# Local secrets
.env
EOF

# 4. Init git + first commit
git init
git add -A
git status                       # eyeball before committing
git commit -m "initial commit: learn-you-an-sft project bootstrap

Pure-synth SFT pipeline for a witty-friend persona LLM.
See HANDOFF.md for full project context."
```

### 7b. Edit `data/filter/pipeline.py` — drop quality+toxicity gates

The `quality_gate()` and `toxicity_gate()` function definitions and
their two call sites need to be removed, plus the `--no-quality` and
`--no-toxicity` argparse flags. The new Claude context can do this
surgically — it's a ~30-line subtraction.

### 7c. Update the 4 affected READMEs

- `data/filter/README.md` — drop quality.py and toxicity.py rows; update filter-order table
- `data/ingest/README.md` — rewrite for synth-only (currently describes 4 scraped sources)
- `data/README.md` — minor: synth flow instead of raw → interim → processed
- `README.md` (top-level) — update deps list (no detoxify, no transformers for quality classifier)

Same path: let the new Claude context do all four.

---

## 8. All stages, with status

| Stage | Name | Status | LoC (approx) |
|---|---|---|---|
| 1 | bash.org Wayback scraper | **DELETED 2026-05-13** — was obsolete | (gone) |
| 1.5 | **Synthesis pipeline** (Gemini) | **TO BUILD — primary data source now** | ~300 |
| 2a | Source ingestion (schema only) | **PARTIALLY BUILT** — only `schema.py` remains | ~50 |
| 2b | Filter pipeline | **PARTIALLY BUILT** — `pipeline.py` needs edit (§ 7b) | ~250 (after edit) |
| 2.5 | Eval scaffolding (lock before training) | **TO BUILD** | ~250 + 130 prompts |
| 3a | SFT data formatting (chat template + loss mask) | **TO BUILD** | ~200 |
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

Stay away from filesystem write, code execution, anything authenticated
until much later.

**Phase 3** — synthesize tool-use traces. Extend `synthesis/generate.py`
to produce multi-turn examples: user-turn → assistant-reasoning →
tool_call → tool_result → final-answer. ~5K traces, second SFT pass
starting from the witty-friend checkpoint.

**Phase 4** — build the runtime. `inference/tool_runtime.py` intercepts
tool_call tokens, executes the real tool, feeds back tool_result tokens.

**Architectural prep already done:** Qwen2.5-0.5B chosen as base
specifically for tool-call chat-template support.

---

## 9. Open items still to confirm

| # | Item | Default if unspecified | Resolution path |
|---|---|---|---|
| 1 | Gemini API key configured? | Blocks synthesis | Set up at aistudio.google.com tomorrow |
| 2 | Claude API funded (+$10)? | Falls back to local LM Studio judge (worse) | Add credits before Stage 2.5 |
| 3 | Persona prompt drafted? | Co-write in Session 2 with new context | Iterate against 5 hand-eval examples first |
| 4 | Largest LM Studio model available? | Assume Qwen3-14B | Tell new context first thing |
| 5 | W&B or JSON-only experiment tracking? | JSON-only (zero setup) | Decide before first H100 run |
| 6 | Push to GitHub remote? | Local-only is fine | Push if you want backup |

---

## 10. Tomorrow's first session — actionable checklist

**Target:** 2–3 hours. **Outcome:** project renamed, git-tracked,
synth pipeline scaffolded, persona prompt drafted, first 100 synth
examples generated.

### Pre-flight (10 min)

```
[ ] Get Gemini API key (https://aistudio.google.com → Get API key)
    export GEMINI_API_KEY=...                 (add to ~/.zshrc to persist)

[ ] Add $10 to Anthropic Claude API balance
    export ANTHROPIC_API_KEY=...

[ ] Note the largest model you can run in LM Studio: __________________
```

### Rename + cleanup + git init (5 min — paste § 7a verbatim)

Run the shell block in § 7a. End state: `learn-you-an-sft/` with no
stale docs and a first git commit.

### Environment setup (5 min)

```bash
cd /Users/arunmanivannan/projects/ai/learn-you-an-sft
uv venv
uv pip install httpx ftfy fasttext datasketch tqdm \
               transformers tokenizers \
               google-generativeai anthropic

mkdir -p ~/.cache/fasttext
curl -L https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -o ~/.cache/fasttext/lid.176.bin
```

### Resume in a new Claude Code context

Open a fresh Claude Code session in this directory. First message:

> I'm continuing the learn-you-an-sft project. Please read HANDOFF.md
> end-to-end and confirm:
> 1. The persona target (useful + witty + casually profane friend)
> 2. The current code state (cleanup done; pipeline.py + READMEs need edits)
> 3. The Stage 1.5 work that comes next
>
> Once confirmed, do these in order:
> a. Edit `data/filter/pipeline.py` to remove `quality_gate`,
>    `toxicity_gate`, and their argparse flags
> b. Rewrite `data/ingest/README.md` for synth-only
> c. Update `data/filter/README.md` and `data/README.md` accordingly
> d. Update `README.md` (top-level) with the new dep list
> e. Commit everything as "stage 2b cleanup: drop quality/toxicity gates"
>
> Then we'll build `synthesis/` together.

### Session 2 work (~1.5 hr after the new context catches up)

```
[ ] Build synthesis/ scaffold:
      synthesis/__init__.py
      synthesis/providers/{__init__,base,gemini,lmstudio}.py
      synthesis/question_pool.py
      synthesis/generate.py
      synthesis/persona_prompt.md       ← spend 30 min here, iterate
      synthesis/README.md

[ ] Hand-iterate persona_prompt.md against 5 example questions until
    each response feels right.

[ ] Run a pilot: 100 examples via Gemini 2.5 Pro. Eyeball all 100.
    Tweak persona prompt if anything reads bland, mean-without-funny,
    or refuses-to-curse.

[ ] git commit -m "stage 1.5: synthesis scaffold + persona prompt v0"
```

---

## 11. Cost summary

| Item | Estimated cost |
|---|---|
| Gemini 2.5 Pro synthesis (15K examples) | $15 (or free across 2–3 days on free tier) |
| Gemini 2.5 Flash question-pool generation | Free tier |
| Claude Haiku 4.5 judge (~25 runs × $0.40) | $10 |
| Lambda H100 training (4 runs × 30 min) | $6–10 |
| Lambda H100 ablations (3–4 short runs) | $4–6 |
| **Subtotal** | **~$35–45** |
| Buffer for mistakes | $10–15 |
| **Total project budget** | **~$50–60** |

**Worst-case scenarios to avoid:**
- Forgot to terminate Lambda for 24 hours → +$72
- Used Sonnet 4.6 instead of Haiku 4.5 for judge → +$20
- Bulk-synthesized 50K examples instead of 15K → +$30

---

## 12. Definition of done

### Per stage

| Stage | Done when |
|---|---|
| 1.5 | `data/interim/synth_gemini.pairs.jsonl` ≥ 15K examples; persona prompt locked; 50-example spot-check reads useful + witty + edged correctly |
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

## 13. Standing ops practices

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

### Lambda H100 discipline

1. **Calendar reminder 1 hour after launch** — verify instance terminated
2. Always `tmux new -s train` before starting training
3. Pull checkpoints back to local before terminating
4. Terminate via web UI (don't trust CLI)
5. Refresh dashboard after — confirm zero instances

### Reproducibility check after every checkpoint save

```python
loaded = AutoModelForCausalLM.from_pretrained(ckpt_path)
out = loaded.generate("Tell me a joke.", do_sample=False, max_new_tokens=50)
# Save to runs/<run_id>/parity_check.txt
```

If a re-load produces different greedy output, the save/load is
broken — fix before training more.

---

## 14. Useful commands

```bash
# === Project setup (after rename) ===
cd /Users/arunmanivannan/projects/ai/learn-you-an-sft
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

# === Lambda H100 ===
ssh ubuntu@<public-ip>
git clone https://github.com/<you>/learn-you-an-sft.git
cd learn-you-an-sft && uv pip install -r requirements.txt
tmux new -s train
uv run python -m runs.sft_v1_trl.train
# Ctrl-b d to detach
scp -r ubuntu@<public-ip>:~/learn-you-an-sft/runs/sft_v1_trl/checkpoints \
       runs/sft_v1_trl/checkpoints
# *** TERMINATE INSTANCE ON THE DASHBOARD ***

# === Git workflow ===
git add -A
git status
git commit -m "stage X: <what changed>"
git log --oneline
```

---

## 15. Failure modes to expect

| Symptom | Cause | Fix |
|---|---|---|
| Gemini API "quota exceeded" | Free tier limit (~15 RPM) | Spread across days, or pay-as-you-go |
| Gemini refuses profane requests | Safety filter | Roleplay frame: "dialogue for a sharp-tongued friend character…"; drop refusals at filter stage |
| Synth examples all sound the same | Persona prompt has no variety hooks | Add temperature variation + per-example random seed |
| MinHash uses 4 GB RAM | NUM_PERM × N too high | Lower NUM_PERM to 64; lower JACCARD_THRESHOLD to 0.6 |
| fasttext lid.176 not found | Skipped download step | `curl -L <url> -o ~/.cache/fasttext/lid.176.bin` |
| Loss NaN after few hundred steps | LR too high for bf16 | Drop LR to 1e-5, or fall back to fp32 |
| OOM on H100 step 0 | `torch.compile` first-step spike | Drop BATCH_SIZE 64 → 32 |
| Lambda instance left running overnight | Forgot to terminate | Calendar reminder discipline |
| Manifest SHA256 changes on re-run | Non-deterministic step | Seed every `random.Random()` and `np.random.seed()` |
| Judge gives different verdict on rerun | Temperature > 0 in judge call | Set temperature=0 |
| v1 (TRL) and v2 (hand-rolled) loss curves diverge | Loss mask / packing / template mismatch | Run Stage 3a round-trip test |
| Model says "tell me a joke" back to anything | Overfit to manufactured-prompt distribution | More variety in `question_pool.py` |
| Model is too polite (no edge) | Persona prompt too soft, or toxicity filter still applied | Re-read `persona_prompt.md`; confirm filter has no toxicity gate |
| Model is gratuitously cruel | Persona prompt over-corrected toward edge | Tighten "friend, not enemy" framing; add bright lines re: punching down |

### Things that aren't bugs but look like bugs

- First H100 step takes 60+ seconds — `torch.compile` tracing
- Stage 2b filter prints nothing for a minute on first run — fasttext model loading
- Gemini occasionally returns empty string — retry with backoff; drop after 3 failures

---

## 16. Glossary

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

## 17. How to bootstrap this in a new Claude context

Open a fresh Claude Code session in the renamed
`/Users/arunmanivannan/projects/ai/learn-you-an-sft` directory. Your
first message:

> I'm continuing the learn-you-an-sft project. Please read HANDOFF.md
> end-to-end and confirm:
> 1. The persona target (useful + witty + casually profane friend)
> 2. The current code state (cleanup done; pipeline.py + READMEs need edits per § 7b/c)
> 3. The Stage 1.5 work that's next
>
> Once you've confirmed, do these in order:
> a. Edit `data/filter/pipeline.py` to remove `quality_gate`,
>    `toxicity_gate`, and their argparse flags
> b. Rewrite `data/ingest/README.md` for synth-only
> c. Update `data/filter/README.md` and `data/README.md`
> d. Update `README.md` (top-level) deps list
> e. `git commit -m "stage 2b cleanup: drop quality/toxicity gates"`
>
> Then we'll build `synthesis/` together.

The new context will read HANDOFF.md, internalize the locked decisions,
and resume work without re-litigating the journey. If the new context
ever tries to suggest re-adding scraped corpora or removing profanity
from the persona, point it back to **§ 2 (the journey)** and **§ 5
(locked decisions)** — those debates are settled.

### What the new context does NOT need to re-read

- Old conversation history (this file is the bootstrap)
- The deleted `scrape/` directory
- The deleted scraped-source ingesters
- The deleted quality/toxicity filter modules

### What the new context SHOULD read

- This file (`HANDOFF.md`) — everything
- `data/filter/{normalize,language,dedup,pipeline}.py` — to understand
  the current filter pipeline before editing it
- `data/ingest/schema.py` — the canonical Pair schema
- `/Users/arunmanivannan/projects/ai/learn-you-an-hf-llm/runs/h100_tinystories/train.py`
  — the basis for Phase B's hand-rolled training loop

---

## Final note

The single most important thing in this project is `synthesis/persona_prompt.md`.
Its biases become the model's biases. Iterate against 5 hand-eval
examples before generating any bulk data.

The second most important thing is locking `eval/prompts.jsonl` on day
one of Stage 2.5 and never modifying it based on results. Cheating on
eval is invisible — you won't know you did it until the model fails
in real conversation.

Everything else is just code.

Tomorrow's first task: paste § 7a, run the rename + cleanup + git init,
then open a new Claude context with the prompt in § 17.
