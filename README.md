# learn-you-an-sft

End-to-end pipeline for fine-tuning a 4B language model into a specific
persona via LoRA and a synthesised training corpus. Worked example included:
**Monty**, a foul-mouthed opinionated friend.

- **The story:** [Costume vs Character: Fine-tuning Qwen into Monty for $35 →](https://arunma.com/costume-vs-character-fine-tuning-qwen-into-monty-for-35/)
- **The how-to, part 1 — the data:** [Persona Fine-Tuning, Part 1: The Data →](https://arunma.com/fine-tuning-a-4b-model-into-your-own-persona-part-1-the-data/)
- **The how-to, part 2 — training & ship:** [Persona Fine-Tuning, Part 2: Training, Eval, Ship →](https://arunma.com/fine-tuning-a-4b-model-into-your-own-persona-part-2-training-eval-ship/)
- **Trained adapter + GGUFs:** [`arunma/monty3`](https://huggingface.co/arunma/monty3)

---

## Quickstart

```bash
git clone <this-repo>
cd learn-you-an-sft
uv sync
cp .env.example .env
# Fill in HF_TOKEN, ANTHROPIC_API_KEY, GEMINI_API_KEY.
```

Then either:

1. **Try Monty.** Pull `arunma/monty3`'s f16 GGUF, drop it in LM Studio with
   the system prompt from `run/train.py:SYSTEM`. ~7.5 GB. Done.
2. **Train your own.** Edit `persona_prompt.md` for your character, then
   follow the two how-to blog posts linked above end-to-end. ~2-3 hours of
   GPU time plus iteration.

---

## Repo layout

| Path | What's in it |
|---|---|
| `persona_prompt.md` | The character definition (you edit this) |
| `prep/` | Data prep — `distill` (Gemini), `filter` (pandas), `score` (Haiku judge + split) |
| `run/` | Model lifecycle — `train` (LoRA SFT), `eval` (gen + judge), `merge` (adapter → HF) |
| `data/` | Gemini synth + cleaned + train/val (gitignored) |
| `runs/` | Training checkpoints + eval reports (gitignored) |

---

## Commands

```bash
# Build the corpus
uv run python -m prep                       # all stages in order
uv run python -m prep questions             # Gemini Flash → data/interim/question_pool.jsonl
uv run python -m prep answers               # Gemini Pro → data/interim/gemini_synth_v0.pairs.jsonl
uv run python -m prep filter                # → data/processed/cleaned.jsonl
uv run python -m prep score-and-split       # judge + passes_all filter + train/val split

# Train, evaluate, ship
uv run python -m run train                  # LoRA SFT on data/processed/{train,val}.jsonl
uv run python -m run eval                   # generate + judge against the trained adapter
uv run python -m run merge                  # merge adapter → HF format (then convert to GGUF)
```

No flags. Want a different concurrency, adapter, or model? Edit the
constants at the top of the relevant module — that's the whole knob-tuning
interface.

To run the eval end-to-end on a rented GPU (spin up → eval → scp reports
back → terminate), set `RUNPOD_API_KEY` in `.env` then:

```bash
uv run python -m scripts.eval_on_pod
```

Reports land in `runs/eval_reports/`. Pod terminates in a `finally` block,
so a crashed eval doesn't leave a billable GPU running.

---

## The five-axis rubric

Same binary multi-criteria rubric is used twice: once in `prep score-and-split`
to gate the training corpus, once in `run eval` to grade the trained model.
`PersonaScore` (Pydantic) is the schema; Claude Haiku is the judge via
[Instructor](https://python.useinstructor.com/).

| Criterion | What it asks |
|---|---|
| `on_persona` | Does the response sound like the character? |
| `uses_profanity_appropriately` | Casual swearing as rhythm, not gratuitous; voice-off for crisis prompts |
| `takes_stance` | Clear position, no hedging |
| `is_helpful` | Actually useful to the asker |
| `factual_floor` | Free of slurs, dangerous advice, catastrophic hallucination |

A response **passes_all** iff every criterion is yes. Binary > Likert —
judges are noisy on 1-5 scales and reliable on yes/no.

---

## License

MIT.
