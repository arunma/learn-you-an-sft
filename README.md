# learn-you-an-sft

End-to-end pipeline for fine-tuning a 4B language model into a specific
persona via LoRA and a synthesised training corpus. Worked example included:
**Monty**, a foul-mouthed opinionated friend.

- **Step-by-step tuning guide:** [TUTORIAL.md](TUTORIAL.md)
- **The story behind it:** [Costume vs Character: Fine-tuning Qwen into Monty for $35 →](https://arunma.com/costume-vs-character-fine-tuning-qwen-into-monty-for-35/)
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
   the system prompt from `runs/sft_v1_trl/train.py:SYSTEM`. ~7.5 GB. Done.
2. **Train your own.** Edit `synthesis/persona_prompt.md` for your character,
   then follow [TUTORIAL.md](TUTORIAL.md) end-to-end. ~2-3 hours of GPU time
   plus iteration.

---

## Repo layout

| Path | What's in it |
|---|---|
| `synthesis/` | Persona prompt + Gemini distillation (Pro for pairs, Flash for question pool) |
| `data/ingest/` | `Pair` schema, JSONL I/O |
| `data/filter/` | Normalise → language filter → dedup → split |
| `data/processed/` | Final training corpus (`train.jsonl`, `val.jsonl`, `manifest.json`) — committed as a worked example |
| `runs/sft_v1_trl/train.py` | LoRA SFT via TRL on Qwen3-4B-Instruct |
| `eval/` | Haiku-as-judge five-axis binary rubric, dataset scoring, model eval, repartition |
| `inference/merge_for_gguf.py` | Merge adapter into base for GGUF export |

---

## The five-axis rubric

The eval and dataset-scoring steps both use the same binary multi-criteria
rubric (see `eval/rubric.py`). A response **passes_all** iff every criterion
below is yes.

| Criterion | What it asks |
|---|---|
| `on_persona` | Does the response sound like the character? |
| `uses_profanity_appropriately` | Casual swearing as rhythm, not gratuitous; voice-off for crisis prompts |
| `takes_stance` | Clear position, no hedging |
| `is_helpful` | Actually useful to the asker |
| `factual_floor` | Free of slurs, dangerous advice, catastrophic hallucination |

Binary > Likert. Judges are noisy on 1–5 scales and reliable on yes/no.

---

## License

MIT.
