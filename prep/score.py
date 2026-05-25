"""Score the cleaned corpus with Haiku, keep passes_all rows, split train/val."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import instructor
import pandas as pd
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as atqdm

from prep import DATA_DIR, PROCESSED_DIR

load_dotenv()


CLEANED_PATH = PROCESSED_DIR / "cleaned.jsonl"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "val.jsonl"
EXTRAS_PATH = DATA_DIR / "handcrafted_extras.jsonl"
REPORTS_DIR = PROCESSED_DIR / "eval_reports"

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MAX_TOKENS = 400
JUDGE_CONCURRENCY = 10
JUDGE_MAX_RETRIES = 3

VAL_FRACTION = 0.05
SEED = 42

JUDGE_SYSTEM_PROMPT = """\
You are evaluating responses from a fine-tuned LLM with a specific persona called "Monty".

Monty's voice:
- A foul-mouthed, opinionated friend — not a customer-service chatbot
- Swears casually and often (fuck, shit, bullshit, knackered, arsed) as rhythm, not shock
- Lowercase by default; proper nouns, acronyms, and "I" capitalised
- Takes a clear stance — no "it depends" hedging, no both-sides equivocation
- Genuinely helpful under the grumpiness; the swearing is texture, not refusal
- Concrete examples, metaphors, and the occasional fake-attributed quote
  ("as my nan used to say") over abstractions

Hard limits Monty respects:
- No slurs (race, gender, sexuality, religion, disability)
- No mocking the user for what they don't know
- Crisis prompts (suicidal ideation, active self-harm, immediate danger): voice off
  entirely — no swearing, no jokes, sober and present, lowercase preserved

Return a strict yes/no verdict on each of the five criteria and a one-sentence rationale.
"""

JUDGE_USER_TEMPLATE = """\
User prompt:
<prompt>{prompt}</prompt>

Model response:
<response>{response}</response>

Score each criterion as a boolean:

- on_persona: Does the response sound like Monty — foul-mouthed, opinionated, lowercase-by-default, friend-not-service tone?
- uses_profanity_appropriately: Casual profanity used as rhythm/texture (not forced, not gratuitous)? For crisis prompts, true if appropriately sober and voice-off. For short banter or trivial factual asks, true if profanity is absent but the response still feels in voice.
- takes_stance: Does the response take a clear position rather than hedging or both-sidesing?
- is_helpful: Does the response actually help the user (answer, opinion, useful action)?
- factual_floor: Free of slurs, dangerous advice, and catastrophic hallucination?
"""


class PersonaScore(BaseModel):
    on_persona: bool
    uses_profanity_appropriately: bool
    takes_stance: bool
    is_helpful: bool
    factual_floor: bool
    rationale: str = Field(
        description="One short sentence on the most notable issue, or 'all pass' if none.",
    )

    def to_dict(self) -> dict:
        d = self.model_dump()
        bools = [v for v in d.values() if isinstance(v, bool)]
        d["pass_count"] = sum(bools)
        d["passes_all"] = all(bools)
        return d


CRITERIA = tuple(
    name for name, field in PersonaScore.model_fields.items()
    if field.annotation is bool
)


@dataclass(frozen=True)
class JudgeResult:
    score: PersonaScore | None
    error: str | None


async def _judge_one(client, sem, prompt, response) -> JudgeResult:
    async with sem:
        try:
            score = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response),
                }],
                response_model=PersonaScore,
                max_retries=JUDGE_MAX_RETRIES,
            )
            return JudgeResult(score=score, error=None)
        except Exception as exc:
            return JudgeResult(score=None, error=f"{type(exc).__name__}: {exc}")


async def _judge(pairs) -> list[JudgeResult]:
    client = instructor.from_anthropic(AsyncAnthropic())
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    tasks = [_judge_one(client, sem, p, r) for p, r in pairs]
    return await atqdm.gather(*tasks, desc="judging")


def _summarize(df: pd.DataFrame) -> dict:
    judged = df[df["eval"].notna()]
    if judged.empty:
        return {
            "judged": 0,
            "judge_failures": len(df),
            "pass_rates_per_criterion": {c: 0.0 for c in CRITERIA},
            "passes_all_count": 0,
            "passes_all_rate": 0.0,
            "pass_count_histogram": {k: 0 for k in range(len(CRITERIA) + 1)},
        }
    flat = pd.json_normalize(judged["eval"])
    hist = flat["pass_count"].value_counts().to_dict()
    return {
        "judged": len(judged),
        "judge_failures": len(df) - len(judged),
        "pass_rates_per_criterion": {c: float(flat[c].mean()) for c in CRITERIA},
        "passes_all_count": int(flat["passes_all"].sum()),
        "passes_all_rate": float(flat["passes_all"].mean()),
        "pass_count_histogram": {
            k: int(hist.get(k, 0)) for k in range(len(CRITERIA) + 1)
        },
    }


def run_score_and_split() -> None:
    if not CLEANED_PATH.exists():
        raise SystemExit(f"missing {CLEANED_PATH} — run `python -m prep filter` first")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — add it to .env")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(CLEANED_PATH, lines=True)
    pairs = list(zip(df["prompt"], df["response"]))
    results = asyncio.run(_judge(pairs))

    df_scored = pd.DataFrame([
        {
            **row,
            "index": i,
            "eval": r.score.to_dict() if r.score else None,
            "error": r.error,
        }
        for i, (row, r) in enumerate(zip(df.to_dict("records"), results))
    ])
    summary = _summarize(df_scored)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scored_path = REPORTS_DIR / f"dataset_scored_{stamp}.jsonl"
    df_scored.to_json(scored_path, orient="records", lines=True, force_ascii=False)
    summary_path = REPORTS_DIR / f"dataset_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(scored_path)
    print(summary_path)
    print(f"passes_all: {summary['passes_all_rate']:.1%}")
    for c, rate in summary["pass_rates_per_criterion"].items():
        print(f"  {c}: {rate:.1%}")

    passes_all = df_scored["eval"].apply(
        lambda e: e is not None and e.get("passes_all", False)
    )
    df_kept = df_scored.loc[passes_all].copy()
    print(f"kept: {len(df_kept)}/{len(df_scored)}")

    if EXTRAS_PATH.exists():
        df_extras = pd.read_json(EXTRAS_PATH, lines=True)
        print(f"+{len(df_extras)} extras from {EXTRAS_PATH.name}")
        df_kept = pd.concat([df_kept, df_extras], ignore_index=True)

    if df_kept.empty:
        raise SystemExit("no rows passed the filter")

    pair_cols = [
        c for c in ("prompt", "response", "source", "score", "meta")
        if c in df_kept.columns
    ]
    df_kept = (
        df_kept[pair_cols]
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    n_val = max(1, round(len(df_kept) * VAL_FRACTION))
    df_val_out = df_kept.head(n_val).reset_index(drop=True)
    df_train_out = df_kept.iloc[n_val:].reset_index(drop=True)

    df_train_out.to_json(TRAIN_PATH, orient="records", lines=True, force_ascii=False)
    df_val_out.to_json(VAL_PATH, orient="records", lines=True, force_ascii=False)
    print(f"train: {len(df_train_out)} -> {TRAIN_PATH}")
    print(f"val:   {len(df_val_out)} -> {VAL_PATH}")
