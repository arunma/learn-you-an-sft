"""Quality-gate the training corpus.

One operation, three steps:
  1. Score every (prompt, response) pair in data/processed/{train,val}.jsonl
     with Claude Haiku against the five-axis persona rubric.
  2. Combine the two splits and filter to passes_all rows.
  3. Re-split into train/val (stable shuffle).

If data/handcrafted_extras.jsonl exists, those rows bypass the filter and are
added directly to the kept pool — useful for hand-crafted corrections that
shouldn't have to pass the judge.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

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

CRITERIA = (
    "on_persona",
    "uses_profanity_appropriately",
    "takes_stance",
    "is_helpful",
    "factual_floor",
)

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
    """Five binary criteria + a short rationale."""
    on_persona: bool
    uses_profanity_appropriately: bool
    takes_stance: bool
    is_helpful: bool
    factual_floor: bool
    rationale: str = Field(
        description="One short sentence on the most notable issue, or 'all pass' if none.",
    )

    @property
    def pass_count(self) -> int:
        return sum([
            self.on_persona,
            self.uses_profanity_appropriately,
            self.takes_stance,
            self.is_helpful,
            self.factual_floor,
        ])

    @property
    def passes_all(self) -> bool:
        return self.pass_count == 5

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["pass_count"] = self.pass_count
        d["passes_all"] = self.passes_all
        return d


@dataclass(frozen=True)
class JudgeResult:
    index: int
    score: PersonaScore | None
    error: str | None


async def _judge_one(
    client: "instructor.AsyncInstructor",
    index: int,
    prompt: str,
    response: str,
) -> JudgeResult:
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
        return JudgeResult(index=index, score=score, error=None)
    except Exception as exc:
        return JudgeResult(
            index=index,
            score=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _judge_pairs(pairs: Sequence[tuple[str, str]]) -> list[JudgeResult]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export before running."
        )

    client = instructor.from_anthropic(AsyncAnthropic())
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def judge(i: int, prompt: str, response: str) -> JudgeResult:
        async with sem:
            return await _judge_one(client, i, prompt, response)

    print(
        f"  Judging {len(pairs)} pairs with {JUDGE_MODEL} "
        f"(concurrency={JUDGE_CONCURRENCY})..."
    )
    tasks = [judge(i, p, r) for i, (p, r) in enumerate(pairs)]
    results = await atqdm.gather(*tasks, desc="judging")

    failures = sum(1 for r in results if r.score is None)
    if failures:
        print(f"  {failures} judge failures (see error field in output)")
    return results


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z").replace(":", "-")


def _attach_eval(df: pd.DataFrame, results: list[JudgeResult]) -> pd.DataFrame:
    rows = df.to_dict("records")
    out: list[dict] = []
    for i, (row, r) in enumerate(zip(rows, results)):
        merged = dict(row)
        merged["index"] = i
        if r.score is not None:
            merged["eval"] = r.score.to_dict()
            merged["error"] = None
        else:
            merged["eval"] = None
            merged["error"] = r.error
        out.append(merged)
    return pd.DataFrame(out)


def _summarize(df_scored: pd.DataFrame) -> dict:
    judged = df_scored[df_scored["eval"].notna()]
    n_judged = len(judged)
    if n_judged == 0:
        return {
            "judged": 0,
            "judge_failures": int(len(df_scored)),
            "pass_rates_per_criterion": {c: 0.0 for c in CRITERIA},
            "passes_all_count": 0,
            "passes_all_rate": 0.0,
            "pass_count_histogram": {str(k): 0 for k in range(6)},
        }
    flat = pd.json_normalize(judged["eval"])
    hist = flat["pass_count"].value_counts().sort_index().to_dict()
    return {
        "judged": int(n_judged),
        "judge_failures": int(len(df_scored) - n_judged),
        "pass_rates_per_criterion": {
            c: round(float(flat[c].mean()), 4) for c in CRITERIA
        },
        "passes_all_count": int(flat["passes_all"].sum()),
        "passes_all_rate": round(float(flat["passes_all"].mean()), 4),
        "pass_count_histogram": {str(k): int(hist.get(k, 0)) for k in range(6)},
    }


def _print_summary(label: str, summary: dict) -> None:
    print(f"\n=== Summary ({label}) ===")
    print(f"  judged:           {summary['judged']}")
    print(f"  judge failures:   {summary['judge_failures']}")
    print(f"  passes_all rate:  {summary['passes_all_rate']:.1%}")
    for c, rate in summary["pass_rates_per_criterion"].items():
        print(f"    {c:32s} {rate:.1%}")


def run_score_and_split() -> None:
    if not CLEANED_PATH.exists():
        raise SystemExit(
            f"Need {CLEANED_PATH}. Run `python -m prep filter` first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(CLEANED_PATH, lines=True)
    print(f"Loaded {len(df)} cleaned pairs")

    pairs = list(zip(df["prompt"], df["response"]))
    results = asyncio.run(_judge_pairs(pairs))

    df_scored = _attach_eval(df, results)
    summary = _summarize(df_scored)
    _print_summary("scored corpus", summary)

    safe_ts = _utc_stamp()
    scored_path = REPORTS_DIR / f"dataset_scored_{safe_ts}.jsonl"
    df_scored.to_json(scored_path, orient="records", lines=True, force_ascii=False)
    summary_path = REPORTS_DIR / f"dataset_summary_{safe_ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote scored rows -> {scored_path}")
    print(f"Wrote summary    -> {summary_path}")

    passes_all = df_scored["eval"].apply(
        lambda e: e is not None and e.get("passes_all", False)
    )
    df_kept = df_scored.loc[passes_all].copy()
    print(f"\nKept (passes_all): {len(df_kept)} / {len(df_scored)}")

    n_extras = 0
    if EXTRAS_PATH.exists():
        df_extras = pd.read_json(EXTRAS_PATH, lines=True)
        n_extras = len(df_extras)
        print(f"Adding {n_extras} pre-passed extras from {EXTRAS_PATH.name}")
        df_kept = pd.concat([df_kept, df_extras], ignore_index=True)

    if df_kept.empty:
        raise SystemExit("No rows passed the filter; nothing to write.")

    pair_cols = [
        c for c in ("prompt", "response", "source", "score", "meta")
        if c in df_kept.columns
    ]
    df_kept = (
        df_kept[pair_cols]
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    n_val = max(1, int(round(len(df_kept) * VAL_FRACTION)))
    df_val_out = df_kept.head(n_val).reset_index(drop=True)
    df_train_out = df_kept.iloc[n_val:].reset_index(drop=True)

    df_train_out.to_json(TRAIN_PATH, orient="records", lines=True, force_ascii=False)
    df_val_out.to_json(VAL_PATH, orient="records", lines=True, force_ascii=False)

    print("\n=== Split (passes_all + extras) ===")
    print(f"  val fraction:  {VAL_FRACTION:.0%}  (seed={SEED})")
    if n_extras:
        print(f"  extras:        +{n_extras} rows (bypass filter)")
    print(f"  train: {len(df_train_out):>5d} rows -> {TRAIN_PATH}")
    print(f"  val:   {len(df_val_out):>5d} rows -> {VAL_PATH}")
