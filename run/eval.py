"""Generate persona responses with the trained adapter and judge them with Haiku."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import instructor
import pandas as pd
import torch
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from peft import PeftModel
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prep import PROCESSED_DIR, RUNS_DIR
from run.train import SYSTEM


VAL_PATH = PROCESSED_DIR / "val.jsonl"
REPORTS_DIR = RUNS_DIR / "eval_reports"

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = "arunma/monty3"

GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOKENS = 512

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MAX_TOKENS = 400
JUDGE_CONCURRENCY = 10
JUDGE_MAX_RETRIES = 3

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


@dataclass(frozen=True)
class Generation:
    prompt: str
    gold: str
    response: str


def _utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    iso = now.isoformat().replace("+00:00", "Z")
    return iso.replace(":", "-"), iso


def _load_model():
    print(f"Loading {BASE_MODEL} + adapter {ADAPTER}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return model, tokenizer


def _generate(
    model, tokenizer, pairs: Sequence[tuple[str, str]],
) -> list[Generation]:
    out: list[Generation] = []
    total = len(pairs)
    device = next(model.parameters()).device
    start = time.monotonic()
    print(
        f"  Generating {total} responses "
        f"(T={GEN_TEMPERATURE}, top_p={GEN_TOP_P})..."
    )

    for i, (prompt, gold) in enumerate(pairs):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = gen[0][input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        out.append(Generation(prompt=prompt, gold=gold, response=response))

        done = i + 1
        if done % 10 == 0 or done == total:
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 else 0.0
            print(
                f"  generated {done}/{total}  "
                f"({elapsed:.0f}s elapsed, {rate:.2f} prompts/s, "
                f"~{remaining:.0f}s remaining)",
                flush=True,
            )
    return out


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


def _summarize(df: pd.DataFrame) -> dict:
    judged = df[df["eval"].notna()]
    n_judged = len(judged)
    if n_judged == 0:
        return {
            "judged": 0,
            "judge_failures": int(len(df)),
            "pass_rates_per_criterion": {c: 0.0 for c in CRITERIA},
            "passes_all_count": 0,
            "passes_all_rate": 0.0,
            "pass_count_histogram": {str(k): 0 for k in range(6)},
        }
    flat = pd.json_normalize(judged["eval"])
    hist = flat["pass_count"].value_counts().sort_index().to_dict()
    return {
        "judged": int(n_judged),
        "judge_failures": int(len(df) - n_judged),
        "pass_rates_per_criterion": {
            c: round(float(flat[c].mean()), 4) for c in CRITERIA
        },
        "passes_all_count": int(flat["passes_all"].sum()),
        "passes_all_rate": round(float(flat["passes_all"].mean()), 4),
        "pass_count_histogram": {str(k): int(hist.get(k, 0)) for k in range(6)},
    }


async def _run() -> int:
    if not VAL_PATH.exists():
        raise SystemExit(
            f"Val file not found: {VAL_PATH}. Run `python -m prep` first."
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_val = pd.read_json(VAL_PATH, lines=True)
    print(f"Reading val: {VAL_PATH}  ({len(df_val)} prompts)")

    model, tokenizer = _load_model()
    pairs = list(zip(df_val["prompt"], df_val["response"]))
    generations = _generate(model, tokenizer, pairs)

    judge_inputs = [(g.prompt, g.response) for g in generations]
    judge_results = await _judge_pairs(judge_inputs)

    rows: list[dict] = []
    for i, (g, r) in enumerate(zip(generations, judge_results)):
        row = {
            "index": i,
            "prompt": g.prompt,
            "gold": g.gold,
            "response": g.response,
        }
        if r.score is not None:
            row["eval"] = r.score.to_dict()
            row["error"] = None
        else:
            row["eval"] = None
            row["error"] = r.error
        rows.append(row)
    df = pd.DataFrame(rows)

    safe_ts, iso_ts = _utc_stamp()
    out_path = REPORTS_DIR / f"model_eval_{safe_ts}.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"\nWrote per-prompt eval -> {out_path}")

    summary = _summarize(df)
    summary.update({
        "timestamp_utc": iso_ts,
        "base_model": BASE_MODEL,
        "adapter": ADAPTER,
        "judge_model": JUDGE_MODEL,
        "temperature": GEN_TEMPERATURE,
        "top_p": GEN_TOP_P,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "prompts": int(len(df_val)),
    })
    summary_path = REPORTS_DIR / f"model_eval_summary_{safe_ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote summary         -> {summary_path}")

    print("\n=== Model eval summary ===")
    print(f"  base:           {BASE_MODEL}")
    print(f"  adapter:        {ADAPTER}")
    print(f"  judged:         {summary['judged']}")
    print(f"  judge failures: {summary['judge_failures']}")
    print(f"  passes_all:     {summary['passes_all_rate']:.1%}")
    for c, rate in summary["pass_rates_per_criterion"].items():
        print(f"    {c:32s} {rate:.1%}")
    return 0


def main() -> None:
    load_dotenv()
    asyncio.run(_run())
