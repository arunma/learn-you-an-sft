"""Generate persona responses with the trained adapter and judge them with Haiku."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import instructor
import pandas as pd
import torch
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from peft import PeftModel
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
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


@dataclass(frozen=True)
class Generation:
    prompt: str
    gold: str
    response: str


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, pairs) -> list[Generation]:
    device = next(model.parameters()).device
    out: list[Generation] = []
    for prompt, gold in tqdm(pairs, desc="generating"):
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
        response = tokenizer.decode(
            gen[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        out.append(Generation(prompt=prompt, gold=gold, response=response))
    return out


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
        zeros = {c: 0.0 for c in CRITERIA}
        return {
            "judged": 0,
            "judge_failures": len(df),
            "pass_rates_per_criterion": zeros,
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


async def _run() -> None:
    if not VAL_PATH.exists():
        raise SystemExit(f"missing {VAL_PATH} — run `python -m prep` first")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_val = pd.read_json(VAL_PATH, lines=True)
    model, tokenizer = _load_model()
    generations = _generate(
        model, tokenizer, list(zip(df_val["prompt"], df_val["response"]))
    )
    judged = await _judge([(g.prompt, g.response) for g in generations])

    df = pd.DataFrame([
        {
            "index": i,
            "prompt": g.prompt,
            "gold": g.gold,
            "response": g.response,
            "eval": r.score.to_dict() if r.score else None,
            "error": r.error,
        }
        for i, (g, r) in enumerate(zip(generations, judged))
    ])

    now = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"model_eval_{stamp}.jsonl"
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    summary = _summarize(df) | {
        "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
        "base_model": BASE_MODEL,
        "adapter": ADAPTER,
        "judge_model": JUDGE_MODEL,
        "temperature": GEN_TEMPERATURE,
        "top_p": GEN_TOP_P,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "prompts": len(df_val),
    }
    summary_path = REPORTS_DIR / f"model_eval_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(out_path)
    print(summary_path)
    print(f"passes_all: {summary['passes_all_rate']:.1%}")
    for c, rate in summary["pass_rates_per_criterion"].items():
        print(f"  {c}: {rate:.1%}")


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — add it to .env")
    asyncio.run(_run())
