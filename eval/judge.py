"""Async Haiku judge: scores (prompt, response) pairs with bounded concurrency + retries.

Reads ANTHROPIC_API_KEY from the environment (or .env, if loaded by the caller).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Sequence

from anthropic import AsyncAnthropic, APIError

from eval.rubric import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    JudgeParseError,
    PersonaScore,
    parse_judge_response,
)

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 400
DEFAULT_CONCURRENCY = 5
DEFAULT_MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds; exponential backoff


@dataclass(frozen=True)
class JudgeResult:
    index: int
    score: PersonaScore | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.score is not None


async def _judge_one(
    client: AsyncAnthropic,
    index: int,
    prompt: str,
    response: str,
    *,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> JudgeResult:
    user_msg = JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response)

    last_err: str | None = None
    for attempt in range(max_retries):
        try:
            msg = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            )
            score = parse_judge_response(text)
            return JudgeResult(index=index, score=score, error=None)
        except JudgeParseError as exc:
            last_err = f"parse error (attempt {attempt + 1}): {exc}"
        except APIError as exc:
            last_err = f"api error (attempt {attempt + 1}): {exc}"
        except Exception as exc:
            last_err = f"unexpected error (attempt {attempt + 1}): {type(exc).__name__}: {exc}"

        if attempt < max_retries - 1:
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))

    return JudgeResult(index=index, score=None, error=last_err)


async def judge_pairs_async(
    pairs: Sequence[tuple[str, str]],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress: bool = True,
) -> list[JudgeResult]:
    """Judge a batch of (prompt, response) pairs. Returns results in input order."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it before running."
        )

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    total = len(pairs)
    completed = 0
    failures = 0
    lock = asyncio.Lock()

    async def bounded(idx: int, prompt: str, response: str) -> JudgeResult:
        nonlocal completed, failures
        async with sem:
            result = await _judge_one(
                client,
                idx,
                prompt,
                response,
                model=model,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        async with lock:
            completed += 1
            if not result.ok:
                failures += 1
            if progress and (completed % 50 == 0 or completed == total):
                print(
                    f"  judged {completed}/{total} ({failures} failures so far)",
                    flush=True,
                )
        return result

    tasks = [bounded(i, p, r) for i, (p, r) in enumerate(pairs)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r.index)
