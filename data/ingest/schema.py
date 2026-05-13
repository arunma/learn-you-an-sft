"""Canonical (prompt, response) pair schema used downstream by SFT.

Every source — bash.org, rJokesData, SARC, dad jokes — gets converted
into this shape before any filtering or training. Two reasons:

1. The training code only ever sees one schema, regardless of how
   many sources you stack.
2. Source-specific quirks live in source-specific ingesters, where they
   belong, not in the training loop.

The `score` field is the most important design choice here: it's
normalised to [0, 1] across sources so the filter pipeline (Stage 2b)
can use a single threshold across the whole mix. Each ingester is
responsible for its own normalisation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass
class Pair:
    """One (prompt, response) training pair, with provenance."""
    prompt: str
    response: str
    source: str
    score: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        # ensure_ascii=False so unicode (emoji, non-English) survives
        # without becoming \uXXXX escape soup.
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(pairs: Iterable[Pair], path: Path) -> int:
    """Stream-write pairs to JSONL. Returns the number of pairs written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for pair in pairs:
            f.write(pair.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Pair]:
    """Stream-read pairs from JSONL. Skips malformed lines silently."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield Pair(**obj)
