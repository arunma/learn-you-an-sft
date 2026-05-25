"""Canonical (prompt, response) pair schema used downstream by SFT.

Every producer of training data converts into this shape before any
filtering or training. Two reasons:

1. The training code only ever sees one schema, regardless of how
   many producers you stack.
2. Producer-specific quirks live in the producer, where they belong,
   not in the training loop.

The `score` field is optional and producer-defined: a value in [0, 1]
if the producer has a quality signal, otherwise None. Stage 2b can
threshold on it when present.
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
