"""Canonical (prompt, response) Pair schema + JSONL helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass
class Pair:
    prompt: str
    response: str
    source: str
    score: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_jsonl(pairs: Iterable[Pair], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for pair in pairs:
            f.write(pair.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Pair]:
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
