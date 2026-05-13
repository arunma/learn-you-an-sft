# `data/ingest/` — the canonical `Pair` schema

This used to be where four scraped sources (bash.org, rJokesData,
SARC, dad jokes) got converted into a unified shape before filtering.
After the 2026-05-13 pivot to **pure synth**, those source-specific
ingesters are gone. What remains is the schema itself — the one shape
every downstream stage agrees on.

## The `Pair` shape

```python
@dataclass
class Pair:
    prompt:   str                    # what the user (or synthesized question) asks
    response: str                    # the witty-friend reply
    source:   str                    # e.g. "gemini_synth", "gemini_synth_v2"
    score:    Optional[float] = None # per-pair quality score in [0, 1] if available
    meta:     dict = field(default_factory=dict)  # everything else — model id, seed, etc.
```

Why one schema across the whole pipeline:

1. The training loop (Stage 4) only ever sees `Pair`s. It never has
   to branch on "is this from synthesis run A or B?" — that info
   lives in `source` and `meta`, opaque to training.
2. Filters (Stage 2b) operate on `Pair` regardless of provenance.
3. Adding a new synthesis variant (different teacher, different
   persona prompt) is just another value in `source` — no schema
   change, no downstream rewrite.

## Files

| File | What it does |
|---|---|
| `schema.py` | Defines `Pair` and JSONL read/write helpers. |

The producers of `Pair`s now live in `synthesis/` (Stage 1.5, to be
built). Each synthesis run writes one `data/interim/<run_id>.pairs.jsonl`
file, which Stage 2b filters consume.

## Output convention

```
data/interim/
  └── gemini_synth_v0.pairs.jsonl       ← one file per synth run
  └── gemini_synth_v1.pairs.jsonl
```

One file per run keeps the `manifest.json` interpretable ("which synth
runs are in this training set?") and lets you re-filter without
re-synthesizing.

## What you'll learn

- **Schema-first design.** One canonical shape; every upstream quirk
  handled in producers, not consumers. The training loop never sees
  source-specific code.
- **Provenance as a first-class field.** `source` + `meta` survive
  the whole pipeline, so a single line in `train.jsonl` tells you
  which run, which model, which prompt version produced it.
- **Streaming JSONL.** `write_jsonl` / `read_jsonl` work pair by
  pair, never materialising the corpus in memory. Necessary discipline
  for any data pipeline that might grow.
