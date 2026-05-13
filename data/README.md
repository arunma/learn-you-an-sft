# `data/` — corpus pipeline

Three subdirectories, one direction of flow:

```
raw/        ← unused in the pure-synth era                              (kept as a hook)

interim/    ← Pair JSONL from synthesis runs                            (Stage 1.5 — synthesis/)
  └── gemini_synth_v0.pairs.jsonl, gemini_synth_v1.pairs.jsonl, ...

processed/  ← filtered, deduped, split into train/val                   (Stage 2b — filter/)
  └── train.jsonl, val.jsonl, manifest.json
```

`raw/` is unused for now. Kept as a hook in case we later want to
mix in hand-curated data (a small set the user hand-edits) alongside
the synth corpus.

`interim/` is fully reproducible from a synthesis run + its seed +
the persona prompt. Don't commit it.

`processed/` is the only directory training reads from. Don't commit
either — it's regeneratable from `interim/` + the filter config.

## See also

- `ingest/README.md` — the canonical `Pair` schema (now schema-only, no source-specific code)
- `filter/README.md` — Stage 2b: normalize → language → dedup → train/val split + manifest
