# `data/` — corpus pipeline

Three subdirectories, one direction of flow:

```
raw/        ← downloaded sources, untouched          (Stage 1 + manual download)
  └── bash_org.jsonl, rjokes/, sarc.flat.jsonl, dadjokes.jsonl

interim/    ← unified Pair schema, per source        (Stage 2a — ingest/)
  └── bash_org.pairs.jsonl, rjokes.pairs.jsonl, sarc.pairs.jsonl, dadjokes.pairs.jsonl

processed/  ← filtered, deduped, quality-scored, split into train/val   (Stage 2b — coming next)
  └── train.jsonl, val.jsonl, manifest.json
```

Never edit files under `raw/`. If a source's format changes upstream,
delete and re-download — that way the manifest's content hashes stay
honest.

`interim/` is fully reproducible from `raw/` by re-running the ingest
scripts. Cheap to regenerate. Don't commit it.

`processed/` is the only directory training reads from.

## See also

- `ingest/README.md` — Stage 2a: source ingestion (this is built)
- (coming next) Stage 2b README — filtering, dedup, quality, toxicity
