# Evaluation set

Labeled queries for offline retrieval benchmarking. Used by `python -m choirnetwork eval`.

## Format

Each line in `queries.jsonl` is a JSON object:

```json
{
  "query": "Ruth and Naomi",
  "relevant_titles": ["Trust", "Guide", "Redeem", "Providence"],
  "category": "bible_narrative",
  "notes": "Loyalty and providence themes"
}
```

- **`relevant_titles`** — substring-matched against hymn titles in the built index
- **`relevant_slugs`** — optional exact slug matches (e.g. `"127"`)
- **`category`** — `hymn_title`, `sermon_topic`, or `bible_narrative`

Queries with no matching slugs in the index are skipped at eval time.

## Adding labels

1. Run a search and note good hymn picks
2. Add a row to `queries.jsonl` with `relevant_titles` fragments
3. Re-run `python -m choirnetwork eval --top-k 5`

## Metrics

See [`choirnetwork/ENGINE.md`](../choirnetwork/ENGINE.md#offline-evaluation) for Hit@k, Recall@k, MRR@k, and nDCG@k definitions.
