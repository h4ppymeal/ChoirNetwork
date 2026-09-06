# Observed-service development results

Command:

```bash
python -m choirnetwork eval --top-k 5 --split development
```

Run on September 7, 2026 using 66 development queries from
`eval/datasets/service_hymns.csv`. LLM expansion was disabled. The 40-query
test split was not evaluated.

| Configuration | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 13.6% | 6.8% | 7.8% | 5.7% |
| Bi-encoder only | 10.6% | 6.8% | 5.7% | 5.3% |
| + cross-encoder reranking | 12.1% | 7.6% | 6.3% | 5.7% |
| + curated expansion | 12.1% | 7.6% | 6.3% | 5.7% |
| + lyric keyword boost (full system) | 13.6% | 8.3% | 6.4% | 6.1% |

## Interpretation

The observed-service labels are difficult because the input contains only an
often-abstract sermon title. The full system only slightly exceeds BM25 on
nDCG@5, while the plain bi-encoder trails BM25. Curated expansion has no
measurable effect because most sermon titles do not match the small checked-in
topic map.

These labels measure whether retrieval reproduces two historical human
selections. They do not establish that all other hymns are irrelevant, and a
service hymn may reflect liturgical context or personal preference that is not
expressed in the sermon title.

Do not run or publish the 40-query test split until the retrieval configuration
and evaluation interpretation are frozen.
