# Observed-service development results

Command:

```bash
python -m choirnetwork eval --top-k 5 --split development
```

Run on September 7, 2026 using 66 development queries from
`eval/datasets/service_hymns.csv`. Curated and LLM expansion were disabled.
The 40-query test split was not evaluated.

| Configuration | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25, title | 13.6% | 6.8% | 7.8% | 5.7% |
| BM25, Bible-grounded | 10.6% | 5.3% | 6.1% | 4.4% |
| Dense, title | 10.6% | 6.8% | 5.7% | 5.3% |
| Dense, title + reranker | 12.1% | 7.6% | 6.3% | 5.7% |
| Dense, title + lyric boost | 12.1% | 7.6% | 6.8% | 6.0% |
| Dense, title + reranker + boost | 13.6% | 8.3% | 6.4% | **6.1%** |
| Dense, Bible-grounded | 10.6% | 6.1% | 4.3% | 4.2% |
| Dense, Bible-grounded + reranker | 10.6% | 6.8% | 4.8% | 4.8% |
| Dense, Bible-grounded + boost | 10.6% | 6.1% | 4.3% | 4.1% |
| Dense, Bible-grounded + reranker + boost | 13.6% | 8.3% | 6.0% | 5.8% |

## Interpretation

Deterministic grounding parsed explicit references first, then used BM25 over
overlapping five-verse windows from the public-domain World English Bible.
It grounded 13 queries: 1 explicit reference, 10 quotations, and 2 narratives;
the remaining 53 titles were classified as abstract and left unchanged.

Naively appending the inferred passage did not improve the aggregate result.
It reduced BM25 nDCG@5 from 5.7% to 4.4%, plain dense nDCG@5 from 5.3% to
4.2%, and full-system nDCG@5 from 6.1% to 5.8%. The grounding audit shows why:
some abstract titles match the wrong passage, and even a correct full chapter
can dilute the short query. This is retained as a documented negative result.

The isolated title-only ablations show that the lyric boost contributes most
of the gain over the plain dense retriever. Adding the cross-encoder to the
boosted system increases nDCG@5 only from 6.0% to 6.1% and lowers MRR@5 from
6.8% to 6.4%.

These labels measure whether retrieval reproduces two historical human
selections. They do not establish that all other hymns are irrelevant, and a
service hymn may reflect liturgical context or personal preference that is not
expressed in the sermon title.

## Frozen candidate

The candidate frozen for held-out evaluation is `dense_title_full`: title-only
dense retrieval with lyric boost and cross-encoder reranking. Deterministic
Bible grounding remains an opt-in experimental feature and is disabled in the
frozen candidate. Grounding parameters and per-category output are recorded in
`service-development-bible.json`; the readable audit is in
`service-development-bible.md`.

The 40-query test split remains untouched. Run it once only after explicitly
confirming this freeze.
