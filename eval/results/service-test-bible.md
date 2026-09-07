# Retrieval test results

`k=5` · World English Bible (Public Domain) · curated and LLM expansion disabled

Command:

```bash
python -m choirnetwork eval --top-k 5 --split test --confirm-held-out
```

Run once on September 7, 2026 using the 40 held-out queries and frozen commit
`1ca1aff`. `dense_title_full` was the candidate selected on development data;
the other rows preserve the predeclared ablation matrix and were not used to
reselect the system after this run.

## Aggregate

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 40 | 10.0% | 6.2% | 6.1% | 5.1% |
| bm25_bible | 40 | 10.0% | 6.2% | 6.1% | 5.1% |
| dense_title | 40 | 17.5% | 8.8% | 8.6% | 6.6% |
| dense_title_rerank | 40 | 17.5% | 8.8% | 8.6% | 6.6% |
| dense_title_boost | 40 | 15.0% | 7.5% | 8.5% | 6.2% |
| dense_title_full | 40 | 15.0% | 8.8% | 8.5% | 7.2% |
| dense_bible | 40 | 15.0% | 7.5% | 8.8% | 6.4% |
| dense_bible_rerank | 40 | 15.0% | 7.5% | 8.4% | 6.2% |
| dense_bible_boost | 40 | 15.0% | 7.5% | 10.5% | 7.1% |
| dense_bible_full | 40 | 15.0% | 8.8% | 8.8% | 7.3% |

## Abstract

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 33 | 12.1% | 7.6% | 7.4% | 6.2% |
| bm25_bible | 33 | 12.1% | 7.6% | 7.4% | 6.2% |
| dense_title | 33 | 15.2% | 7.6% | 9.2% | 6.5% |
| dense_title_rerank | 33 | 15.2% | 7.6% | 9.2% | 6.5% |
| dense_title_boost | 33 | 15.2% | 7.6% | 9.7% | 6.8% |
| dense_title_full | 33 | 15.2% | 9.1% | 9.7% | 8.0% |
| dense_bible | 33 | 15.2% | 7.6% | 9.2% | 6.5% |
| dense_bible_rerank | 33 | 15.2% | 7.6% | 9.2% | 6.5% |
| dense_bible_boost | 33 | 15.2% | 7.6% | 9.7% | 6.8% |
| dense_bible_full | 33 | 15.2% | 9.1% | 9.7% | 8.0% |

## Narrative

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| bm25_bible | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title_rerank | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title_boost | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title_full | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible_rerank | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible_boost | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible_full | 3 | 0.0% | 0.0% | 0.0% | 0.0% |

## Quotation

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 4 | 0.0% | 0.0% | 0.0% | 0.0% |
| bm25_bible | 4 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title | 4 | 50.0% | 25.0% | 10.0% | 11.9% |
| dense_title_rerank | 4 | 50.0% | 25.0% | 10.0% | 11.9% |
| dense_title_boost | 4 | 25.0% | 12.5% | 5.0% | 5.9% |
| dense_title_full | 4 | 25.0% | 12.5% | 5.0% | 5.9% |
| dense_bible | 4 | 25.0% | 12.5% | 12.5% | 9.7% |
| dense_bible_rerank | 4 | 25.0% | 12.5% | 8.3% | 7.7% |
| dense_bible_boost | 4 | 25.0% | 12.5% | 25.0% | 15.3% |
| dense_bible_full | 4 | 25.0% | 12.5% | 8.3% | 7.7% |

## Grounding audit

| Query | Type | Reference | Confidence |
|---|---|---|---:|
| Who is your Friend? | abstract | — | 0.00 |
| Jesus The Sweetest Name | abstract | — | 0.00 |
| David - A Man After God's Heart | abstract | — | 0.00 |
| Death of Asahel | abstract | — | 0.00 |
| Going Back Home | abstract | — | 0.00 |
| Why are you busy? | abstract | — | 0.00 |
| For You Have Been Called to Liberty | abstract | — | 0.00 |
| Watchmen in the House of God | abstract | — | 0.00 |
| Called to be Set Apart | abstract | — | 0.00 |
| From the Valley to Victory - The Two Paths of the Spiritual Life | abstract | — | 0.00 |
| I Dare Not Reach Out My Hand Against the LORD's Anointed | abstract | — | 0.00 |
| Sword and Ephod | abstract | — | 0.00 |
| Jonathan: The Prince of God's Kingdom and a Mighty Warrior | abstract | — | 0.00 |
| It May Be that the Lord Will Work For Us | abstract | — | 0.00 |
| Life is in the Blood | abstract | — | 0.00 |
| The Champion of Israel | abstract | — | 0.00 |
| Young David | abstract | — | 0.00 |
| The King Rejected By God | abstract | — | 0.00 |
| On Whom is the Desire of All? | abstract | — | 0.00 |
| From Ichabod to Ebenezer | abstract | — | 0.00 |
| Israel Demands a King | narrative | Daniel 2:25-29 | 0.70 |
| Create in me a clean heart, O God | quotation | Psalms 51:7-11 | 1.00 |
| I planted, Apollos watered, but God was causing the growth | quotation | 1 Corinthians 3:4-8 | 0.57 |
| The Two Debtors | abstract | — | 0.00 |
| Naomi Return with Ruth | abstract | — | 0.00 |
| They Become Perfectly One | narrative | Jeremiah 23:19-23 | 0.55 |
| Coming Home | abstract | — | 0.00 |
| Secrets of Renewed Strength | abstract | — | 0.00 |
| The Fragrance of Christ | abstract | — | 0.00 |
| What do these stones mean to you? | abstract | — | 0.00 |
| Christ is the Firstfruits of Those Who have Fallen Asleep | quotation | 1 Corinthians 15:16-20 | 0.82 |
| Family Education in the Book of Judges | abstract | — | 0.00 |
| I Am The Bread of Life | abstract | — | 0.00 |
| What Are You Building? | abstract | — | 0.00 |
| Servitude of Moses | abstract | — | 0.00 |
| Do Not Forget | abstract | — | 0.00 |
| Be United in the Same Mind and the Same Judgement | abstract | — | 0.00 |
| Two Questions That Paul asked the Lord | abstract | — | 0.00 |
| The Glory of God Fills the Temple | quotation | Revelation 15:4-8 | 0.89 |
| The Jawbone of a Donkey of Samson | narrative | Judges 15:13-17 | 0.93 |
