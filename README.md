# ChoirNetwork

**Dense retrieval system for thematic hymn recommendation** over the [True Jesus Church hymnal](https://hymnal.tjc.org/hymnal-library). Given a sermon title or Bible narrative (e.g. *Ruth and Naomi*), returns the most thematically similar hymns from the complete 536-entry English hymnbook.

Built as an end-to-end NLP pipeline: unstructured web ingest → text preprocessing → embedding index → two-stage retrieval → offline evaluation.

## ML problem

| | |
|---|---|
| **Task** | Semantic retrieval: map a short query (2–10 words) to relevant documents (hymn title + multi-stanza lyrics) |
| **Challenge** | Short-query vs long-document asymmetry; word-overlap false positives (e.g. *holy communion* → *Holy, Holy, Holy*) |
| **Approach** | Verse-level chunking + weighted max-pooling → cross-encoder reranking → hybrid query expansion |
| **Evaluation** | 106 manually observed service pairs with development/test splits and IR ranking metrics |

## Architecture

```
Query → [optional topic expansion] → bi-encoder recall (top 50 chunks)
      → cross-encoder rerank → ranked hymns
```

1. **Ingest** — Combine 497 public web transcriptions with 39 local PDF extractions
2. **Preprocess** — Contraction expansion, stop-word removal, stanza splitting (title 2.5× weight + 4-line chunks)
3. **Index** — Embed chunks with `all-MiniLM-L6-v2` (384-dim, cosine similarity)
4. **Retrieve** — Weighted max-pool over chunks → recall top 50 → rerank with `ms-marco-MiniLM-L-6-v2`
5. **Expand** — Curated sermon/Bible topic map (27+ themes) + optional LLM validation/refinement (`gpt-4o-mini`)
6. **Evaluate** — Compare BM25 and four retrieval ablations on 66 development queries while preserving 40 held-out queries

See [`docs/ENGINE.md`](docs/ENGINE.md) for pipeline diagrams and implementation details.

## Tech stack

| Layer | Tools |
|-------|-------|
| **ML / NLP** | PyTorch, sentence-transformers, Hugging Face Transformers (MiniLM bi-encoder, MS MARCO cross-encoder) |
| **Retrieval** | NumPy (cosine similarity, weighted max-pooling), chunked dense index |
| **Text processing** | contractions, stop-words, custom stanza splitter |
| **Query understanding** | Curated topic ontology + OpenAI API (`gpt-4o-mini`, optional; validates curated themes per story) |
| **Evaluation** | BM25 baseline, custom IR metrics (Hit@k, Recall@k, MRR, nDCG@k), pytest |
| **Data pipeline** | cloudscraper, BeautifulSoup, lxml (web scrape → JSON corpus) |
| **Serving** | FastAPI, uvicorn, vanilla JS frontend |
| **Config** | python-dotenv (`.env`) |

## Setup

```bash
conda activate choir_network   # or: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # optional: OPENAI_API_KEY for LLM expansion
```

## Usage

```bash
# Scrape corpus → build index → serve
python -m choirnetwork scrape
python -m choirnetwork build
python -m choirnetwork serve          # http://127.0.0.1:8000 (UI: “AI theme refinement” toggle per search)

# Search
python -m choirnetwork search "Holy, Holy, Holy" --top-k 10
python -m choirnetwork search "Ruth and Naomi" --expand --top-k 10
python -m choirnetwork search "obscure parable" --llm-expand --top-k 10

# Offline evaluation (run after build)
python -m choirnetwork eval --top-k 5

# Tests
pytest tests/
```

### Environment variables (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Optional LLM query expansion |
| `CHOIRNETWORK_EXPAND_QUERIES` | `1` | Curated topic expansion in web app |
| `CHOIRNETWORK_LLM_EXPAND` | `0` | Default for UI “AI theme refinement” checkbox; validates curated themes or expands unknown topics |
| `CHOIRNETWORK_USE_RERANKER` | `1` | Cross-encoder reranking |

## Corpus provenance

The local `data/raw/hymns.json` contains all 536 numbered and lettered entries
in the English hymnbook PDF. Titles and the `bible_verse` /
`bible_reference` fields use the PDF as their source of truth.

Lyrics for 497 entries come from hymnal.tjc.org. The remaining 39 were
reconstructed from positioned syllable text in the sheet-music PDF and are
marked with `lyrics_source = "Hymn_English.pdf positioned text layer
(unreviewed)"`. Those transcriptions require human proofreading before they
should be published or treated as character-perfect.

Both `data/` and the source PDF at `data/reference/Hymn_English.pdf` are
intentionally excluded from Git because
the later hymns carry explicit copyright notices.

## Evaluation

[`eval/datasets/service_hymns.csv`](eval/datasets/service_hymns.csv) records
106 sermon titles and the two hymns manually observed in each service. The
dataset contains 66 development queries and 40 held-out test queries.

```bash
python -m choirnetwork eval --top-k 5
```

Compares five retrieval configs:

| Config | Description |
|--------|-------------|
| `bm25` | Classical lexical baseline over hymn titles and lyrics |
| `bi_encoder_only` | Chunked index and cosine recall |
| `chunked_rerank` | + cross-encoder reranking on top 50 |
| `chunked_rerank_expand` | + curated query expansion |
| `full_system` | + lyric keyword boost |

Development results from September 7, 2026 (`k=5`, 66 queries,
LLM expansion disabled):

| Config | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|--------|------:|---------:|------:|-------:|
| BM25 | 13.6% | 6.8% | 7.8% | 5.7% |
| Bi-encoder only | 10.6% | 6.8% | 5.7% | 5.3% |
| + cross-encoder | 12.1% | 7.6% | 6.3% | 5.7% |
| + curated expansion | 12.1% | 7.6% | 6.3% | 5.7% |
| + lyric keyword boost | 13.6% | 8.3% | 6.4% | 6.1% |

The full system only slightly exceeds BM25 on this independent development
set. These labels measure recovery of two historical human selections, which
may reflect liturgical context and personal preference beyond sermon-title
relevance. The 40-query test split has not been run. See
[`eval/results/service-development.md`](eval/results/service-development.md)
for interpretation and [`eval/README.md`](eval/README.md) for the protocol.

## Project structure

```
choirnetwork/
  engine.py          # Two-stage retrieval, chunked index, lyric theme boost
  preprocess.py      # NLP preprocessing, stanza splitting
  query_expand.py    # Curated + LLM-validated topic expansion
  theme_boost.py     # Lyric keyword boost for sermon/Bible themes
  bm25.py            # Dependency-free lexical baseline
  eval.py            # IR metrics and config comparison
  scraper.py         # Web corpus ingest
  api.py / cli.py    # FastAPI + CLI (UI has per-search AI refinement toggle)
  static/            # Web frontend
docs/
  ENGINE.md          # Retrieval architecture and implementation details
eval/
  datasets/
    service_hymns.csv       # 66 development + 40 held-out observed-service labels
  results/           # Versioned benchmark outputs and interpretation
tests/               # pytest (preprocess, eval metrics, retrieval, expansion)
```

## Resume bullets (template)

Fill in metrics after running `python -m choirnetwork eval`:

```text
• Built a two-stage dense retrieval pipeline over 500+ hymn lyrics: verse-level
  chunking with title weighting, bi-encoder recall (MiniLM), and cross-encoder reranking
  (MS MARCO), deployed via FastAPI

• Diagnosed retrieval failure modes on short sermon queries (word-overlap false positives);
  implemented hybrid query expansion (curated sermon-topic ontology + optional LLM) to map
  Bible narratives (e.g. Ruth and Naomi) to searchable themes

• Built a reproducible benchmark from 106 manually observed sermon–hymn pairs,
  with component ablations, 66 development queries, and 40 held-out test queries
```

## Contributors

Bethany Liu, Mark Chen
