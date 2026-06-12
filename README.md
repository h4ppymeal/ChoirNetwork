# ChoirNetwork

**Dense retrieval system for thematic hymn recommendation** over the [True Jesus Church hymnal](https://hymnal.tjc.org/hymnal-library). Given a sermon title or Bible narrative (e.g. *Ruth and Naomi*), returns the most thematically similar hymns from a corpus of ~525 scraped lyrics.

Built as an end-to-end NLP pipeline: unstructured web ingest → text preprocessing → embedding index → two-stage retrieval → offline evaluation.

## ML problem

| | |
|---|---|
| **Task** | Semantic retrieval: map a short query (2–10 words) to relevant documents (hymn title + multi-stanza lyrics) |
| **Challenge** | Short-query vs long-document asymmetry; word-overlap false positives (e.g. *holy communion* → *Holy, Holy, Holy*) |
| **Approach** | Verse-level chunking + weighted max-pooling → cross-encoder reranking → hybrid query expansion |
| **Evaluation** | Labeled eval set (`eval/queries.jsonl`) with Hit@k, Recall@k, MRR@k, nDCG@k |

## Architecture

```
Query → [optional topic expansion] → bi-encoder recall (top 50 chunks)
      → cross-encoder rerank → ranked hymns
```

1. **Ingest** — Scrape hymnal.tjc.org (~497 of 539 catalog entries with lyrics)
2. **Preprocess** — Contraction expansion, stop-word removal, stanza splitting (title 2.5× weight + 4-line chunks)
3. **Index** — Embed chunks with `all-MiniLM-L6-v2` (384-dim, cosine similarity)
4. **Retrieve** — Weighted max-pool over chunks → recall top 50 → rerank with `ms-marco-MiniLM-L-6-v2`
5. **Expand** — Curated sermon/Bible topic map (27+ themes) + optional LLM validation/refinement (`gpt-4o-mini`)
6. **Evaluate** — Compare bi-encoder vs rerank vs rerank+expand on 39 labeled queries

See [`choirnetwork/ENGINE.md`](choirnetwork/ENGINE.md) for pipeline diagrams and implementation details.

## Tech stack

| Layer | Tools |
|-------|-------|
| **ML / NLP** | PyTorch, sentence-transformers, Hugging Face Transformers (MiniLM bi-encoder, MS MARCO cross-encoder) |
| **Retrieval** | NumPy (cosine similarity, weighted max-pooling), chunked dense index |
| **Text processing** | contractions, stop-words, custom stanza splitter |
| **Query understanding** | Curated topic ontology + OpenAI API (`gpt-4o-mini`, optional; validates curated themes per story) |
| **Evaluation** | Custom IR metrics (Hit@k, Recall@k, MRR, nDCG@k), pytest |
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

## Evaluation

Labeled queries in [`eval/queries.jsonl`](eval/queries.jsonl) (39 queries: hymn titles, sermon topics, Bible narratives).

```bash
python -m choirnetwork eval --top-k 5
```

Compares three retrieval configs:

| Config | Description |
|--------|-------------|
| `bi_encoder_only` | Chunked index, cosine recall, no reranking |
| `chunked_rerank` | + cross-encoder rerank on top 50 |
| `chunked_rerank_expand` | + curated query expansion |

Reports **Hit@k**, **Recall@k**, **MRR@k**, **nDCG@k**.

## Project structure

```
choirnetwork/
  engine.py          # Two-stage retrieval, chunked index, lyric theme boost
  preprocess.py      # NLP preprocessing, stanza splitting
  query_expand.py    # Curated + LLM-validated topic expansion
  theme_boost.py     # Lyric keyword boost for sermon/Bible themes
  eval.py            # IR metrics and config comparison
  scraper.py         # Web corpus ingest
  api.py / cli.py    # FastAPI + CLI (UI has per-search AI refinement toggle)
  static/            # Web frontend
eval/queries.jsonl   # Labeled eval set
tests/               # pytest (preprocess, eval metrics, retrieval, expansion)
```

## Contributors

Bethany Liu, Mark Chen
