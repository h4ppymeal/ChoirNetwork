# ChoirNetwork

ChoirNetwork recommends hymns from the 536-entry True Jesus Church English
hymnal for a sermon title or Bible narrative. The same search index powers a
FastAPI web interface and a command-line tool.

## Retrieval pipeline

```
Query → [optional deterministic Bible grounding] → bi-encoder recall (top 50 chunks)
      → cross-encoder rerank → ranked hymns
```

Hymns are split into title and stanza chunks. Titles receive a 2.5× weight.
`all-MiniLM-L6-v2` retrieves 50 candidates, and
`ms-marco-MiniLM-L-6-v2` optionally reranks them. A small lexical boost can be
applied when query terms occur in the lyrics.

Bible grounding is optional. It parses explicit references or searches
five-verse windows from the World English Bible with BM25. Grounding depends
only on the sermon title and Bible text.

See [`docs/ENGINE.md`](docs/ENGINE.md) for pipeline diagrams and implementation details.

## Setup

```bash
conda activate choir_network
pip install -r requirements.txt
```

## Usage

```bash
python -m choirnetwork scrape
python -m choirnetwork build
python -m choirnetwork serve

python -m choirnetwork search "Holy, Holy, Holy" --top-k 10
python -m choirnetwork search "The crises of rebuilding (Ezra 4)" --bible-ground

pytest tests/
```

The web app applies curated topic expansion by default. In the CLI it is
enabled with `--expand`; `--llm-expand` adds optional OpenAI-backed refinement
when an API key is configured.

## Corpus provenance

The local `data/raw/hymns.json` contains all 536 numbered and lettered entries
in the English hymnbook PDF. Titles and the `bible_verse` /
`bible_reference` fields use the PDF as their source of truth.

Lyrics for 497 entries come from hymnal.tjc.org. The remaining 39 were
reconstructed from positioned syllable text in the sheet-music PDF and are
marked with `lyrics_source = "Hymn_English.pdf positioned text layer
(unreviewed)"`. Those transcriptions require human proofreading before they
should be published or treated as character-perfect.

The copyrighted hymn corpus and source PDF remain excluded from Git. The
public-domain World English Bible verse corpus under `data/public/` is included
so deterministic grounding can be reproduced.

## Project structure

```
choirnetwork/
  bible_grounding.py # Reference parsing, passage BM25, confidence abstention
  engine.py          # Two-stage retrieval, chunked index, lyric theme boost
  preprocess.py      # NLP preprocessing, stanza splitting
  query_expand.py    # Curated + LLM-validated topic expansion
  theme_boost.py     # Lyric keyword boost for sermon/Bible themes
  bm25.py            # Lexical search baseline
  lexical.py         # Shared BM25 scoring
  scraper.py         # Web corpus ingest
  api.py / cli.py    # FastAPI and command-line interfaces
  static/            # Web frontend
docs/
  ENGINE.md          # Retrieval architecture and implementation details
tests/               # pytest suite
```

## Contributors

Bethany Liu, Mark Chen
