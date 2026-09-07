"""FastAPI server for the ChoirNetwork hymn similarity web app."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from choirnetwork.config import load_env
from choirnetwork.engine import HymnSimilarityEngine, load_engine
from choirnetwork.query_expand import expand_query
from choirnetwork.scraper import HYMN_BASE_URL

load_env()

DEFAULT_INDEX_PATH = Path(os.environ.get("CHOIRNETWORK_INDEX_PATH", "data/index"))
USE_RERANKER = os.environ.get("CHOIRNETWORK_USE_RERANKER", "1") != "0"
USE_QUERY_EXPANSION = os.environ.get("CHOIRNETWORK_EXPAND_QUERIES", "1") != "0"
USE_LLM_EXPANSION = os.environ.get("CHOIRNETWORK_LLM_EXPAND", "0") == "1"
STATIC_DIR = Path(__file__).parent / "static"

_engine: HymnSimilarityEngine | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    mode: Literal["top_k", "threshold"] = "top_k"
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    use_llm_expansion: Optional[bool] = None


class HymnResult(BaseModel):
    label: str
    title: str
    slug: str
    score: float
    url: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: list[HymnResult]
    expansion_source: Optional[str] = None
    expanded_query: Optional[str] = None


def get_engine() -> HymnSimilarityEngine:
    if _engine is None:
        raise RuntimeError("Similarity engine is not loaded")
    return _engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine
    index_path = DEFAULT_INDEX_PATH
    if not index_path.exists():
        raise RuntimeError(
            f"Hymn index not found at {index_path}. Run: python -m choirnetwork build"
        )
    _engine = load_engine(index_path, use_reranker=USE_RERANKER)
    yield
    _engine = None


app = FastAPI(title="ChoirNetwork", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    return {
        "status": "ok",
        "hymns_indexed": len(engine.index.slugs),
        "llm_available": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "default_llm_expand": USE_LLM_EXPANSION,
        "default_query_expand": USE_QUERY_EXPANSION,
    }


@app.post("/api/search", response_model=SearchResponse)
def search_hymns(body: SearchRequest) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    engine = get_engine()
    use_llm = (
        body.use_llm_expansion
        if body.use_llm_expansion is not None
        else USE_LLM_EXPANSION
    )
    expand_query_flag = USE_QUERY_EXPANSION or use_llm

    expanded_query, expansion_source = (
        expand_query(query, use_llm=use_llm)
        if expand_query_flag
        else (query, None)
    )

    matches = engine.search(
        query,
        top_k=0 if body.mode == "threshold" else body.top_k,
        min_score=body.min_score if body.mode == "threshold" else 0.0,
        retrieval_query=expanded_query,
    )

    results = [
        HymnResult(
            label=match.label,
            title=match.title,
            slug=match.slug,
            score=round(match.score, 4),
            url=f"{HYMN_BASE_URL}/{match.slug}",
        )
        for match in matches
    ]

    return SearchResponse(
        query=query,
        mode=body.mode,
        count=len(results),
        results=results,
        expansion_source=expansion_source,
        expanded_query=expanded_query if expansion_source else None,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
