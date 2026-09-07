"""Lexical BM25 baseline over hymn titles and lyrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from choirnetwork.engine import HymnIndex, SimilarHymn
from choirnetwork.lexical import BM25Index, DEFAULT_B, DEFAULT_K1
from choirnetwork.preprocess import preprocess_text


def _tokenize(text: str) -> list[str]:
    return preprocess_text(text, remove_stop_words=False).split()


@dataclass
class BM25Retriever:
    """Rank hymns with Okapi BM25 over title and lyric text."""

    index: HymnIndex
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    def __post_init__(self) -> None:
        lyrics = self.index.lyrics_preprocessed or [""] * len(self.index.slugs)
        if len(lyrics) != len(self.index.titles):
            raise ValueError("BM25 requires one lyric document per hymn title")
        documents = [
            _tokenize(f"{title} {lyric}")
            for title, lyric in zip(self.index.titles, lyrics)
        ]
        self._bm25 = BM25Index(documents, k1=self.k1, b=self.b)

    def search(self, query: str, *, top_k: int = 10) -> list[SimilarHymn]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        scores = self._bm25.scores(query_terms)
        ranked_indices = np.argsort(-scores, kind="stable")
        limit = min(top_k, len(ranked_indices))
        return [
            self.index.hymn_at(int(index), float(scores[index]))
            for index in ranked_indices[:limit]
        ]
