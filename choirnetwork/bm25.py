"""Small, dependency-free BM25 baseline for hymn retrieval."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from choirnetwork.engine import HymnIndex, SimilarHymn
from choirnetwork.preprocess import preprocess_text

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


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
        self._term_frequencies = [Counter(document) for document in documents]
        self._document_lengths = np.asarray(
            [len(document) for document in documents],
            dtype=np.float32,
        )
        self._average_document_length = float(self._document_lengths.mean())

        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequencies.update(frequencies.keys())

        document_count = len(documents)
        self._inverse_document_frequencies = {
            term: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequencies.items()
        }

    def search(self, query: str, *, top_k: int = 10) -> list[SimilarHymn]:
        query_terms = _tokenize(query)
        scores = np.zeros(len(self.index.slugs), dtype=np.float32)
        if not query_terms or self._average_document_length == 0:
            return []

        length_normalization = 1.0 - self.b + (
            self.b * self._document_lengths / self._average_document_length
        )
        for term in query_terms:
            inverse_document_frequency = self._inverse_document_frequencies.get(term)
            if inverse_document_frequency is None:
                continue

            term_frequency = np.asarray(
                [frequencies.get(term, 0) for frequencies in self._term_frequencies],
                dtype=np.float32,
            )
            denominator = term_frequency + self.k1 * length_normalization
            scores += inverse_document_frequency * (
                term_frequency * (self.k1 + 1.0) / denominator
            )

        ranked_indices = np.argsort(-scores, kind="stable")
        limit = min(top_k, len(ranked_indices))
        return [
            self.index.hymn_at(int(index), float(scores[index]))
            for index in ranked_indices[:limit]
        ]
