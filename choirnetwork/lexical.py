"""Shared lexical-ranking utilities."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass
class BM25Index:
    documents: list[list[str]]
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    document_frequencies: dict[str, int] = field(init=False)
    inverse_document_frequencies: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self._term_frequencies = [Counter(document) for document in self.documents]
        self._document_lengths = np.asarray(
            [len(document) for document in self.documents],
            dtype=np.float32,
        )
        self._average_document_length = (
            float(self._document_lengths.mean()) if self.documents else 0.0
        )

        frequencies: Counter[str] = Counter()
        for terms in self._term_frequencies:
            frequencies.update(terms.keys())
        self.document_frequencies = dict(frequencies)

        count = len(self.documents)
        self.inverse_document_frequencies = {
            term: math.log(
                1.0 + (count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in frequencies.items()
        }

    def scores(self, query_terms: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not query_terms or self._average_document_length == 0:
            return scores

        length_normalization = 1.0 - self.b + (
            self.b * self._document_lengths / self._average_document_length
        )
        for term in query_terms:
            inverse_frequency = self.inverse_document_frequencies.get(term)
            if inverse_frequency is None:
                continue
            term_frequency = np.asarray(
                [terms.get(term, 0) for terms in self._term_frequencies],
                dtype=np.float32,
            )
            denominator = term_frequency + self.k1 * length_normalization
            scores += inverse_frequency * (
                term_frequency * (self.k1 + 1.0) / denominator
            )
        return scores
