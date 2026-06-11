"""Boost retrieval when hymn lyrics match sermon / Bible theme keywords."""

from __future__ import annotations

import re

from choirnetwork.preprocess import preprocess_text

# Common hymn vocabulary that dominates titles but rarely distinguishes themes.
THEME_NOISE_WORDS = frozenset(
    {
        "jesus",
        "god",
        "lord",
        "christ",
        "holy",
        "thee",
        "thy",
        "thine",
        "hath",
        "hast",
        "doth",
        "dost",
        "art",
        "shalt",
        "wilt",
        "ye",
        "lo",
        "o",
        "oh",
    }
)

LYRIC_KEYWORD_WEIGHT = 0.08
MAX_LYRIC_BOOST = 0.35


def extract_theme_keywords(text: str) -> frozenset[str]:
    """Theme words from a query, excluding stop words and generic hymn vocabulary."""
    processed = preprocess_text(text, remove_stop_words=True)
    return frozenset(
        word
        for word in processed.split()
        if len(word) >= 3 and word not in THEME_NOISE_WORDS
    )


def lyric_keyword_boost(lyrics: str, keywords: frozenset[str]) -> float:
    """Return a 0–1 boost when preprocessed lyrics contain theme keywords."""
    if not lyrics or not keywords:
        return 0.0

    text = lyrics.lower()
    hits = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text))
    return min(MAX_LYRIC_BOOST, hits * LYRIC_KEYWORD_WEIGHT)
