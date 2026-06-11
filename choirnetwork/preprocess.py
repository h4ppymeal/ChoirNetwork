"""Text preprocessing for hymn lyrics and titles."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import contractions
from stop_words import get_stop_words

ENGLISH_STOP_WORDS = set(get_stop_words("english"))
PUNCTUATION_TABLE = str.maketrans(
    "",
    "",
    string.punctuation.replace("-", "").replace("'", ""),
)
STANZA_MARKER_RE = re.compile(r"^(chorus|refrain|bridge)\s*:\s*", re.I)
NUMBERED_STANZA_RE = re.compile(r"^\d+\.\s")
DEFAULT_STANZA_SIZE = 4
DEFAULT_TITLE_WEIGHT = 2.5


@dataclass(frozen=True)
class HymnChunk:
    """One embeddable unit for a hymn (title or stanza)."""

    text: str
    snippet: str
    weight: float
    kind: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str, *, remove_stop_words: bool = True) -> str:
    text = contractions.fix(text.lower().translate(PUNCTUATION_TABLE))
    text = text.replace("--", " ").replace("-", " ")
    text = normalize_whitespace(text)

    if remove_stop_words:
        words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
        text = " ".join(words)

    return text


def preprocess_hymn(title: str, lyrics: str, *, remove_stop_words: bool = True) -> str:
    return preprocess_text(f"{title}\n{lyrics}".strip(), remove_stop_words=remove_stop_words)


def split_lyrics_into_stanzas(lyrics: str, *, stanza_size: int = DEFAULT_STANZA_SIZE) -> list[str]:
    """Split hymn lyrics into stanza-sized chunks for embedding."""
    lyrics = lyrics.strip()
    if not lyrics:
        return []

    blocks = [block.strip() for block in re.split(r"\n\s*\n", lyrics) if block.strip()]
    if len(blocks) > 1:
        return blocks

    lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    if not lines:
        return []

    if any(NUMBERED_STANZA_RE.match(line) for line in lines):
        stanzas: list[str] = []
        current: list[str] = []
        for line in lines:
            if NUMBERED_STANZA_RE.match(line) and current:
                stanzas.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            stanzas.append("\n".join(current))
        return stanzas

    if any(STANZA_MARKER_RE.match(line) for line in lines):
        stanzas = []
        current: list[str] = []
        for line in lines:
            if STANZA_MARKER_RE.match(line) and current:
                stanzas.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            stanzas.append("\n".join(current))
        return stanzas

    return [
        "\n".join(lines[index : index + stanza_size])
        for index in range(0, len(lines), stanza_size)
    ]


def build_hymn_chunks(
    title: str,
    lyrics: str,
    *,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    remove_stop_words: bool = True,
) -> list[HymnChunk]:
    """Build weighted title + stanza chunks for indexing and reranking."""
    chunks: list[HymnChunk] = []
    title = title.strip()
    if title:
        chunks.append(
            HymnChunk(
                text=preprocess_text(title, remove_stop_words=remove_stop_words),
                snippet=title,
                weight=title_weight,
                kind="title",
            )
        )

    for stanza in split_lyrics_into_stanzas(lyrics):
        stanza = stanza.strip()
        if not stanza:
            continue
        chunks.append(
            HymnChunk(
                text=preprocess_text(stanza, remove_stop_words=remove_stop_words),
                snippet=stanza,
                weight=1.0,
                kind="stanza",
            )
        )

    return chunks
