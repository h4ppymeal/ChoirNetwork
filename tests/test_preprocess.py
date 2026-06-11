"""Tests for hymn text preprocessing and stanza splitting."""

from choirnetwork.preprocess import build_hymn_chunks, split_lyrics_into_stanzas


def test_split_lyrics_groups_four_line_stanzas():
    lyrics = "\n".join(f"line {index}" for index in range(1, 9))
    stanzas = split_lyrics_into_stanzas(lyrics)
    assert len(stanzas) == 2
    assert stanzas[0].startswith("line 1")
    assert "line 4" in stanzas[0]
    assert "line 5" in stanzas[1]


def test_split_lyrics_respects_blank_lines():
    lyrics = "verse one line\n\nverse two line"
    stanzas = split_lyrics_into_stanzas(lyrics)
    assert stanzas == ["verse one line", "verse two line"]


def test_build_hymn_chunks_includes_weighted_title():
    chunks = build_hymn_chunks("Holy Communion", "Bread and wine\nRemember me")
    assert chunks[0].kind == "title"
    assert chunks[0].weight == 2.5
    assert chunks[0].snippet == "Holy Communion"
    assert any(chunk.kind == "stanza" for chunk in chunks)
