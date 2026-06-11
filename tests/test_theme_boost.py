"""Tests for lyric theme keyword boosting."""

from choirnetwork.theme_boost import extract_theme_keywords, lyric_keyword_boost


def test_extract_theme_keywords_drops_jesus_and_stop_words():
    keywords = extract_theme_keywords(
        "Jesus turns water into wine miracle wedding Cana feast"
    )
    assert "jesus" not in keywords
    assert "water" in keywords
    assert "wine" in keywords
    assert "miracle" in keywords
    assert "feast" in keywords


def test_lyric_keyword_boost_counts_lyric_hits():
    lyrics = "at the wedding feast we share the wine a miracle of grace"
    keywords = extract_theme_keywords("water into wine wedding feast miracle")
    boost = lyric_keyword_boost(lyrics, keywords)
    assert boost > 0.15


def test_lyric_keyword_boost_zero_for_title_only_jesus_match():
    lyrics = "my saviour dear i love thee all day long"
    keywords = extract_theme_keywords("Jesus turns water into wine miracle feast")
    assert lyric_keyword_boost(lyrics, keywords) == 0.0
