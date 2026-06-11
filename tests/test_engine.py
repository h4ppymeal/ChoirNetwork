"""Tests for chunked indexing and retrieval."""

from choirnetwork.engine import (
    HymnSimilarityEngine,
    _filter_displayable,
    _display_percent,
    _normalize_bi_encoder_score,
    load_engine,
    save_engine,
)
from choirnetwork.preprocess import DEFAULT_TITLE_WEIGHT
from choirnetwork.scraper import HymnRecord


def test_display_percent_matches_frontend_rounding():
    assert _display_percent(0.0) == 0
    assert _display_percent(0.004) == 0
    assert _display_percent(0.005) == 1
    assert _display_percent(0.42) == 42


def test_normalize_bi_encoder_score_uses_title_weight():
    assert _normalize_bi_encoder_score(2.5, DEFAULT_TITLE_WEIGHT) == 1.0
    assert _normalize_bi_encoder_score(1.25, DEFAULT_TITLE_WEIGHT) == 0.5
    assert _normalize_bi_encoder_score(0.5, DEFAULT_TITLE_WEIGHT) == 0.2


def test_filter_displayable_drops_zero_percent_matches():
    from choirnetwork.engine import SimilarHymn

    matches = [
        SimilarHymn(1, "", "1", "Good", 0.42),
        SimilarHymn(2, "", "2", "Weak", 0.004),
        SimilarHymn(3, "", "3", "Zero", 0.0),
    ]
    filtered = _filter_displayable(matches)
    assert [match.slug for match in filtered] == ["1"]


def _sample_hymns() -> list[HymnRecord]:
    return [
        HymnRecord(
            number=11,
            variant="",
            title="Holy, Holy, Holy",
            lyrics="\n".join(
                [
                    "Holy, holy, holy! Lord God Almighty!",
                    "Early in the morning our song shall rise to Thee;",
                    "Holy, holy, holy! Merciful and mighty!",
                    "God in three Persons, blessed Trinity!",
                    "Holy, holy, holy! All the saints adore Thee,",
                    "Casting down their golden crowns around the glassy sea;",
                    "Cherubim and seraphim falling down before Thee,",
                    "Which wert, and art, and evermore shalt be.",
                ]
            ),
            url="https://example.com/11",
            slug="11",
        ),
        HymnRecord(
            number=210,
            variant="",
            title="At the Table of the Lord",
            lyrics="\n".join(
                [
                    "At the table of the Lord we gather in His name,",
                    "Holy communion with the bread and cup we share;",
                    "Remembering His body broken and His blood outpoured,",
                    "We proclaim the Lord's death until He comes again.",
                    "Eat this bread and drink this cup in remembrance of Me,",
                    "Until that day when we shall feast with Him above;",
                    "One body in the Lord, united by this sacred meal,",
                    "Holy communion, fellowship of grace and love.",
                ]
            ),
            url="https://example.com/210",
            slug="210",
        ),
        HymnRecord(
            number=127,
            variant="",
            title="When I Survey the Wondrous Cross",
            lyrics="\n".join(
                [
                    "When I survey the wondrous cross",
                    "On which the Prince of glory died,",
                    "My richest gain I count but loss,",
                    "And pour contempt on all my pride.",
                    "See from His head, His hands, His feet,",
                    "Sorrow and love flow mingled down;",
                    "Did e'er such love and sorrow meet,",
                    "Or thorns compose so rich a crown?",
                ]
            ),
            url="https://example.com/127",
            slug="127",
        ),
    ]


def _lyric_boost_hymns() -> list[HymnRecord]:
    return _sample_hymns() + [
        HymnRecord(
            number=295,
            variant="",
            title="My Jesus, I Love Thee",
            lyrics="\n".join(
                [
                    "My Jesus, I love Thee, I know Thou art mine;",
                    "For Thee all the follies of sin I resign;",
                    "My gracious Redeemer, my Saviour art Thou;",
                    "If ever I loved Thee, my Jesus, 'tis now.",
                ]
            ),
            url="https://example.com/295",
            slug="295",
        ),
        HymnRecord(
            number=400,
            variant="",
            title="Wedding Feast Hymn",
            lyrics="\n".join(
                [
                    "At the wedding feast the water turned to wine,",
                    "A miracle of glory, first sign divine;",
                    "Believe and celebrate the feast of joy,",
                    "The bridegroom's table set for every guest.",
                ]
            ),
            url="https://example.com/400",
            slug="400",
        ),
    ]


def test_chunked_search_prefers_thematic_match_over_word_overlap(tmp_path):
    engine = HymnSimilarityEngine.from_raw_hymns(_sample_hymns())
    save_engine(engine, tmp_path)
    loaded = load_engine(tmp_path)

    with_rerank = loaded.search("holy communion", top_k=2)
    without_rerank = load_engine(tmp_path, use_reranker=False).search(
        "holy communion",
        top_k=2,
    )

    assert with_rerank[0].slug == "210"
    assert without_rerank[0].slug == "11"
    assert all(_display_percent(match.score) > 0 for match in with_rerank)
    if len(with_rerank) > 1:
        assert with_rerank[0].score > with_rerank[1].score


def test_lyric_boost_ranks_feast_hymn_above_jesus_title(tmp_path):
    engine = HymnSimilarityEngine.from_raw_hymns(_lyric_boost_hymns())
    save_engine(engine, tmp_path)
    loaded = load_engine(tmp_path, use_reranker=False)

    results = loaded.search(
        "Jesus turns water into wine",
        top_k=3,
        expand_query_flag=True,
    )

    assert results[0].slug == "400"
    assert results[0].score <= 1.0
    assert results[0].score > results[1].score
