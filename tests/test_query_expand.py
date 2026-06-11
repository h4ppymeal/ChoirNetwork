"""Tests for query expansion."""

from choirnetwork.query_expand import (
    _sanitize_themes,
    expand_query,
)


def test_curated_expansion_for_bible_narrative():
    expanded, source = expand_query("Ruth and Naomi")
    assert source == "curated"
    assert "Ruth and Naomi" in expanded
    assert "loyalty" in expanded
    assert "providence" in expanded


def test_curated_expansion_for_holy_communion():
    expanded, source = expand_query("holy communion")
    assert source == "curated"
    assert "bread" in expanded
    assert "remembrance" in expanded


def test_curated_expansion_for_water_into_wine():
    expanded, source = expand_query("Jesus turns water into wine")
    assert source == "curated"
    assert "miracle" in expanded
    assert "Cana" in expanded or "cana" in expanded


def test_unknown_query_returns_original():
    expanded, source = expand_query("xyz unknown topic")
    assert expanded == "xyz unknown topic"
    assert source is None


def test_sanitize_themes_strips_punctuation_and_newlines():
    assert _sanitize_themes("Miracle, wedding feast.") == "miracle wedding feast"
    assert _sanitize_themes("themes: miracle wine\nextra line") == "themes miracle wine"


def test_curated_plus_llm_validates_instead_of_blind_concat(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_validate(query, curated, *, api_key, model):
        assert query == "Jesus turns water into wine"
        assert "miracle" in curated
        return "miracle wedding wine feast sacrament table"

    monkeypatch.setattr(
        "choirnetwork.query_expand._llm_validate_curated",
        fake_validate,
    )

    expanded, source = expand_query("Jesus turns water into wine", use_llm=True)
    assert source == "curated+llm"
    assert "miracle wedding wine feast sacrament table" in expanded
    assert "obedience" not in expanded


def test_curated_falls_back_when_llm_validation_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "choirnetwork.query_expand._llm_validate_curated",
        lambda *args, **kwargs: None,
    )

    expanded, source = expand_query("holy communion", use_llm=True)
    assert source == "curated"
    assert "bread" in expanded


def test_unknown_query_uses_llm_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "choirnetwork.query_expand._llm_expansion",
        lambda query, *, api_key, model: "custom theme words",
    )

    expanded, source = expand_query("obscure minor prophet story", use_llm=True)
    assert source == "llm"
    assert "custom theme words" in expanded
