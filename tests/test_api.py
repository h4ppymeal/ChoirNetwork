from choirnetwork import api


class _FakeIndex:
    slugs = ["1", "2"]


class _FakeEngine:
    index = _FakeIndex()

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return []


def test_search_expands_once_and_passes_the_result_to_retrieval(monkeypatch):
    engine = _FakeEngine()
    expansion_calls = []

    def expand(query, *, use_llm):
        expansion_calls.append((query, use_llm))
        return f"{query} mercy grace", "curated"

    monkeypatch.setattr(api, "_engine", engine)
    monkeypatch.setattr(api, "expand_query", expand)
    monkeypatch.setattr(api, "USE_QUERY_EXPANSION", True)
    monkeypatch.setattr(api, "USE_LLM_EXPANSION", False)

    response = api.search_hymns(api.SearchRequest(query="Repentance"))

    assert expansion_calls == [("Repentance", False)]
    assert engine.calls == [
        (
            "Repentance",
            {
                "top_k": 10,
                "min_score": 0.0,
                "retrieval_query": "Repentance mercy grace",
            },
        )
    ]
    assert response.expansion_source == "curated"


def test_threshold_mode_uses_zero_top_k(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setattr(api, "_engine", engine)
    monkeypatch.setattr(api, "USE_QUERY_EXPANSION", False)
    monkeypatch.setattr(api, "USE_LLM_EXPANSION", False)

    api.search_hymns(
        api.SearchRequest(
            query="Grace",
            mode="threshold",
            min_score=0.4,
        )
    )

    assert engine.calls[0][1]["top_k"] == 0
    assert engine.calls[0][1]["min_score"] == 0.4
