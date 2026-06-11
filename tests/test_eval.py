"""Tests for retrieval evaluation metrics."""

from choirnetwork.eval import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    resolve_relevant_slugs,
)
from choirnetwork.engine import HymnIndex


def _sample_index() -> HymnIndex:
    return HymnIndex(
        model_name="test",
        numbers=[1, 2, 3],
        variants=["", "", ""],
        slugs=["1", "2", "3"],
        titles=["Rock of Ages", "Amazing Grace", "Trust and Obey"],
    )


def test_resolve_relevant_slugs_by_title_fragment():
    index = _sample_index()
    slugs = resolve_relevant_slugs(index, {"relevant_titles": ["Rock", "Trust"]})
    assert slugs == ["1", "3"]


def test_hit_rate_and_mrr():
    relevant = {"2", "3"}
    retrieved = ["1", "3", "2"]
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    assert mrr_at_k(retrieved, relevant, k=3) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert ndcg_at_k(retrieved, relevant, k=3) > 0.0
