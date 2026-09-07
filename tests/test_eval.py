"""Tests for retrieval evaluation metrics."""

import json

from choirnetwork.bm25 import BM25Retriever
from choirnetwork.bible_grounding import GroundingResult
from choirnetwork.eval import (
    RETRIEVAL_CONFIGS,
    EvalMetrics,
    EvalQuery,
    EvalRunResult,
    EvaluationReport,
    evaluate_retriever,
    format_report_markdown,
    hit_rate_at_k,
    load_eval_queries,
    mrr_at_k,
    ndcg_at_k,
    prepare_eval_queries,
    recall_at_k,
    resolve_relevant_slugs,
    write_evaluation_report,
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


def test_bm25_ranks_lexical_match_first():
    index = _sample_index()
    index.lyrics_preprocessed = [
        "rock ages cleft me",
        "amazing grace sweet sound",
        "trust obey no other way",
    ]

    matches = BM25Retriever(index).search("sweet grace", top_k=2)

    assert [match.slug for match in matches] == ["2", "1"]


def test_evaluate_retriever_accepts_non_neural_baseline():
    queries = [
        EvalQuery(
            query="grace",
            relevant_slugs=("2",),
            category="hymn_title",
        )
    ]

    metrics = evaluate_retriever(
        queries,
        lambda _query, _top_k: ["2", "1"],
        k=2,
    )

    assert metrics.hit_rate == 1.0
    assert metrics.recall == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg == 1.0


def test_prepare_eval_queries_filters_held_out_split(tmp_path):
    eval_path = tmp_path / "queries.jsonl"
    rows = [
        {
            "query": "grace",
            "relevant_slugs": ["2"],
            "category": "topic",
            "split": "development",
        },
        {
            "query": "obedience",
            "relevant_slugs": ["3"],
            "category": "topic",
            "split": "test",
        },
    ]
    eval_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    queries = prepare_eval_queries(_sample_index(), eval_path, split="test")

    assert [query.query for query in queries] == ["obedience"]


def test_load_eval_queries_reads_observed_service_csv(tmp_path):
    eval_path = tmp_path / "services.csv"
    eval_path.write_text(
        "query,hymn_1_slug,hymn_2_slug,split,category,notes\n"
        "Faith under trial,136,208,test,observed_service,manually verified\n",
        encoding="utf-8",
    )

    queries = load_eval_queries(eval_path)

    assert queries == [
        {
            "query": "Faith under trial",
            "relevant_slugs": ["136", "208"],
            "category": "observed_service",
            "split": "test",
            "notes": "manually verified",
        }
    ]


def test_retrieval_configs_isolate_grounding_reranker_and_boost():
    configs = {config.name: config for config in RETRIEVAL_CONFIGS}

    assert configs["dense_title"].use_bible is False
    assert configs["dense_title_rerank"].use_reranker is True
    assert configs["dense_title_rerank"].use_lyric_boost is False
    assert configs["dense_title_boost"].use_reranker is False
    assert configs["dense_title_boost"].use_lyric_boost is True
    assert configs["dense_title_full"].use_bible is False
    assert configs["dense_title_full"].use_reranker is True
    assert configs["dense_title_full"].use_lyric_boost is True
    assert configs["dense_bible"].use_bible is True
    assert configs["dense_bible"].use_reranker is False
    assert configs["dense_bible"].use_lyric_boost is False
    assert configs["dense_bible_rerank"].use_reranker is True
    assert configs["dense_bible_rerank"].use_lyric_boost is False
    assert configs["dense_bible_boost"].use_reranker is False
    assert configs["dense_bible_boost"].use_lyric_boost is True
    assert configs["dense_bible_full"].use_reranker is True
    assert configs["dense_bible_full"].use_lyric_boost is True


def test_evaluation_report_writes_category_metrics_and_grounding_audit(tmp_path):
    metrics = EvalMetrics(1.0, 0.5, 1.0, 0.75, 1)
    report = EvaluationReport(
        results=(
            EvalRunResult(
                name="dense_bible",
                metrics=metrics,
                k=5,
                category_metrics={"quotation": metrics},
            ),
        ),
        groundings=(
            GroundingResult(
                original_query="Believe in the Lord",
                grounded_query="Believe in the Lord. Bible context: Acts 16:31.",
                reference="Acts 16:31",
                passage_text="Believe in the Lord Jesus Christ.",
                source="passage_bm25",
                confidence=0.9,
                query_type="quotation",
            ),
        ),
        split="development",
        k=5,
    )

    markdown = format_report_markdown(report)
    markdown_path, json_path = write_evaluation_report(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "## Quotation" in markdown
    assert "Acts 16:31" in markdown
    assert markdown_path.exists()
    assert payload["results"][0]["categories"]["quotation"]["ndcg"] == 0.75
    assert payload["grounding_audit"][0]["original_query"] == "Believe in the Lord"
