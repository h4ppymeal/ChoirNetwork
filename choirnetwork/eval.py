"""Offline evaluation metrics for hymn retrieval."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from choirnetwork.bible_grounding import (
    BM25_B,
    BM25_K1,
    DEFAULT_BIBLE_PATH,
    NARRATIVE_MIN_COVERAGE,
    NARRATIVE_MIN_CONFIDENCE,
    NARRATIVE_MIN_TERMS,
    PASSAGE_WINDOW_SIZE,
    PASSAGE_WINDOW_STRIDE,
    QUOTATION_MIN_COVERAGE,
    QUOTATION_MIN_TERMS,
    RARE_DOCUMENT_FREQUENCY,
    BibleGrounder,
    GroundingResult,
)
from choirnetwork.bm25 import BM25Retriever
from choirnetwork.engine import HymnIndex, HymnSimilarityEngine, load_engine

DEFAULT_EVAL_PATH = Path("eval/datasets/service_hymns.csv")
DEFAULT_RESULTS_PATH = Path("eval/results")


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_slugs: tuple[str, ...]
    category: str
    notes: str = ""
    split: str = "development"
    grounding_type: str = "unknown"


@dataclass(frozen=True)
class EvalMetrics:
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    queries_evaluated: int


@dataclass(frozen=True)
class EvalRunResult:
    name: str
    metrics: EvalMetrics
    k: int
    category_metrics: dict[str, EvalMetrics] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    retriever: str
    use_bible: bool
    use_reranker: bool = False
    use_lyric_boost: bool = False


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[EvalRunResult, ...]
    groundings: tuple[GroundingResult, ...]
    split: str
    k: int


RETRIEVAL_CONFIGS = (
    RetrievalConfig("bm25_title", "bm25", False),
    RetrievalConfig("bm25_bible", "bm25", True),
    RetrievalConfig("dense_title", "dense", False),
    RetrievalConfig(
        "dense_title_rerank",
        "dense",
        False,
        use_reranker=True,
    ),
    RetrievalConfig(
        "dense_title_boost",
        "dense",
        False,
        use_lyric_boost=True,
    ),
    RetrievalConfig(
        "dense_title_full",
        "dense",
        False,
        use_reranker=True,
        use_lyric_boost=True,
    ),
    RetrievalConfig("dense_bible", "dense", True),
    RetrievalConfig("dense_bible_rerank", "dense", True, use_reranker=True),
    RetrievalConfig("dense_bible_boost", "dense", True, use_lyric_boost=True),
    RetrievalConfig(
        "dense_bible_full",
        "dense",
        True,
        use_reranker=True,
        use_lyric_boost=True,
    ),
)


def load_eval_queries(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as file:
            rows = []
            for row in csv.DictReader(file):
                rows.append(
                    {
                        "query": row["query"].strip(),
                        "relevant_slugs": [
                            row["hymn_1_slug"].strip(),
                            row["hymn_2_slug"].strip(),
                        ],
                        "category": (
                            row.get("category") or "observed_service"
                        ).strip(),
                        "split": (row.get("split") or "development").strip(),
                        "notes": (row.get("notes") or "").strip(),
                    }
                )
            return rows

    queries: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def resolve_relevant_slugs(index: HymnIndex, item: dict) -> list[str]:
    slugs: set[str] = set()

    for slug in item.get("relevant_slugs", []):
        normalized = str(slug).lower().strip()
        if normalized in index.slugs:
            slugs.add(normalized)

    for title in item.get("relevant_titles", []):
        needle = title.lower().strip()
        for idx, hymn_title in enumerate(index.titles):
            if needle in hymn_title.lower():
                slugs.add(index.slugs[idx])

    return sorted(slugs)


def prepare_eval_queries(
    index: HymnIndex,
    path: Path,
    *,
    split: str | None = None,
) -> list[EvalQuery]:
    prepared: list[EvalQuery] = []
    for item in load_eval_queries(path):
        item_split = item.get("split", "development")
        if split is not None and item_split != split:
            continue
        relevant_slugs = resolve_relevant_slugs(index, item)
        if not relevant_slugs:
            continue
        prepared.append(
            EvalQuery(
                query=item["query"],
                relevant_slugs=tuple(relevant_slugs),
                category=item.get("category", "unknown"),
                notes=item.get("notes", ""),
                split=item_split,
            )
        )
    return prepared


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant.intersection(retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, slug in enumerate(retrieved[:k], start=1):
        if slug in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(scores: list[float]) -> float:
        total = 0.0
        for index, score in enumerate(scores):
            total += score / math.log2(index + 2)
        return total

    gains = [1.0 if slug in relevant else 0.0 for slug in retrieved[:k]]
    ideal_gains = [1.0] * min(len(relevant), k)
    ideal_gains.extend([0.0] * max(0, k - len(ideal_gains)))

    ideal_dcg = dcg(ideal_gains)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def evaluate_retriever(
    eval_queries: list[EvalQuery],
    retrieve: Callable[[str, int], list[str]],
    *,
    k: int = 5,
) -> EvalMetrics:
    if not eval_queries:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0)

    hit_rates: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []

    for item in eval_queries:
        retrieved = retrieve(item.query, k)
        relevant = set(item.relevant_slugs)

        hit_rates.append(hit_rate_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        mrrs.append(mrr_at_k(retrieved, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))

    count = len(eval_queries)
    return EvalMetrics(
        hit_rate=sum(hit_rates) / count,
        recall=sum(recalls) / count,
        mrr=sum(mrrs) / count,
        ndcg=sum(ndcgs) / count,
        queries_evaluated=count,
    )


def _metrics_from_rankings(
    eval_queries: list[EvalQuery],
    rankings: list[list[str]],
    *,
    k: int,
) -> EvalMetrics:
    if len(eval_queries) != len(rankings):
        raise ValueError("Each evaluation query requires one ranking")
    if not eval_queries:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0)

    values = []
    for query, retrieved in zip(eval_queries, rankings):
        relevant = set(query.relevant_slugs)
        values.append(
            (
                hit_rate_at_k(retrieved, relevant, k),
                recall_at_k(retrieved, relevant, k),
                mrr_at_k(retrieved, relevant, k),
                ndcg_at_k(retrieved, relevant, k),
            )
        )
    count = len(values)
    return EvalMetrics(
        hit_rate=sum(value[0] for value in values) / count,
        recall=sum(value[1] for value in values) / count,
        mrr=sum(value[2] for value in values) / count,
        ndcg=sum(value[3] for value in values) / count,
        queries_evaluated=count,
    )


def _category_metrics(
    eval_queries: list[EvalQuery],
    rankings: list[list[str]],
    *,
    k: int,
) -> dict[str, EvalMetrics]:
    grouped: dict[str, tuple[list[EvalQuery], list[list[str]]]] = {}
    for query, ranking in zip(eval_queries, rankings):
        category_queries, category_rankings = grouped.setdefault(
            query.grounding_type, ([], [])
        )
        category_queries.append(query)
        category_rankings.append(ranking)
    return {
        category: _metrics_from_rankings(queries, list(category_rankings), k=k)
        for category, (queries, category_rankings) in sorted(grouped.items())
    }


def _queries_with_grounding(
    eval_queries: list[EvalQuery],
    grounder: BibleGrounder,
) -> tuple[list[EvalQuery], list[EvalQuery], list[GroundingResult]]:
    title_queries: list[EvalQuery] = []
    bible_queries: list[EvalQuery] = []
    groundings: list[GroundingResult] = []
    for query in eval_queries:
        grounding = grounder.ground(query.query)
        groundings.append(grounding)
        shared = {
            "relevant_slugs": query.relevant_slugs,
            "category": query.category,
            "notes": query.notes,
            "split": query.split,
            "grounding_type": grounding.query_type,
        }
        title_queries.append(EvalQuery(query=query.query, **shared))
        bible_queries.append(EvalQuery(query=grounding.grounded_query, **shared))
    return title_queries, bible_queries, groundings


def compare_retrieval_configs(
    index_path: Path,
    eval_path: Path,
    *,
    k: int = 5,
    split: str = "development",
    bible_path: Path = DEFAULT_BIBLE_PATH,
) -> EvaluationReport:
    index = load_engine(index_path).index
    eval_queries = prepare_eval_queries(index, eval_path, split=split)
    if not eval_queries:
        raise SystemExit(
            f"No eval queries matched the index at {index_path}. "
            f"Build the index first, then verify labels in {eval_path}."
        )

    grounder = BibleGrounder.load(bible_path)
    title_queries, bible_queries, groundings = _queries_with_grounding(
        eval_queries, grounder
    )
    bm25 = BM25Retriever(index)
    engines: dict[tuple[bool, bool], HymnSimilarityEngine] = {}
    results: list[EvalRunResult] = []

    for config in RETRIEVAL_CONFIGS:
        queries = bible_queries if config.use_bible else title_queries
        if config.retriever == "bm25":
            rankings = [
                [match.slug for match in bm25.search(query.query, top_k=k)]
                for query in queries
            ]
        else:
            engine_key = (config.use_reranker, config.use_lyric_boost)
            if engine_key not in engines:
                engines[engine_key] = load_engine(
                    index_path,
                    use_reranker=config.use_reranker,
                    use_lyric_boost=config.use_lyric_boost,
                )
            engine = engines[engine_key]
            rankings = [
                [match.slug for match in engine.search(query.query, top_k=k)]
                for query in queries
            ]
        results.append(
            EvalRunResult(
                name=config.name,
                metrics=_metrics_from_rankings(queries, list(rankings), k=k),
                k=k,
                category_metrics=_category_metrics(queries, rankings, k=k),
            )
        )

    return EvaluationReport(
        results=tuple(results),
        groundings=tuple(groundings),
        split=split,
        k=k,
    )


def format_comparison_table(report: EvaluationReport) -> str:
    if not report.results:
        return "No eval results."

    k = report.k
    header = (
        f"Retrieval evaluation "
        f"(split={report.split}, k={k}, "
        f"n={report.results[0].metrics.queries_evaluated})"
    )
    lines = [
        header,
        "-" * len(header),
        f"{'Config':<26} {'Hit@k':>8} {'Recall@k':>10} {'MRR@k':>8} {'nDCG@k':>8}",
    ]
    for result in report.results:
        metrics = result.metrics
        lines.append(
            f"{result.name:<26} "
            f"{metrics.hit_rate:>7.1%} "
            f"{metrics.recall:>9.1%} "
            f"{metrics.mrr:>7.1%} "
            f"{metrics.ndcg:>7.1%}"
        )
    return "\n".join(lines)


def _metric_dict(metrics: EvalMetrics) -> dict:
    return asdict(metrics)


def report_payload(report: EvaluationReport) -> dict:
    return {
        "split": report.split,
        "k": report.k,
        "grounding": {
            "bible": "World English Bible (Public Domain)",
            "window_size": PASSAGE_WINDOW_SIZE,
            "window_stride": PASSAGE_WINDOW_STRIDE,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "quotation_min_terms": QUOTATION_MIN_TERMS,
            "quotation_min_coverage": QUOTATION_MIN_COVERAGE,
            "narrative_min_coverage": NARRATIVE_MIN_COVERAGE,
            "narrative_min_confidence": NARRATIVE_MIN_CONFIDENCE,
            "narrative_min_terms": NARRATIVE_MIN_TERMS,
            "rare_document_frequency": RARE_DOCUMENT_FREQUENCY,
        },
        "results": [
            {
                "name": result.name,
                "metrics": _metric_dict(result.metrics),
                "categories": {
                    category: _metric_dict(metrics)
                    for category, metrics in result.category_metrics.items()
                },
            }
            for result in report.results
        ],
        "grounding_audit": [asdict(grounding) for grounding in report.groundings],
    }


def _markdown_metrics(metrics: EvalMetrics) -> str:
    return (
        f"{metrics.queries_evaluated} | {metrics.hit_rate:.1%} | "
        f"{metrics.recall:.1%} | {metrics.mrr:.1%} | {metrics.ndcg:.1%}"
    )


def format_report_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Bible-grounded {report.split} results",
        "",
        f"`k={report.k}` · World English Bible (Public Domain) · "
        "curated and LLM expansion disabled",
        "",
        "## Aggregate",
        "",
        f"| Configuration | n | Hit@{report.k} | Recall@{report.k} | "
        f"MRR@{report.k} | nDCG@{report.k} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in report.results:
        lines.append(f"| {result.name} | {_markdown_metrics(result.metrics)} |")

    categories = sorted(
        {
            category
            for result in report.results
            for category in result.category_metrics
        }
    )
    for category in categories:
        lines.extend(
            [
                "",
                f"## {category.replace('_', ' ').title()}",
                "",
                f"| Configuration | n | Hit@{report.k} | "
                f"Recall@{report.k} | MRR@{report.k} | nDCG@{report.k} |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for result in report.results:
            metrics = result.category_metrics.get(category)
            if metrics:
                lines.append(f"| {result.name} | {_markdown_metrics(metrics)} |")

    lines.extend(
        [
            "",
            "## Grounding audit",
            "",
            "| Query | Type | Reference | Confidence |",
            "|---|---|---|---:|",
        ]
    )
    for grounding in report.groundings:
        query = grounding.original_query.replace("|", "\\|")
        reference = (grounding.reference or "—").replace("|", "\\|")
        lines.append(
            f"| {query} | {grounding.query_type} | {reference} | "
            f"{grounding.confidence:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_evaluation_report(
    report: EvaluationReport,
    output_directory: Path = DEFAULT_RESULTS_PATH,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"service-{report.split}-bible"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report_payload(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(format_report_markdown(report), encoding="utf-8")
    return markdown_path, json_path
