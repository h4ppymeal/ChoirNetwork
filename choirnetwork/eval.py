"""Offline evaluation metrics for hymn retrieval."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from choirnetwork.bm25 import BM25Retriever
from choirnetwork.engine import HymnIndex, HymnSimilarityEngine, load_engine

DEFAULT_EVAL_PATH = Path("eval/datasets/service_hymns.csv")


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_slugs: tuple[str, ...]
    category: str
    notes: str = ""
    split: str = "development"


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


def evaluate_engine(
    engine: HymnSimilarityEngine,
    eval_queries: list[EvalQuery],
    *,
    k: int = 5,
    expand_query_flag: bool = False,
    use_llm: bool = False,
) -> EvalMetrics:
    return evaluate_retriever(
        eval_queries,
        lambda query, top_k: [
            match.slug
            for match in engine.search(
                query,
                top_k=top_k,
                expand_query_flag=expand_query_flag,
                use_llm_expansion=use_llm,
            )
        ],
        k=k,
    )


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


def compare_retrieval_configs(
    index_path: Path,
    eval_path: Path,
    *,
    k: int = 5,
    use_llm: bool = False,
    split: str | None = "development",
) -> list[EvalRunResult]:
    index = load_engine(index_path).index
    eval_queries = prepare_eval_queries(index, eval_path, split=split)
    if not eval_queries:
        raise SystemExit(
            f"No eval queries matched the index at {index_path}. "
            f"Build the index first, then verify labels in {eval_path}."
        )

    neural_configs = [
        ("bi_encoder_only", False, False, False),
        ("chunked_rerank", True, False, False),
        ("chunked_rerank_expand", True, True, False),
        ("full_system", True, True, True),
    ]

    bm25 = BM25Retriever(index)
    bm25_metrics = evaluate_retriever(
        eval_queries,
        lambda query, top_k: [
            match.slug for match in bm25.search(query, top_k=top_k)
        ],
        k=k,
    )
    results = [EvalRunResult(name="bm25", metrics=bm25_metrics, k=k)]

    for name, use_reranker, expand_flag, use_lyric_boost in neural_configs:
        engine = load_engine(
            index_path,
            use_reranker=use_reranker,
            use_lyric_boost=use_lyric_boost,
        )
        metrics = evaluate_engine(
            engine,
            eval_queries,
            k=k,
            expand_query_flag=expand_flag,
            use_llm=use_llm,
        )
        results.append(EvalRunResult(name=name, metrics=metrics, k=k))

    return results


def format_comparison_table(results: list[EvalRunResult]) -> str:
    if not results:
        return "No eval results."

    k = results[0].k
    header = f"Retrieval evaluation (k={k}, n={results[0].metrics.queries_evaluated})"
    lines = [
        header,
        "-" * len(header),
        f"{'Config':<26} {'Hit@k':>8} {'Recall@k':>10} {'MRR@k':>8} {'nDCG@k':>8}",
    ]
    for result in results:
        metrics = result.metrics
        lines.append(
            f"{result.name:<26} "
            f"{metrics.hit_rate:>7.1%} "
            f"{metrics.recall:>9.1%} "
            f"{metrics.mrr:>7.1%} "
            f"{metrics.ndcg:>7.1%}"
        )
    return "\n".join(lines)
