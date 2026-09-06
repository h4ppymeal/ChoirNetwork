"""Embedding and similarity retrieval for hymns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from choirnetwork.preprocess import (
    DEFAULT_TITLE_WEIGHT,
    build_hymn_chunks,
    preprocess_text,
)
from choirnetwork.query_expand import expand_query
from choirnetwork.scraper import HymnRecord, load_hymns, parse_slug
from choirnetwork.theme_boost import extract_theme_keywords, lyric_keyword_boost

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RECALL_K = 50
INDEX_VERSION = 2
THRESHOLD_RESULT_LIMIT = 50


@dataclass(frozen=True)
class SimilarHymn:
    number: int
    variant: str
    slug: str
    title: str
    score: float

    @property
    def label(self) -> str:
        if self.variant:
            return f"{self.number}{self.variant.upper()}"
        return str(self.number)


@dataclass
class HymnIndex:
    model_name: str
    numbers: list[int]
    variants: list[str]
    slugs: list[str]
    titles: list[str]
    index_version: int = INDEX_VERSION
    cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    title_weight: float = DEFAULT_TITLE_WEIGHT
    recall_k: int = DEFAULT_RECALL_K
    chunk_embeddings: np.ndarray | None = None
    chunk_hymn_indices: np.ndarray | None = None
    chunk_weights: np.ndarray | None = None
    chunk_snippets: list[str] | None = None
    embeddings: np.ndarray | None = None
    lyrics_preprocessed: list[str] | None = None

    @property
    def is_chunked(self) -> bool:
        return self.chunk_embeddings is not None

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "index_version": self.index_version,
            "model_name": self.model_name,
            "cross_encoder_model_name": self.cross_encoder_model_name,
            "title_weight": self.title_weight,
            "recall_k": self.recall_k,
            "numbers": self.numbers,
            "variants": self.variants,
            "slugs": self.slugs,
            "titles": self.titles,
        }
        if self.is_chunked:
            metadata["chunk_snippets"] = self.chunk_snippets
            if self.lyrics_preprocessed:
                metadata["lyrics_preprocessed"] = self.lyrics_preprocessed
            np.save(directory / "chunk_embeddings.npy", self.chunk_embeddings)
            np.save(directory / "chunk_hymn_indices.npy", self.chunk_hymn_indices)
            np.save(directory / "chunk_weights.npy", self.chunk_weights)
        else:
            np.save(directory / "embeddings.npy", self.embeddings)

        with (directory / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> HymnIndex:
        with (directory / "metadata.json").open(encoding="utf-8") as file:
            metadata = json.load(file)

        slugs = metadata.get("slugs") or [str(number) for number in metadata["numbers"]]
        index_version = metadata.get("index_version", 1)

        if index_version >= INDEX_VERSION and (directory / "chunk_embeddings.npy").exists():
            return cls(
                model_name=metadata["model_name"],
                numbers=metadata["numbers"],
                variants=metadata.get("variants", [""] * len(slugs)),
                slugs=slugs,
                titles=metadata["titles"],
                index_version=index_version,
                cross_encoder_model_name=metadata.get(
                    "cross_encoder_model_name",
                    DEFAULT_CROSS_ENCODER_MODEL,
                ),
                title_weight=metadata.get("title_weight", DEFAULT_TITLE_WEIGHT),
                recall_k=metadata.get("recall_k", DEFAULT_RECALL_K),
                chunk_embeddings=np.load(directory / "chunk_embeddings.npy"),
                chunk_hymn_indices=np.load(directory / "chunk_hymn_indices.npy"),
                chunk_weights=np.load(directory / "chunk_weights.npy"),
                chunk_snippets=metadata.get("chunk_snippets", []),
                lyrics_preprocessed=metadata.get("lyrics_preprocessed"),
            )

        return cls(
            model_name=metadata["model_name"],
            numbers=metadata["numbers"],
            variants=metadata.get("variants", [""] * len(slugs)),
            slugs=slugs,
            titles=metadata["titles"],
            index_version=index_version,
            embeddings=np.load(directory / "embeddings.npy"),
        )

    def slug_index(self, slug: str) -> int:
        try:
            return self.slugs.index(slug.lower())
        except ValueError as exc:
            raise ValueError(f"Hymn {slug} is not in the index") from exc

    def hymn_at(self, idx: int, score: float) -> SimilarHymn:
        return SimilarHymn(
            number=self.numbers[idx],
            variant=self.variants[idx],
            slug=self.slugs[idx],
            title=self.titles[idx],
            score=score,
        )


def _normalize_bi_encoder_score(raw_score: float, title_weight: float) -> float:
    """Map weighted max-pool cosine scores (0..title_weight) to 0..1."""
    if title_weight <= 0:
        return max(0.0, min(raw_score, 1.0))
    return max(0.0, min(raw_score / title_weight, 1.0))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _display_percent(score: float) -> int:
    """Match frontend Math.round(score * 100)."""
    if score <= 0:
        return 0
    return int(score * 100 + 0.5)


def _filter_displayable(matches: list[SimilarHymn]) -> list[SimilarHymn]:
    return [match for match in matches if _display_percent(match.score) > 0]


class HymnSimilarityEngine:
    def __init__(
        self,
        index: HymnIndex,
        *,
        use_reranker: bool = True,
        use_lyric_boost: bool = True,
    ):
        self.index = index
        self.use_reranker = use_reranker and index.is_chunked
        self.use_lyric_boost = use_lyric_boost and index.is_chunked
        self._model: SentenceTransformer | None = None
        self._cross_encoder: CrossEncoder | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.index.model_name)
        return self._model

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.index.cross_encoder_model_name)
        return self._cross_encoder

    def _encode_query(self, query: str) -> np.ndarray:
        text = preprocess_text(query, remove_stop_words=True)
        return self._get_model().encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    @classmethod
    def from_raw_hymns(
        cls,
        hymns: list[HymnRecord],
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        title_weight: float = DEFAULT_TITLE_WEIGHT,
        recall_k: int = DEFAULT_RECALL_K,
    ) -> HymnSimilarityEngine:
        chunk_texts: list[str] = []
        chunk_hymn_indices: list[int] = []
        chunk_weights: list[float] = []
        chunk_snippets: list[str] = []

        for hymn_idx, hymn in enumerate(hymns):
            chunks = build_hymn_chunks(
                hymn.title,
                hymn.lyrics,
                title_weight=title_weight,
                remove_stop_words=True,
            )
            for chunk in chunks:
                if not chunk.text:
                    continue
                chunk_texts.append(chunk.text)
                chunk_hymn_indices.append(hymn_idx)
                chunk_weights.append(chunk.weight)
                chunk_snippets.append(chunk.snippet)

        model = SentenceTransformer(model_name)
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        index = HymnIndex(
            model_name=model_name,
            numbers=[hymn.number for hymn in hymns],
            variants=[hymn.variant for hymn in hymns],
            slugs=[hymn.slug for hymn in hymns],
            titles=[hymn.title for hymn in hymns],
            index_version=INDEX_VERSION,
            cross_encoder_model_name=cross_encoder_model_name,
            title_weight=title_weight,
            recall_k=recall_k,
            chunk_embeddings=chunk_embeddings,
            chunk_hymn_indices=np.asarray(chunk_hymn_indices, dtype=np.int32),
            chunk_weights=np.asarray(chunk_weights, dtype=np.float32),
            chunk_snippets=chunk_snippets,
            lyrics_preprocessed=[
                preprocess_text(hymn.lyrics, remove_stop_words=False) for hymn in hymns
            ],
        )
        return cls(index)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float = 0.0,
        exclude_slug: str | None = None,
        expand_query_flag: bool = False,
        use_llm_expansion: bool = False,
    ) -> list[SimilarHymn]:
        """Rank hymns by similarity. top_k=0 returns all matches above min_score."""
        search_query = query
        if expand_query_flag:
            search_query, _ = expand_query(query, use_llm=use_llm_expansion)

        if self.index.is_chunked:
            return self._search_chunked(
                search_query,
                rerank_query=query,
                query_embedding=self._encode_query(search_query),
                top_k=top_k,
                min_score=min_score,
                exclude_slug=exclude_slug,
            )

        return self._rank_legacy(
            self._encode_query(search_query),
            top_k=top_k,
            min_score=min_score,
            exclude_slug=exclude_slug,
        )

    def search_by_slug(self, slug: str, *, top_k: int = 2) -> list[SimilarHymn]:
        idx = self.index.slug_index(slug)
        if self.index.is_chunked:
            assert self.index.chunk_embeddings is not None
            assert self.index.chunk_hymn_indices is not None
            mask = self.index.chunk_hymn_indices == idx
            source_embeddings = self.index.chunk_embeddings[mask]
            query_embedding = source_embeddings.mean(axis=0)
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            return self._search_chunked(
                query=self.index.titles[idx],
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=0.0,
                exclude_slug=slug,
            )

        assert self.index.embeddings is not None
        return self._rank_legacy(
            self.index.embeddings[idx],
            top_k=top_k,
            min_score=0.0,
            exclude_slug=slug,
        )

    def _chunk_scores(self, query_embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.index.chunk_embeddings is not None
        assert self.index.chunk_hymn_indices is not None
        assert self.index.chunk_weights is not None

        scores = self.index.chunk_embeddings @ query_embedding
        weighted_scores = scores * self.index.chunk_weights

        hymn_scores = np.full(len(self.index.slugs), -np.inf, dtype=np.float32)
        np.maximum.at(hymn_scores, self.index.chunk_hymn_indices, weighted_scores)
        return hymn_scores, weighted_scores

    def _apply_lyric_boost(self, hymn_scores: np.ndarray, query: str) -> np.ndarray:
        if not self.index.lyrics_preprocessed:
            return hymn_scores

        keywords = extract_theme_keywords(query)
        if not keywords:
            return hymn_scores

        boosted = hymn_scores.copy()
        title_weight = self.index.title_weight
        for idx, lyrics in enumerate(self.index.lyrics_preprocessed):
            if not np.isfinite(boosted[idx]):
                continue
            boost = lyric_keyword_boost(lyrics, keywords)
            if boost > 0:
                boosted[idx] += boost * title_weight
        return boosted

    def _best_chunk_for_hymn(
        self,
        hymn_idx: int,
        weighted_chunk_scores: np.ndarray,
    ) -> str:
        assert self.index.chunk_hymn_indices is not None
        assert self.index.chunk_snippets is not None

        mask = self.index.chunk_hymn_indices == hymn_idx
        chunk_indices = np.flatnonzero(mask)
        if len(chunk_indices) == 0:
            return self.index.titles[hymn_idx]

        best_chunk_idx = chunk_indices[int(weighted_chunk_scores[chunk_indices].argmax())]
        return self.index.chunk_snippets[best_chunk_idx]

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        weighted_chunk_scores: np.ndarray,
    ) -> list[SimilarHymn]:
        if not candidates:
            return []

        if not self.use_reranker:
            return [
                self.index.hymn_at(
                    idx,
                    _normalize_bi_encoder_score(score, self.index.title_weight),
                )
                for idx, score in candidates
            ]

        pairs = []
        for hymn_idx, _ in candidates:
            snippet = self._best_chunk_for_hymn(hymn_idx, weighted_chunk_scores)
            pairs.append((query, f"{self.index.titles[hymn_idx]}. {snippet}"))

        cross_scores = self._get_cross_encoder().predict(pairs)
        reranked = sorted(
            zip(candidates, cross_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            self.index.hymn_at(hymn_idx, _sigmoid(float(cross_score)))
            for (hymn_idx, _), cross_score in reranked
        ]

    def _search_chunked(
        self,
        query: str,
        *,
        rerank_query: str | None = None,
        query_embedding: np.ndarray,
        top_k: int,
        min_score: float,
        exclude_slug: str | None,
    ) -> list[SimilarHymn]:
        rerank_query = rerank_query or query
        hymn_scores, weighted_chunk_scores = self._chunk_scores(query_embedding)
        if self.use_lyric_boost:
            hymn_scores = self._apply_lyric_boost(hymn_scores, query)
        recall_limit = self.index.recall_k if top_k > 0 else THRESHOLD_RESULT_LIMIT
        title_weight = self.index.title_weight

        candidates: list[tuple[int, float]] = []
        for idx in np.argsort(-hymn_scores):
            normalized = _normalize_bi_encoder_score(float(hymn_scores[idx]), title_weight)
            if normalized < min_score:
                break
            slug = self.index.slugs[idx]
            if exclude_slug and slug == exclude_slug.lower():
                continue
            candidates.append((int(idx), normalized))
            if len(candidates) == recall_limit:
                break

        reranked = self._rerank_candidates(rerank_query, candidates, weighted_chunk_scores)
        limit = top_k if top_k > 0 else THRESHOLD_RESULT_LIMIT
        displayable = _filter_displayable(reranked)

        # Only trust rerank when we have enough confident results; otherwise
        # a single low-score hit (e.g. 4%) would hide better bi-encoder matches.
        if len(displayable) >= limit:
            return displayable[:limit]

        if candidates:
            fallback = [self.index.hymn_at(idx, score) for idx, score in candidates[:limit]]
            displayable_fallback = _filter_displayable(fallback)
            return displayable_fallback[:limit] if displayable_fallback else fallback[:limit]

        return displayable[:limit]

    def _finalize_results(self, matches: list[SimilarHymn], *, top_k: int) -> list[SimilarHymn]:
        displayable = _filter_displayable(matches)
        limit = top_k if top_k > 0 else THRESHOLD_RESULT_LIMIT
        return displayable[:limit]

    def _rank_legacy(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
        min_score: float,
        exclude_slug: str | None,
    ) -> list[SimilarHymn]:
        assert self.index.embeddings is not None
        scores = self.index.embeddings @ query_embedding
        limit = top_k if top_k > 0 else THRESHOLD_RESULT_LIMIT

        results: list[SimilarHymn] = []
        for idx in np.argsort(-scores):
            score = float(scores[idx])
            if score < min_score:
                break
            slug = self.index.slugs[idx]
            if exclude_slug and slug == exclude_slug.lower():
                continue
            results.append(self.index.hymn_at(idx, score))
            if len(results) == limit:
                break

        return self._finalize_results(results, top_k=top_k)

    def resolve_slug(self, hymn_id: str) -> str:
        normalized = hymn_id.lower().strip()
        if normalized in self.index.slugs:
            return normalized

        number, variant = parse_slug(normalized)
        if variant:
            raise ValueError(f"Hymn {hymn_id} is not in the index")

        matches = [slug for slug in self.index.slugs if parse_slug(slug)[0] == number]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            labels = ", ".join(
                f"{n}{v.upper()}" if v else str(n)
                for slug in matches
                for n, v in [parse_slug(slug)]
            )
            raise ValueError(
                f"Hymn {number} has multiple variants ({labels}). "
                f"Specify one, e.g. {matches[0]}."
            )
        raise ValueError(f"Hymn {hymn_id} is not in the index")


def _backfill_lyrics(index: HymnIndex, hymns_path: Path | None = None) -> None:
    """Load preprocessed lyrics for lyric boosting when the index omits them."""
    if index.lyrics_preprocessed and len(index.lyrics_preprocessed) == len(index.slugs):
        return

    hymns_path = hymns_path or Path("data/raw/hymns.json")
    if not hymns_path.exists():
        index.lyrics_preprocessed = [""] * len(index.slugs)
        return

    slug_to_lyrics = {hymn.slug: hymn.lyrics for hymn in load_hymns(hymns_path)}
    index.lyrics_preprocessed = [
        preprocess_text(slug_to_lyrics.get(slug, ""), remove_stop_words=False)
        for slug in index.slugs
    ]


def save_engine(engine: HymnSimilarityEngine, directory: Path) -> None:
    engine.index.save(directory)


def load_engine(
    directory: Path,
    *,
    use_reranker: bool = True,
    use_lyric_boost: bool = True,
) -> HymnSimilarityEngine:
    index = HymnIndex.load(directory)
    _backfill_lyrics(index)
    return HymnSimilarityEngine(
        index,
        use_reranker=use_reranker,
        use_lyric_boost=use_lyric_boost,
    )
