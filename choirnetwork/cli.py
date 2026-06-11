"""Command-line interface for ChoirNetwork."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from choirnetwork.config import load_env

from choirnetwork.engine import (
    HymnSimilarityEngine,
    load_engine,
    save_engine,
)
from choirnetwork.eval import (
    DEFAULT_EVAL_PATH,
    compare_retrieval_configs,
    format_comparison_table,
)
from choirnetwork.query_expand import expand_query, get_last_llm_error
from choirnetwork.scraper import (
    DEFAULT_HYMN_COUNT,
    fetch_catalog,
    load_hymns,
    lookup_by_title,
    save_hymns,
    save_missing_lyrics,
    scrape_hymnal,
)

DEFAULT_RAW_PATH = Path("data/raw/hymns.json")
DEFAULT_MISSING_PATH = Path("data/raw/missing_lyrics.json")
DEFAULT_INDEX_PATH = Path("data/index")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ChoirNetwork: semantic hymn similarity search for the TJC hymnal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape hymn lyrics from hymnal.tjc.org using the library title index",
    )
    scrape_parser.add_argument(
        "--title",
        action="append",
        dest="titles",
        help="Scrape only hymns matching this title (repeatable)",
    )
    scrape_parser.add_argument("--start", type=int, default=None)
    scrape_parser.add_argument("--end", type=int, default=None)
    scrape_parser.add_argument("--output", type=Path, default=DEFAULT_RAW_PATH)
    scrape_parser.add_argument(
        "--missing-output",
        type=Path,
        default=DEFAULT_MISSING_PATH,
        help="Write hymns with no online lyrics to this file",
    )

    lookup_parser = subparsers.add_parser(
        "lookup-title",
        help="Find hymnal slug(s) for a title in the library index",
    )
    lookup_parser.add_argument("title")

    build_parser = subparsers.add_parser(
        "build",
        help="Embed scraped hymns and build a similarity index",
    )
    build_parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)

    search_parser = subparsers.add_parser(
        "search",
        help="Find the most similar hymns for a title query",
    )
    search_parser.add_argument("query", help="Hymn or sermon title")
    search_parser.add_argument("--top-k", type=int, default=2)
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    search_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking (faster, lower quality)",
    )
    search_parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand sermon/Bible topics into thematic search terms",
    )
    search_parser.add_argument(
        "--llm-expand",
        action="store_true",
        help="Use OpenAI for query expansion when no curated theme exists (requires OPENAI_API_KEY)",
    )

    number_parser = subparsers.add_parser(
        "search-by-number",
        help="Find the most similar hymns for a hymnal number or variant (e.g. 51a)",
    )
    number_parser.add_argument(
        "hymn_id",
        help="Hymn number or variant slug, e.g. 1, 51a, 124b",
    )
    number_parser.add_argument("--top-k", type=int, default=2)
    number_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    number_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking (faster, lower quality)",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the web app for sermon-title hymn search",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    serve_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking (faster, lower quality)",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Compare retrieval configs on the labeled eval set (Hit@k, Recall@k, MRR, nDCG)",
    )
    eval_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    eval_parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_PATH)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument(
        "--llm-expand",
        action="store_true",
        help="Allow LLM query expansion for uncovered eval queries (requires OPENAI_API_KEY)",
    )

    return parser


def cmd_scrape(args: argparse.Namespace) -> None:
    start = args.start
    end = args.end
    if start is None and end is None and not args.titles:
        start, end = 1, DEFAULT_HYMN_COUNT

    hymns, missing = scrape_hymnal(
        titles=args.titles,
        start=start,
        end=end,
    )
    output_path = save_hymns(hymns, args.output)
    missing_path = save_missing_lyrics(missing, args.missing_output)
    print(f"Saved {len(hymns)} hymns to {output_path}")
    print(f"Saved {len(missing)} title-only entries to {missing_path}")


def cmd_lookup_title(args: argparse.Namespace) -> None:
    matches = lookup_by_title(args.title)
    if not matches:
        raise SystemExit(f"No catalog match for title: {args.title}")

    print(f"Matches for '{args.title}':")
    for entry in matches:
        print(f"  {entry.label}: {entry.title}")
        print(f"    {entry.url}")


def cmd_build(args: argparse.Namespace) -> None:
    hymns = load_hymns(args.input)
    engine = HymnSimilarityEngine.from_raw_hymns(hymns)
    save_engine(engine, args.output)
    print(f"Built index for {len(hymns)} hymns at {args.output}")


def cmd_search(args: argparse.Namespace) -> None:
    engine = load_engine(args.index, use_reranker=not args.no_rerank)
    display_query = args.query
    if args.expand or args.llm_expand:
        display_query, source = expand_query(args.query, use_llm=args.llm_expand)
        if source:
            print(f"Expanded ({source}): {display_query}")
        elif args.llm_expand:
            import os

            if not os.environ.get("OPENAI_API_KEY", "").strip():
                print(
                    "Warning: --llm-expand requires OPENAI_API_KEY in .env; "
                    "searching without expansion.",
                    file=sys.stderr,
                )
            else:
                detail = get_last_llm_error() or "unknown error"
                print(
                    f"Warning: LLM expansion failed ({detail}); searching without expansion.",
                    file=sys.stderr,
                )
        elif args.expand:
            print("Note: no curated theme matched; searching the raw query.")
    matches = engine.search(
        args.query,
        top_k=args.top_k,
        expand_query_flag=args.expand or args.llm_expand,
        use_llm_expansion=args.llm_expand,
    )
    _print_matches(display_query, matches)


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["CHOIRNETWORK_INDEX_PATH"] = str(args.index)
    os.environ["CHOIRNETWORK_USE_RERANKER"] = "0" if args.no_rerank else "1"
    uvicorn.run(
        "choirnetwork.api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    results = compare_retrieval_configs(
        args.index,
        args.eval_file,
        k=args.top_k,
        use_llm=args.llm_expand,
    )
    print(format_comparison_table(results))


def cmd_search_by_number(args: argparse.Namespace) -> None:
    engine = load_engine(args.index, use_reranker=not args.no_rerank)
    try:
        slug = engine.resolve_slug(args.hymn_id)
        idx = engine.index.slug_index(slug)
        source_title = engine.index.titles[idx]
        source_label = (
            f"{engine.index.numbers[idx]}{engine.index.variants[idx].upper()}"
            if engine.index.variants[idx]
            else str(engine.index.numbers[idx])
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    matches = engine.search_by_slug(slug, top_k=args.top_k)
    _print_matches(f"Hymn {source_label} - {source_title}", matches)


def _print_matches(query: str, matches) -> None:
    print(f"Query: {query}")
    print("Similar hymns:")
    for rank, match in enumerate(matches, start=1):
        print(f"  {rank}. Hymn {match.label} - {match.title} (score={match.score:.3f})")


def main(argv: list[str] | None = None) -> None:
    load_env()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scrape":
        cmd_scrape(args)
    elif args.command == "lookup-title":
        cmd_lookup_title(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "search-by-number":
        cmd_search_by_number(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
