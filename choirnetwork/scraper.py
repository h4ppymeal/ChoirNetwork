"""Scrape hymn metadata and lyrics from hymnal.tjc.org."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cloudscraper
from bs4 import BeautifulSoup

from choirnetwork.preprocess import normalize_text_for_match

HYMN_BASE_URL = "https://hymnal.tjc.org/hymnal-library"
DEFAULT_HYMN_COUNT = 525
REQUEST_DELAY_SECONDS = 0.4
MAX_RETRIES = 3
HYMN_HEADING_RE = re.compile(r"Hymn\s+([\d\s]+)([A-Za-z]?)\s*-\s*(.+)")
SLUG_RE = re.compile(r"^(\d+)([a-z]?)$")
CATALOG_LINK_RE = re.compile(r"/hymnal-library/([\w]+)$")


def hymn_label(number: int, variant: str = "") -> str:
    return f"{number}{variant.upper()}" if variant else str(number)


@dataclass(frozen=True)
class HymnRecord:
    number: int
    variant: str
    title: str
    lyrics: str
    url: str
    slug: str
    bible_verse: str = ""
    bible_reference: str = ""
    lyrics_source: str = HYMN_BASE_URL
    pdf_page: int | None = None

    @property
    def label(self) -> str:
        return hymn_label(self.number, self.variant)


@dataclass(frozen=True)
class CatalogEntry:
    slug: str
    number: int
    variant: str
    title: str
    url: str

    @property
    def label(self) -> str:
        return hymn_label(self.number, self.variant)


def parse_slug(slug: str) -> tuple[int, str]:
    match = SLUG_RE.match(slug.lower())
    if not match:
        raise ValueError(f"Invalid hymn slug: {slug}")
    return int(match.group(1)), match.group(2)


def hymn_url(slug: str) -> str:
    return f"{HYMN_BASE_URL}/{slug.lower()}"


def _parse_catalog_label(slug: str, label: str) -> CatalogEntry | None:
    label = label.replace("\xa0", " ").strip()
    label = re.sub(r"\s*🎧\s*$", "", label).strip()
    match = re.match(r"^(\d+)([A-Za-z]?)\s*-\s*(.+)$", label)
    if not match:
        return None

    number = int(match.group(1))
    variant = match.group(2).lower()
    title = match.group(3).strip()
    normalized_slug = f"{number}{variant}" if variant else str(number)
    if normalized_slug != slug.lower():
        slug = normalized_slug

    return CatalogEntry(
        slug=slug.lower(),
        number=number,
        variant=variant,
        title=title,
        url=hymn_url(slug),
    )


def fetch_catalog(
    scraper: cloudscraper.CloudScraper | None = None,
) -> list[CatalogEntry]:
    """Load slug/title pairs from the hymnal library index page."""
    scraper = scraper or cloudscraper.create_scraper()
    response = scraper.get(HYMN_BASE_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    entries: dict[str, CatalogEntry] = {}
    for anchor in soup.find_all("a", href=True):
        match = CATALOG_LINK_RE.search(anchor["href"])
        if not match:
            continue
        slug = match.group(1).lower()
        label = anchor.get_text(" ", strip=True)
        if not label:
            continue
        entry = _parse_catalog_label(slug, label)
        if entry is not None:
            entries[entry.slug] = entry

    return sorted(entries.values(), key=lambda entry: (entry.number, entry.variant))


def lookup_by_title(
    title: str,
    catalog: list[CatalogEntry] | None = None,
) -> list[CatalogEntry]:
    """Find catalog entries matching a hymn title (exact, then substring)."""
    catalog = catalog or fetch_catalog()
    normalized_query = normalize_text_for_match(title)
    if not normalized_query:
        return []

    exact = [
        entry
        for entry in catalog
        if normalize_text_for_match(entry.title) == normalized_query
    ]
    if exact:
        return exact

    substring = [
        entry
        for entry in catalog
        if normalized_query in normalize_text_for_match(entry.title)
        or normalize_text_for_match(entry.title) in normalized_query
    ]
    return substring


def _find_hymn_heading(soup: BeautifulSoup) -> str | None:
    for heading in soup.find_all("h2"):
        text = heading.get_text(" ", strip=True)
        if HYMN_HEADING_RE.match(text):
            return text
    return None


def _find_lyrics_header(soup: BeautifulSoup):
    for heading in soup.find_all("h3"):
        if heading.get_text(strip=True) == "Lyrics":
            return heading
    return None


def _extract_lyrics(lyrics_header) -> str:
    lines: list[str] = []
    for node in lyrics_header.find_all_next():
        if node.name in ("h2", "h3") and node is not lyrics_header:
            break
        if node.name == "p":
            line = node.get_text(" ", strip=True)
            if line:
                lines.append(line)
    return "\n".join(lines).strip()


def parse_hymn_page(html: str, slug: str, url: str) -> HymnRecord | None:
    soup = BeautifulSoup(html, "lxml")
    heading = _find_hymn_heading(soup)
    if not heading:
        return None

    match = HYMN_HEADING_RE.match(heading)
    if not match:
        return None

    parsed_number = int(re.sub(r"\s+", "", match.group(1)))
    parsed_variant = match.group(2).lower()
    title = match.group(3).strip()
    expected_number, expected_variant = parse_slug(slug)

    lyrics_header = _find_lyrics_header(soup)
    if lyrics_header is None:
        return None

    lyrics = _extract_lyrics(lyrics_header)
    if not lyrics:
        return None

    return HymnRecord(
        number=parsed_number if parsed_number == expected_number else expected_number,
        variant=parsed_variant or expected_variant,
        title=title,
        lyrics=lyrics,
        url=url,
        slug=slug.lower(),
    )


def scrape_slug(
    scraper: cloudscraper.CloudScraper,
    slug: str,
    *,
    retries: int = MAX_RETRIES,
) -> HymnRecord | None:
    url = hymn_url(slug)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = scraper.get(url, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return parse_hymn_page(response.text, slug, url)
        except Exception as exc:  # noqa: BLE001 - retry wrapper
            last_error = exc
            if attempt < retries:
                time.sleep(REQUEST_DELAY_SECONDS * attempt)

    raise RuntimeError(f"Failed to scrape hymn {slug}") from last_error


def scrape_hymnal(
    *,
    titles: list[str] | None = None,
    start: int | None = None,
    end: int | None = None,
    catalog: list[CatalogEntry] | None = None,
    delay_seconds: float = REQUEST_DELAY_SECONDS,
    progress: bool = True,
) -> tuple[list[HymnRecord], list[CatalogEntry]]:
    """
    Scrape hymns using the library index (title -> slug), not blind number guessing.

    Returns scraped hymns with lyrics and catalog entries that had no lyrics online.
    """
    scraper = cloudscraper.create_scraper()
    catalog = catalog or fetch_catalog(scraper)

    if titles:
        selected: dict[str, CatalogEntry] = {}
        for title in titles:
            matches = lookup_by_title(title, catalog)
            if not matches:
                if progress:
                    print(f"No catalog match for title: {title}")
                continue
            for entry in matches:
                selected[entry.slug] = entry
        entries = sorted(selected.values(), key=lambda entry: (entry.number, entry.variant))
    else:
        entries = catalog
        if start is not None:
            entries = [entry for entry in entries if entry.number >= start]
        if end is not None:
            entries = [entry for entry in entries if entry.number <= end]

    hymns: list[HymnRecord] = []
    missing_lyrics: list[CatalogEntry] = []
    failures: list[str] = []

    for index, entry in enumerate(entries, start=1):
        try:
            hymn = scrape_slug(scraper, entry.slug)
        except RuntimeError:
            failures.append(entry.slug)
            if progress:
                print(f"Failed hymn {entry.label}")
            continue

        if hymn is None:
            missing_lyrics.append(entry)
            if progress:
                print(f"No lyrics  {entry.label:>4}: {entry.title}")
        else:
            hymns.append(hymn)
            if progress:
                print(f"Scraped hymn {hymn.label:>4} ({len(hymns):>3} found): {hymn.title}")

        if index < len(entries):
            time.sleep(delay_seconds)

    if failures and progress:
        print(f"Failed to fetch {len(failures)} slugs: {', '.join(failures)}")
    if missing_lyrics and progress:
        print(f"{len(missing_lyrics)} hymns have pages but no lyrics on hymnal.tjc.org")

    return hymns, missing_lyrics


def save_hymns(hymns: Iterable[HymnRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hymn_list = list(hymns)
    payload = {
        "source": HYMN_BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "count": len(hymn_list),
        "hymns": [asdict(hymn) for hymn in hymn_list],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return output_path


def save_missing_lyrics(
    entries: Iterable[CatalogEntry],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry_list = list(entries)
    payload = {
        "source": HYMN_BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "count": len(entry_list),
        "reason": "Page exists on hymnal.tjc.org but no Lyrics section is published",
        "hymns": [asdict(entry) for entry in entry_list],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return output_path


def load_hymns(input_path: Path) -> list[HymnRecord]:
    with input_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    hymns: list[HymnRecord] = []
    for item in payload["hymns"]:
        hymns.append(
            HymnRecord(
                number=item["number"],
                variant=item.get("variant", ""),
                title=item["title"],
                lyrics=item["lyrics"],
                url=item["url"],
                slug=item.get("slug", str(item["number"])),
                bible_verse=item.get("bible_verse", ""),
                bible_reference=item.get("bible_reference", ""),
                lyrics_source=item.get("lyrics_source", HYMN_BASE_URL),
                pdf_page=item.get("pdf_page"),
            )
        )
    return hymns
