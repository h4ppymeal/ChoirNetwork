"""Build the checked-in World English Bible verse corpus from official USFM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = "https://ebible.org/Scriptures/eng-web_usfm.zip"
SOURCE_SHA256 = "2403c879aa6b0c9e5e43a4db6f604f7dd1a1f8f32b959c08c8a1fc32c4833e00"
DEFAULT_OUTPUT = Path("data/public/world_english_bible.json")

BOOKS = {
    "GEN": "Genesis",
    "EXO": "Exodus",
    "LEV": "Leviticus",
    "NUM": "Numbers",
    "DEU": "Deuteronomy",
    "JOS": "Joshua",
    "JDG": "Judges",
    "RUT": "Ruth",
    "1SA": "1 Samuel",
    "2SA": "2 Samuel",
    "1KI": "1 Kings",
    "2KI": "2 Kings",
    "1CH": "1 Chronicles",
    "2CH": "2 Chronicles",
    "EZR": "Ezra",
    "NEH": "Nehemiah",
    "EST": "Esther",
    "JOB": "Job",
    "PSA": "Psalms",
    "PRO": "Proverbs",
    "ECC": "Ecclesiastes",
    "SNG": "Song of Solomon",
    "ISA": "Isaiah",
    "JER": "Jeremiah",
    "LAM": "Lamentations",
    "EZK": "Ezekiel",
    "DAN": "Daniel",
    "HOS": "Hosea",
    "JOL": "Joel",
    "AMO": "Amos",
    "OBA": "Obadiah",
    "JON": "Jonah",
    "MIC": "Micah",
    "NAM": "Nahum",
    "HAB": "Habakkuk",
    "ZEP": "Zephaniah",
    "HAG": "Haggai",
    "ZEC": "Zechariah",
    "MAL": "Malachi",
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
    "JHN": "John",
    "ACT": "Acts",
    "ROM": "Romans",
    "1CO": "1 Corinthians",
    "2CO": "2 Corinthians",
    "GAL": "Galatians",
    "EPH": "Ephesians",
    "PHP": "Philippians",
    "COL": "Colossians",
    "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians",
    "1TI": "1 Timothy",
    "2TI": "2 Timothy",
    "TIT": "Titus",
    "PHM": "Philemon",
    "HEB": "Hebrews",
    "JAS": "James",
    "1PE": "1 Peter",
    "2PE": "2 Peter",
    "1JN": "1 John",
    "2JN": "2 John",
    "3JN": "3 John",
    "JUD": "Jude",
    "REV": "Revelation",
}

_NOTE_RE = re.compile(r"\\(?:f|x) .*?\\(?:f|x)\*", re.DOTALL)
_WORD_RE = re.compile(r"\\\+?w ([^|\\]+)(?:\|[^\\]*)?\\\+?w\*")
_CHARACTER_MARKER_RE = re.compile(r"\\\+?[a-z][a-z0-9-]*\*?")
_ATTRIBUTE_RE = re.compile(r"\|[a-z]+=\"[^\"]*\"")
_CONTINUATION_RE = re.compile(r"^\\(?:p|m|q\d*|li\d*|nb)\s*")


def _clean_usfm(text: str) -> str:
    text = _NOTE_RE.sub("", text)
    text = _WORD_RE.sub(r"\1", text)
    text = _ATTRIBUTE_RE.sub("", text)
    text = _CHARACTER_MARKER_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_book(raw: str, book: str) -> list[dict]:
    verses: list[dict] = []
    chapter: int | None = None
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current["text"] = _clean_usfm(current["text"])
            if current["text"]:
                verses.append(current)
        current = None

    for line in raw.splitlines():
        chapter_match = re.match(r"^\\c\s+(\d+)", line)
        if chapter_match:
            flush()
            chapter = int(chapter_match.group(1))
            continue

        verse_match = re.match(r"^\\v\s+(\d+[a-z]?)\s*(.*)", line)
        if verse_match and chapter is not None:
            flush()
            current = {
                "book": book,
                "chapter": chapter,
                "verse": verse_match.group(1),
                "text": verse_match.group(2),
            }
            continue

        if current is not None and _CONTINUATION_RE.match(line):
            current["text"] += " " + _CONTINUATION_RE.sub("", line)

    flush()
    return verses


def build_corpus(archive: Path, output: Path) -> int:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected source checksum {digest}; expected {SOURCE_SHA256}"
        )

    verses: list[dict] = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        for code, book in BOOKS.items():
            suffix = f"-{code}eng-web.usfm"
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected one USFM file for {book}, found {matches}")
            raw = source.read(matches[0]).decode("utf-8-sig")
            verses.extend(_parse_book(raw, book))

    payload = {
        "translation": "World English Bible (WEB)",
        "license": "Public Domain",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "verse_count": len(verses),
        "verses": verses,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return len(verses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.archive:
        count = build_corpus(args.archive, args.output)
    else:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "eng-web_usfm.zip"
            urllib.request.urlretrieve(SOURCE_URL, archive)
            count = build_corpus(archive, args.output)
    print(f"Wrote {count} WEB verses to {args.output}")


if __name__ == "__main__":
    main()
