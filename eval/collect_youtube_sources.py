"""Attach reproducible YouTube provenance to the observed-service dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

CHANNEL_STREAMS_URL = "https://www.youtube.com/@TrueJesusChurchToronto/streams"
DEFAULT_DATASET = Path("eval/datasets/service_hymns.csv")
SOURCE_FIELDS = [
    "video_id",
    "video_url",
    "service_date",
    "source_hymn_1_slug",
    "source_hymn_2_slug",
    "source_verification",
]


def normalize_title(value: str) -> str:
    """Normalize punctuation, Unicode, translations, and capitalization."""
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def title_similarity(query: str, video_title: str) -> float:
    query_normalized = normalize_title(query)
    title_normalized = normalize_title(video_title)
    title_normalized = re.sub(
        r"^(?:\d{4} \d{2} \d{2}|\d{2} \d{2} \d{4})\s*",
        "",
        title_normalized,
    )
    return SequenceMatcher(None, query_normalized, title_normalized).ratio()


def load_streams(channel_url: str) -> list[dict]:
    """Read the channel's public stream inventory without downloading media."""
    with tempfile.NamedTemporaryFile(suffix=".json") as output:
        subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-single-json",
                channel_url,
            ],
            check=True,
            stdout=output,
        )
        output.seek(0)
        return json.load(output).get("entries", [])


def match_video(query: str, videos: list[dict]) -> tuple[dict, float]:
    """Choose the closest title, preferring the 2026 service when duplicated."""
    ranked = sorted(
        videos,
        key=lambda video: (
            title_similarity(query, video.get("title", "")),
            "2026" in str(video.get("title", "")),
        ),
        reverse=True,
    )
    match = ranked[0]
    return match, title_similarity(query, match.get("title", ""))


def fetch_description(video_id: str) -> str:
    """Extract the public description embedded in a YouTube watch page."""
    request = Request(
        f"https://www.youtube.com/watch?v={video_id}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8")

    marker = '"shortDescription":'
    start = page.find(marker)
    if start == -1:
        return ""
    description, _ = json.JSONDecoder().raw_decode(page[start + len(marker) :])
    return description


def description_hymns(description: str) -> tuple[str, str] | None:
    match = re.search(r"(?i)\bhymns?\s*:\s*([^\r\n]+)", description)
    if not match:
        return None
    slugs = re.findall(r"\b\d{1,3}[A-Za-z]?\b", match.group(1))
    if len(slugs) < 2:
        return None
    return slugs[0].upper(), slugs[1].upper()


def service_date(title: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", title)
    return match.group(1) if match else ""


def enrich_dataset(dataset_path: Path, videos: list[dict]) -> dict[str, int]:
    with dataset_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for field in SOURCE_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    counts = {"verified": 0, "mismatch": 0, "unavailable": 0}
    for row in rows:
        video, confidence = match_video(row["query"], videos)
        if confidence < 0.55:
            raise ValueError(
                f"No confident video match for {row['query']!r} "
                f"(best score: {confidence:.3f})"
            )

        video_id = video["id"]
        row["video_id"] = video_id
        row["video_url"] = f"https://www.youtube.com/watch?v={video_id}"
        row["service_date"] = service_date(video.get("title", ""))

        observed = description_hymns(fetch_description(video_id))
        if observed is None:
            row["source_hymn_1_slug"] = ""
            row["source_hymn_2_slug"] = ""
            row["source_verification"] = "description_unavailable"
            counts["unavailable"] += 1
        else:
            row["source_hymn_1_slug"], row["source_hymn_2_slug"] = observed
            expected = (row["hymn_1_slug"].upper(), row["hymn_2_slug"].upper())
            status = "verified" if observed == expected else "mismatch"
            row["source_verification"] = status
            counts[status] += 1
        time.sleep(0.1)

    temporary_path = dataset_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(dataset_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--channel", default=CHANNEL_STREAMS_URL)
    args = parser.parse_args()

    counts = enrich_dataset(args.dataset, load_streams(args.channel))
    print(
        f"Verified: {counts['verified']}; mismatches: {counts['mismatch']}; "
        f"descriptions unavailable: {counts['unavailable']}"
    )


if __name__ == "__main__":
    main()
