"""Load project configuration from environment and .env file."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load variables from .env in the project root if present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(project_root() / ".env")
