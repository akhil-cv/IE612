"""Option-chain ingestion from the HuggingFace NSE dataset (or a local file)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("nifty_sensi.options")

REPO_ID = "rissin/nse-options-intraday"
FILES = (
    "historical_daily/NIFTY/NIFTY_2025.parquet",
    "historical_daily/NIFTY/NIFTY_2026.parquet",
)
COLUMNS = ("date", "timestamp", "expiry", "strike", "option_type", "close")


def _require_ingest_dependencies():
    try:
        import duckdb  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "option ingest needs duckdb and huggingface_hub: "
            "pip install 'nifty-sensi[ingest]'"
        ) from exc


def download_raw_files(revision: str | None = None) -> list[str]:
    _require_ingest_dependencies()
    from huggingface_hub import hf_hub_download

    paths = []
    for filename in FILES:
        log.info("downloading %s", filename)
        paths.append(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=revision,
            )
        )
    return paths


def fetch_options(
    start: str,
    end: str,
    revision: str | None = None,
    local_files: list[str] | None = None,
) -> pd.DataFrame:
    """Pull the raw option chain between *start* and *end*."""
    _require_ingest_dependencies()
    import duckdb

    files = local_files or download_raw_files(revision=revision)
    columns = ", ".join(COLUMNS)
    query = f"""
        SELECT {columns}
        FROM read_parquet(?)
        WHERE date BETWEEN ? AND ?
    """
    frame = duckdb.execute(query, [files, start, end]).df()
    frame["option_type"] = frame["option_type"].astype(str).str.upper()
    log.info("options: %d rows, %d dates", len(frame), frame["date"].nunique())
    return frame


def save_options(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path
