"""NSE NIFTY 50 spot downloader with retries and a CLI-friendly API."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("nifty_sensi.spot")

URL = "https://www.nseindia.com/api/historicalOR/indicesHistory"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
}
MAX_DAYS_PER_REQUEST = 60


class NSEClient:
    def __init__(self, index_type: str = "NIFTY 50", retries: int = 3):
        self.index_type = index_type
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._init_session()

    def _init_session(self) -> None:
        self._get("https://www.nseindia.com")

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                wait = 2**attempt
                log.warning("request failed (%s/%s): %s; retrying in %ss", attempt, self.retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"NSE request failed after {self.retries} attempts: {last_error}")

    def fetch(self, start_date, end_date) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if start > end:
            raise ValueError("start_date must be before end_date")

        records = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + pd.Timedelta(days=MAX_DAYS_PER_REQUEST - 1), end)
            payload = self._get(
                URL,
                params={
                    "indexType": self.index_type,
                    "from": cursor.strftime("%d-%m-%Y"),
                    "to": chunk_end.strftime("%d-%m-%Y"),
                },
            ).json()
            if "data" not in payload:
                raise RuntimeError(f"unexpected NSE response: {payload}")
            records.extend(payload["data"])
            log.info("spot %s -> %s: %d records", cursor.date(), chunk_end.date(), len(payload["data"]))
            cursor = chunk_end + pd.Timedelta(days=1)
            time.sleep(0.5)
        return _clean(records)


def _clean(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    columns = {
        "EOD_TIMESTAMP": "date",
        "EOD_OPEN_INDEX_VAL": "open",
        "EOD_HIGH_INDEX_VAL": "high",
        "EOD_LOW_INDEX_VAL": "low",
        "EOD_CLOSE_INDEX_VAL": "close",
    }
    missing = [key for key in columns if key not in frame.columns]
    if missing:
        raise RuntimeError(f"unexpected NSE payload columns, missing {missing}")
    frame = frame[list(columns)].rename(columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], format="%d-%b-%Y", errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["date", "close"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def download_spot(start: str, end: str, index_type: str = "NIFTY 50") -> pd.DataFrame:
    return NSEClient(index_type).fetch(start, end)


def save_spot(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path
