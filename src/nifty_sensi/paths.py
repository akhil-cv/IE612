"""Filesystem helpers: repo discovery, atomic writes, hashing, run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root(start: Path | None = None) -> Path:
    """Walk upwards until a ``pyproject.toml`` is found."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_write_parquet(frame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".parquet")
    os.close(fd)
    try:
        frame.to_parquet(tmp, index=index)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_table(path: str | Path):
    """Read a CSV or Parquet file based on its suffix."""
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    return atomic_write_json(path, manifest)


def stage_meta_path(output: str | Path) -> Path:
    output = Path(output)
    return output.with_suffix(output.suffix + ".meta.json")


def is_cached(output: str | Path, key: str) -> bool:
    """True when *output* exists and its sidecar meta matches *key*."""
    output = Path(output)
    meta = stage_meta_path(output)
    if not output.exists() or not meta.exists():
        return False
    try:
        return read_json(meta).get("key") == key
    except (json.JSONDecodeError, OSError):
        return False


def mark_cached(output: str | Path, key: str, extra: dict[str, Any] | None = None) -> None:
    payload = {"key": key, "output": str(output)}
    payload.update(extra or {})
    atomic_write_json(stage_meta_path(output), payload)
