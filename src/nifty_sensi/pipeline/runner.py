"""Stage orchestration with content-addressed caching and a run manifest."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import Settings
from ..logging_utils import configure, get_logger
from ..paths import (
    atomic_write_json,
    atomic_write_parquet,
    is_cached,
    mark_cached,
    new_run_id,
    read_json,
    read_table,
    sha256_file,
    sha256_json,
)
from .greeks import attach_greeks, greeks_report
from .implied_vol import compute_iv, iv_report
from .prepare import prepare_options, prepare_report
from .rate import attach_rates
from .surface import build_surfaces

log = get_logger("runner")

STAGES = ("prepare", "rate", "iv", "greeks", "surface")


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    paths: dict[str, Path] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def greeks(self) -> pd.DataFrame:
        return self.frames.get("greeks", pd.DataFrame())

    @property
    def surfaces(self) -> pd.DataFrame:
        return self.frames.get("surface", pd.DataFrame())

    def summary(self) -> pd.DataFrame:
        rows = []
        for stage in self.manifest.get("stages", []):
            rows.append(
                {
                    "stage": stage["name"],
                    "status": stage["status"],
                    "rows": stage.get("rows"),
                    "seconds": round(stage.get("seconds", 0.0), 2),
                }
            )
        return pd.DataFrame(rows)


def _hash_inputs(paths: list[Path]) -> dict[str, str]:
    return {str(p): sha256_file(p) for p in paths if Path(p).exists()}


def _stage(
    name: str,
    output: Path,
    inputs: list[Path],
    settings: Settings,
    force: bool,
    compute: Callable[[], pd.DataFrame],
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    key = sha256_json(
        {
            "stage": name,
            "config": settings.fingerprint(),
            "inputs": _hash_inputs(inputs),
        }
    )
    if not force and is_cached(output, key):
        frame = pd.read_parquet(output)
        manifest["stages"].append({"name": name, "status": "cached", "rows": len(frame), "seconds": 0.0})
        log.info("%s: cached (%d rows)", name, len(frame))
        return frame, True

    started = time.perf_counter()
    frame = compute()
    atomic_write_parquet(frame, output)
    mark_cached(output, key, {"rows": len(frame), "stage": name})
    manifest["stages"].append(
        {
            "name": name,
            "status": "computed",
            "rows": len(frame),
            "seconds": time.perf_counter() - started,
            "output": str(output),
        }
    )
    log.info("%s: %d rows in %.2fs", name, len(frame), time.perf_counter() - started)
    return frame, False


def run_pipeline(
    settings: Settings,
    stages: list[str] | None = None,
    force: bool = False,
) -> RunResult:
    """Run the requested stages (and their prerequisites) end to end."""
    unknown = set(stages or []) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    settings.ensure_dirs()
    run_id = new_run_id()
    run_dir = settings.artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure(log_file=run_dir / "run.log")

    requested = list(stages or STAGES)
    last = max(STAGES.index(name) for name in requested)
    ordered = list(STAGES[: last + 1])

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "config_fingerprint": settings.fingerprint(),
        "config": settings.to_dict(),
        "stages": [],
    }

    if not settings.options_file.exists():
        raise FileNotFoundError(f"option feed not found: {settings.options_file}")
    if not settings.spot_file.exists():
        raise FileNotFoundError(f"spot feed not found: {settings.spot_file}")

    manifest["inputs"] = {
        "options": {"path": str(settings.options_file), "sha256": sha256_file(settings.options_file)},
        "spot": {"path": str(settings.spot_file), "sha256": sha256_file(settings.spot_file)},
    }

    paths = {
        "prepared": settings.interim_dir / "prepared.parquet",
        "rates": settings.interim_dir / "rates.parquet",
        "rated": settings.interim_dir / "rated.parquet",
        "iv": settings.interim_dir / "iv.parquet",
        "greeks": settings.processed_dir / "greeks.parquet",
        "surface": settings.processed_dir / "surfaces.parquet",
        "surface_qc": settings.processed_dir / "surface_qc.parquet",
        "iv_qc": settings.processed_dir / "iv_qc.parquet",
        "greek_qc": settings.processed_dir / "greek_qc.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}

    raw_options = read_table(settings.options_file)
    spot = read_table(settings.spot_file)

    # prepare -------------------------------------------------------------
    if "prepare" in ordered:
        frames["prepared"], _ = _stage(
            "prepare",
            paths["prepared"],
            [settings.options_file, settings.spot_file],
            settings,
            force,
            lambda: prepare_options(raw_options, spot, settings),
            manifest,
        )
        manifest["qc"] = {"prepare": prepare_report(raw_options, frames["prepared"])}

    # rate ----------------------------------------------------------------
    if "rate" in ordered:
        prepared = frames["prepared"] if "prepared" in frames else pd.read_parquet(paths["prepared"])

        def _rate():
            rated, rates = attach_rates(prepared, settings)
            atomic_write_parquet(rates, paths["rates"])
            return rated

        frames["rated"], _ = _stage(
            "rate", paths["rated"], [paths["prepared"]], settings, force, _rate, manifest
        )
        if paths["rates"].exists():
            rates = pd.read_parquet(paths["rates"])
            manifest.setdefault("qc", {})["rate_sources"] = (
                rates["rate_source"].value_counts().to_dict()
            )

    # iv ------------------------------------------------------------------
    if "iv" in ordered:
        rated = frames["rated"] if "rated" in frames else pd.read_parquet(paths["rated"])
        frames["iv"], _ = _stage(
            "iv", paths["iv"], [paths["rated"]], settings, force, lambda: compute_iv(rated, settings), manifest
        )
        qc = iv_report(frames["iv"])
        atomic_write_parquet(qc, paths["iv_qc"])
        manifest.setdefault("qc", {})["iv"] = {
            "usable": int(frames["iv"]["valid_iv"].sum()),
            "statuses": frames["iv"]["iv_status_label"].value_counts().to_dict(),
        }

    # greeks --------------------------------------------------------------
    if "greeks" in ordered:
        iv_frame = frames["iv"] if "iv" in frames else pd.read_parquet(paths["iv"])
        frames["greeks"], _ = _stage(
            "greeks",
            paths["greeks"],
            [paths["iv"]],
            settings,
            force,
            lambda: attach_greeks(iv_frame, settings),
            manifest,
        )
        qc = greeks_report(frames["greeks"])
        atomic_write_parquet(qc, paths["greek_qc"])
        manifest.setdefault("qc", {})["greeks"] = {"computed": int(frames["greeks"]["delta"].notna().sum())}

    # surface -------------------------------------------------------------
    if "surface" in ordered:
        greeks_frame = frames["greeks"] if "greeks" in frames else pd.read_parquet(paths["greeks"])
        surfaces: dict[str, pd.DataFrame] = {}

        def _surface():
            long_df, qc = build_surfaces(greeks_frame, settings)
            surfaces["qc"] = qc
            atomic_write_parquet(qc, paths["surface_qc"])
            return long_df

        frames["surface"], _ = _stage(
            "surface", paths["surface"], [paths["greeks"]], settings, force, _surface, manifest
        )
        manifest.setdefault("qc", {})["surface"] = {
            "dates": int(frames["surface"]["date"].nunique()),
            "coverage_pct": float(100 * frames["surface"]["iv"].notna().mean()),
        }

    manifest["finished_at"] = pd.Timestamp.utcnow().isoformat()
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(settings.artifacts_dir / "latest.json", {"run_id": run_id, "run_dir": str(run_dir)})

    return RunResult(run_id=run_id, run_dir=run_dir, manifest=manifest, paths=paths, frames=frames)


def load_run(settings: Settings, run_id: str = "latest") -> dict[str, Any]:
    """Load a run manifest by id (or ``latest``)."""
    if run_id == "latest":
        pointer = settings.artifacts_dir / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError("no runs found; call run_pipeline first")
        run_id = read_json(pointer)["run_id"]
    path = settings.artifacts_dir / "runs" / run_id / "manifest.json"
    return read_json(path)
