"""Typed settings loaded from ``config/default.json`` (YAML optional)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

from .paths import repo_root


@dataclass
class DataSettings:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    spot_file: str = "data/nifty_close.csv"
    options_file: str = "data/niftyoptions_df.csv"
    start: str | None = None
    end: str | None = None


@dataclass
class OptionSettings:
    price_column: str = "close"
    otm_only: bool = False
    min_price: float = 0.05
    min_days: int = 1
    max_days: int = 365
    day_count: float = 365.0
    dedupe: bool = True


@dataclass
class IVSettings:
    lower: float = 1e-4
    upper: float = 5.0
    tolerance: float = 1e-8
    max_iter: int = 60


@dataclass
class RateSettings:
    min: float = -0.05
    max: float = 0.30
    fallback: float = 0.065
    rolling_window: int = 5


@dataclass
class GridSettings:
    moneyness: list[float] = field(default_factory=lambda: [0.6 + i * 0.8 / 9 for i in range(10)])
    tau_days: list[int] = field(
        default_factory=lambda: [7, 14, 30, 60, 90, 180, 270, 365]
    )
    min_points: int = 4


@dataclass
class GreekSettings:
    theta_per_day: bool = True
    vega_per_point: bool = True
    rho_per_point: bool = True


@dataclass
class SyntheticSettings:
    spot_pad: float = 6000
    strike_step: float = 300
    strike_round: float = 50
    rate_offsets: list[float] = field(default_factory=lambda: [-0.01, 0.0, 0.01])


@dataclass
class ReportSettings:
    title: str = "NIFTY 50 Option Sensitivity Report"
    dpi: int = 110
    max_dates: int = 4


@dataclass
class Settings:
    data: DataSettings = field(default_factory=DataSettings)
    options: OptionSettings = field(default_factory=OptionSettings)
    iv: IVSettings = field(default_factory=IVSettings)
    rate: RateSettings = field(default_factory=RateSettings)
    grid: GridSettings = field(default_factory=GridSettings)
    greeks: GreekSettings = field(default_factory=GreekSettings)
    synthetic: SyntheticSettings = field(default_factory=SyntheticSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    root: Path = field(default_factory=repo_root, repr=False, compare=False)

    # ------------------------------------------------------------------ paths
    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def raw_dir(self) -> Path:
        return self.path(self.data.raw_dir)

    @property
    def interim_dir(self) -> Path:
        return self.path(self.data.interim_dir)

    @property
    def processed_dir(self) -> Path:
        return self.path(self.data.processed_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.path(self.data.artifacts_dir)

    @property
    def spot_file(self) -> Path:
        return self.path(self.data.spot_file)

    @property
    def options_file(self) -> Path:
        return self.path(self.data.options_file)

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: dataclasses.asdict(getattr(self, f.name))
            for f in fields(self)
            if f.name != "root"
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path | None = None) -> Settings:
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "root" or f.name not in data:
                continue
            section = hints.get(f.name)
            if isinstance(section, type) and dataclasses.is_dataclass(section):
                allowed = {sf.name for sf in fields(section)}
                kwargs[f.name] = section(
                    **{k: v for k, v in (data[f.name] or {}).items() if k in allowed}
                )
            else:
                kwargs[f.name] = data[f.name]
        if root is not None:
            kwargs["root"] = Path(root)
        return cls(**kwargs)


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to read YAML configs") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Settings:
    """Load settings from *path* and apply nested *overrides*."""
    root = Path(root) if root is not None else repo_root()
    path = Path(path) if path is not None else root / "config" / "default.json"
    if not path.is_absolute():
        path = root / path
    data = _read_config(path)
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.from_dict(data, root=root)
