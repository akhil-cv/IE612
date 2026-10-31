"""Dataset quality summaries and legacy-versus-new parity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def dataset_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"rows": 0}
    summary = {
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "expiries": int(frame["expiry"].nunique()),
        "strikes": int(frame["strike"].nunique()),
        "calls": int((frame["option_type"] == "CE").sum()),
        "puts": int((frame["option_type"] == "PE").sum()),
    }
    if "iv" in frame:
        summary["usable_iv"] = int(frame["valid_iv"].sum()) if "valid_iv" in frame else int(frame["iv"].notna().sum())
        summary["median_iv"] = float(frame["iv"].median())
    if "r" in frame:
        summary["rate_missing_dates"] = int(frame.groupby("date")["r"].first().isna().sum())
        summary["median_rate"] = float(frame["r"].median())
    if "days_to_exp" in frame:
        summary["max_days"] = int(frame["days_to_exp"].max())
    return summary


def rejection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "iv_status_label" not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby("iv_status_label", as_index=False).agg(
        contracts=("iv", "size"),
        median_price=("price", "median"),
    )
    grouped["share_pct"] = 100.0 * grouped["contracts"] / grouped["contracts"].sum()
    return grouped.sort_values("contracts", ascending=False)


def anomaly_report(frame: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("contracts over 365 days", int((frame["days_to_exp"] > 365).sum()) if "days_to_exp" in frame else 0),
        ("expiry on or before date", int((frame["days_to_exp"] <= 0).sum()) if "days_to_exp" in frame else 0),
        ("missing rate", int(frame["r"].isna().sum()) if "r" in frame else 0),
        ("negative or zero price", int((frame["price"] <= 0).sum()) if "price" in frame else 0),
        ("duplicate contracts", int(frame.duplicated(["date", "expiry", "strike", "option_type"]).sum())),
    ]
    if "iv_status_label" in frame:
        checks.append(("IV at solver bounds", int(frame["iv_status_label"].isin(["at_lower_bound", "at_upper_bound"]).sum())))
    return pd.DataFrame(checks, columns=["check", "count"])


def parity_report(
    legacy: pd.DataFrame,
    current: pd.DataFrame,
    tolerance: float = 1e-3,
    convention: str = "pipeline",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compare the legacy ``iv`` with the new engine.

    ``convention="pipeline"`` compares the shipped new IVs (forward built from the
    parity rate). ``convention="legacy_zero_rate"`` re-solves the new engine with
    ``F = S`` and ``r = 0`` -- the convention the legacy notebook actually used --
    which isolates solver/lookup differences from the deliberate carry fix.
    """
    from .models import black76

    key = ["date", "expiry", "strike", "option_type"]
    left = legacy[key + ["iv"]].rename(columns={"iv": "legacy_pipeline_iv"})
    if "implied_volatility" in legacy.columns:
        left = left.join(legacy["implied_volatility"])
    left = left.drop_duplicates(key)

    if convention == "legacy_zero_rate":
        needed = list(dict.fromkeys(key + ["spot", "strike", "ttm", "price"]))
        source = current[needed].drop_duplicates(key)
        flag = np.where(source["option_type"].astype(str).str.upper().eq("CE"), "c", "p")
        values, _ = black76.implied_vol(
            source["price"].to_numpy(dtype=float),
            source["spot"].to_numpy(dtype=float),
            source["strike"].to_numpy(dtype=float),
            source["ttm"].to_numpy(dtype=float),
            np.zeros(len(source)),
            flag,
        )
        right = source[key].assign(new_iv=values)
    elif convention == "pipeline":
        right = current[key + ["iv"]].rename(columns={"iv": "new_iv"})
    else:
        raise ValueError(f"unknown convention: {convention}")
    right = right.drop_duplicates(key)

    if "days_to_exp" in current.columns:
        right = right.merge(
            current[key + ["days_to_exp"]].drop_duplicates(key), on=key, how="left"
        )

    merged = left.merge(right, on=key, how="outer", indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()
    matched["diff"] = matched["new_iv"] - matched["legacy_pipeline_iv"]
    matched["abs_diff"] = matched["diff"].abs()
    matched["explained"] = matched["abs_diff"] <= tolerance
    matched["convention"] = convention

    summary = {
        "convention": convention,
        "legacy_rows": int(len(left)),
        "new_rows": int(len(right)),
        "matched": int(len(matched)),
        "legacy_only": int((merged["_merge"] == "left_only").sum()),
        "new_only": int((merged["_merge"] == "right_only").sum()),
        "median_abs_diff": float(matched["abs_diff"].median()) if len(matched) else np.nan,
        "p95_abs_diff": float(matched["abs_diff"].quantile(0.95)) if len(matched) else np.nan,
        "max_abs_diff": float(matched["abs_diff"].max()) if len(matched) else np.nan,
        "within_tolerance_pct": float(100 * matched["explained"].mean()) if len(matched) else np.nan,
    }
    return summary, matched


def parity_by_tenor(matched: pd.DataFrame) -> pd.DataFrame:
    """IV differences bucketed by days to expiry."""
    if matched.empty or "days_to_exp" not in matched.columns:
        return pd.DataFrame()
    frame = matched.copy()
    frame["bucket"] = pd.cut(
        frame["days_to_exp"],
        [-1, 7, 30, 90, 180, 1_000_000],
        labels=["<=7", "8-30", "31-90", "91-180", ">180"],
    )
    grouped = frame.groupby("bucket", observed=True).agg(
        contracts=("abs_diff", "size"),
        median_abs_diff=("abs_diff", "median"),
        p95_abs_diff=("abs_diff", lambda series: float(series.quantile(0.95))),
        within_tolerance_pct=("explained", lambda series: 100.0 * float(series.mean())),
    )
    return grouped.reset_index()


def rate_parity(legacy: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Per-date legacy versus new parity rate."""
    if "r" not in legacy.columns or "r" not in current.columns:
        return pd.DataFrame()
    left = legacy.groupby("date")["r"].first().rename("legacy_rate")
    right = current.groupby("date")["r"].first().rename("new_rate")
    table = pd.concat([left, right], axis=1).dropna()
    table["difference"] = table["new_rate"] - table["legacy_rate"]
    return table.reset_index()


def write_parity_markdown(
    path,
    summary: dict[str, float],
    tenor_table: pd.DataFrame,
    rate_table: pd.DataFrame,
) -> Path:
    """Write a small, self-contained Markdown parity report."""
    from .paths import atomic_write_text

    lines = [
        "# Legacy versus new IV parity",
        "",
        f"Convention: `{summary.get('convention', 'pipeline')}`",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        if key == "convention":
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:,.6g} |")
        else:
            lines.append(f"| {key} | {value:,} |" if isinstance(value, int) else f"| {key} | {value} |")

    if not tenor_table.empty:
        lines += ["", "## By maturity", ""]
        columns = list(tenor_table.columns)
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for _, row in tenor_table.iterrows():
            cells = [
                f"{value:,.6g}" if isinstance(value, float) else str(value) for value in row
            ]
            lines.append("| " + " | ".join(cells) + " |")

    if not rate_table.empty:
        lines += [
            "",
            "## Risk-free rate",
            "",
            f"Dates compared: {len(rate_table)}",
            f"Median legacy rate: {rate_table['legacy_rate'].median():.4%}",
            f"Median new rate: {rate_table['new_rate'].median():.4%}",
            f"Median difference: {rate_table['difference'].median():.4%}",
            f"Correlation: {rate_table['legacy_rate'].corr(rate_table['new_rate']):.4f}",
        ]
    return atomic_write_text(path, "\n".join(lines) + "\n")
