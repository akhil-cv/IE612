"""Command line interface: ``nifty-sensi`` (or ``python -m nifty_sensi``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_settings
from .logging_utils import configure, get_logger
from .pipeline.runner import STAGES, load_run, run_pipeline

log = get_logger("cli")


def _settings(args):
    overrides: dict = {}
    if getattr(args, "start", None) or getattr(args, "end", None):
        overrides["data"] = {}
        if args.start:
            overrides["data"]["start"] = args.start
        if args.end:
            overrides["data"]["end"] = args.end
    root = Path(args.root) if getattr(args, "root", None) else None
    return load_settings(args.config, overrides=overrides or None, root=root)


def _cmd_ingest(args) -> int:
    settings = _settings(args)
    settings.ensure_dirs()

    if args.source in {"spot", "all"}:
        from .ingest_spot import download_spot, save_spot

        start = args.start or "2025-01-01"
        end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        frame = download_spot(start, end)
        path = save_spot(frame, args.out_spot or settings.raw_dir / "nifty_close.parquet")
        print(f"spot: {len(frame)} rows -> {path}")

    if args.source in {"options", "all"}:
        from .ingest_options import fetch_options, save_options

        start = args.start or "2025-01-01"
        end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        frame = fetch_options(start, end, revision=args.revision)
        path = save_options(frame, args.out_options or settings.raw_dir / "options.parquet")
        print(f"options: {len(frame)} rows -> {path}")

    return 0


def _cmd_build(args) -> int:
    settings = _settings(args)
    stages_arg = getattr(args, "stages", None)
    stages = [stage.strip() for stage in stages_arg.split(",") if stage.strip()] if stages_arg else None
    result = run_pipeline(settings, stages=stages, force=args.force)
    print(result.summary().to_string(index=False))
    print(f"run: {result.run_id} -> {result.run_dir}")
    return 0


def _cmd_report(args) -> int:
    from .reporting import build_run_report

    settings = _settings(args)
    path = build_run_report(settings, run_id=args.run_id, output=args.out)
    print(f"report: {path}")
    return 0


def _cmd_qc(args) -> int:
    from .qc import anomaly_report, dataset_summary

    settings = _settings(args)
    manifest = load_run(settings, args.run_id)
    print(f"run {manifest['run_id']} ({manifest.get('created_at', 'unknown')})")
    path = settings.processed_dir / "greeks.parquet"
    if not path.exists():
        print("no processed Greeks found; run 'build' first")
        return 0
    frame = pd.read_parquet(path)
    for key, value in dataset_summary(frame).items():
        print(f"  {key:22s} {value}")
    print()
    print(anomaly_report(frame).to_string(index=False))
    return 0


def _cmd_parity(args) -> int:
    from .paths import atomic_write_parquet
    from .qc import parity_by_tenor, parity_report, rate_parity, write_parity_markdown

    settings = _settings(args)
    legacy = pd.read_parquet(args.legacy)
    current_path = Path(args.new) if args.new else settings.processed_dir / "greeks.parquet"
    if not current_path.exists():
        print(f"missing new output: {current_path}; run 'build' first")
        return 1
    current = pd.read_parquet(current_path)

    summary, matched = parity_report(
        legacy, current, tolerance=args.tolerance, convention=args.convention
    )
    tenors = parity_by_tenor(matched)
    rates = rate_parity(legacy, current)

    print(f"convention: {summary['convention']}")
    for key, value in summary.items():
        if key == "convention":
            continue
        print(f"  {key:24s} {value}")
    if not tenors.empty:
        print()
        print(tenors.to_string(index=False))
    if not rates.empty:
        print()
        print(
            f"rate dates {len(rates)} | median legacy {rates['legacy_rate'].median():.2%} "
            f"| median new {rates['new_rate'].median():.2%} "
            f"| correlation {rates['legacy_rate'].corr(rates['new_rate']):.3f}"
        )

    out = Path(args.out) if args.out else settings.artifacts_dir / "parity_report.md"
    print(f"report: {write_parity_markdown(out, summary, tenors, rates)}")
    if args.details:
        print(f"details: {atomic_write_parquet(matched, args.details)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nifty-sensi", description="NIFTY option IV and Greeks pipeline")
    parser.add_argument("--config", default=None, help="path to JSON/YAML settings")
    parser.add_argument(
        "--root",
        default=None,
        help="project root used to resolve relative data paths (default: repo root)",
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download raw spot/option data")
    ingest.add_argument("--source", choices=["spot", "options", "all"], default="all")
    ingest.add_argument("--start")
    ingest.add_argument("--end")
    ingest.add_argument("--revision", help="pin a HuggingFace dataset revision")
    ingest.add_argument("--out-spot")
    ingest.add_argument("--out-options")
    ingest.set_defaults(func=_cmd_ingest)

    build = sub.add_parser("build", help="run pipeline stages")
    build.add_argument("--stages", help=f"comma separated subset of {STAGES}")
    build.add_argument("--force", action="store_true", help="ignore stage caches")
    build.add_argument("--start")
    build.add_argument("--end")
    build.set_defaults(func=_cmd_build)

    run_all = sub.add_parser("run-all", help="ingest checks + full pipeline")
    run_all.add_argument("--force", action="store_true")
    run_all.add_argument("--start")
    run_all.add_argument("--end")
    run_all.set_defaults(func=_cmd_build, stages=None)

    report = sub.add_parser("report", help="build the HTML report for a run")
    report.add_argument("--run-id", default="latest")
    report.add_argument("--out")
    report.set_defaults(func=_cmd_report)

    qc = sub.add_parser("qc", help="print dataset quality checks")
    qc.add_argument("--run-id", default="latest")
    qc.set_defaults(func=_cmd_qc)

    parity = sub.add_parser("parity", help="compare legacy and new IV outputs")
    parity.add_argument("--legacy", required=True)
    parity.add_argument("--new")
    parity.add_argument("--tolerance", type=float, default=1e-3)
    parity.add_argument(
        "--convention",
        choices=["pipeline", "legacy_zero_rate"],
        default="pipeline",
        help=(
            "pipeline: compare shipped IVs (forward from the parity rate); "
            "legacy_zero_rate: re-solve with F=S and r=0, the legacy convention"
        ),
    )
    parity.add_argument("--out", help="markdown report path (default: artifacts/parity_report.md)")
    parity.add_argument("--details", help="optional parquet path for per-contract differences")
    parity.set_defaults(func=_cmd_parity)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(verbose=args.verbose)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
