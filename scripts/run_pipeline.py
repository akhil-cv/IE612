"""Run the full NIFTY sensitivity pipeline (and optionally the HTML report).

This is a thin convenience wrapper for users who have cloned the repository but
have not installed the package::

    python scripts/run_pipeline.py --start 2025-01-01 --end 2025-01-31 --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nifty_sensi.config import load_settings  # noqa: E402
from nifty_sensi.logging_utils import configure  # noqa: E402
from nifty_sensi.pipeline.runner import STAGES, run_pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="first observation date (YYYY-MM-DD)")
    parser.add_argument("--end", help="last observation date (YYYY-MM-DD)")
    parser.add_argument(
        "--stages",
        default=None,
        help=f"comma separated subset of {', '.join(STAGES)} (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="ignore stage caches")
    parser.add_argument("--report", action="store_true", help="build the HTML report")
    parser.add_argument(
        "--report-out",
        default="artifacts/report.html",
        help="report path relative to the repository root",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure(verbose=not args.quiet)

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()] if args.stages else None
    settings = load_settings(
        overrides={"data": {"start": args.start, "end": args.end}}
        if (args.start or args.end)
        else None
    )
    result = run_pipeline(settings, stages=stages, force=args.force)
    print(result.summary().to_string(index=False))
    print(f"run: {result.run_id} -> {result.run_dir}")

    if args.report:
        from nifty_sensi.reporting import build_run_report

        output = settings.root / args.report_out
        path = build_run_report(settings, run_id=result.run_id, output=output)
        print(f"report: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
