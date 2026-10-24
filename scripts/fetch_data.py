"""Fetch raw NIFTY spot and option data from the public sources.

Existing files are never overwritten unless ``--force`` is given, and the
default output directory is ``data/raw`` so the committed canonical feeds
(``data/nifty_close.csv`` and ``data/niftyoptions_df.csv``) stay untouched.

Examples::

    python scripts/fetch_data.py --source spot --start 2025-01-01 --end 2025-01-31
    python scripts/fetch_data.py --source options --start 2025-01-01 --end 2025-01-31

The option feed requires the optional ingestion dependencies::

    pip install "nifty-sensi[ingest]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["spot", "options", "all"], default="all")
    parser.add_argument("--start", required=True, help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="last date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--revision", help="pin a HuggingFace dataset revision")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    return parser.parse_args(argv)


def _target(out_dir: Path, name: str, force: bool) -> Path:
    path = out_dir / name
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path}; pass --force to replace it")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source in {"spot", "all"}:
        from nifty_sensi.ingest_spot import download_spot, save_spot

        spot = download_spot(args.start, args.end)
        path = save_spot(spot, _target(out_dir, "nifty_close.csv", args.force))
        print(f"spot: {len(spot):,} rows -> {path}")

    if args.source in {"options", "all"}:
        from nifty_sensi.ingest_options import fetch_options, save_options

        options = fetch_options(args.start, args.end, revision=args.revision)
        path = save_options(options, _target(out_dir, "niftyoptions_df.csv", args.force))
        print(f"options: {len(options):,} rows -> {path}")

    print("done. Point data.options_file / data.spot_file in your config at these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
