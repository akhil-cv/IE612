"""Execute the notebook without Jupyter.

This is the notebook smoke test used by CI and by the maintainers::

    python scripts/check_notebook.py --start 2025-01-01 --end 2025-01-10
    python scripts/check_notebook.py --keep-outputs   # save executed notebook

``--keep-outputs`` captures text and PNG figures into the notebook's output
cells, producing a readable artefact with every chart inline. Cell magics such as
``%matplotlib inline`` are skipped (the executor forces the Agg backend).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")


def strip_magics(source: str) -> str:
    lines = [line for line in source.splitlines() if not line.lstrip().startswith("%")]
    return "\n".join(lines)


def figure_output(figure, plt) -> dict:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=90, bbox_inches="tight")
    plt.close(figure)
    return {
        "output_type": "display_data",
        "data": {
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "text/plain": ["<Figure>"],
        },
        "metadata": {},
    }


def figure_outputs(matplotlib, plt) -> list[dict]:
    """Safety net: capture and close figures that were never displayed."""
    return [figure_output(plt.figure(number), plt) for number in list(plt.get_fignums())]


def install_display_hook(matplotlib, plt) -> list[dict]:
    """Capture figures passed to ``IPython.display.display`` and close them.

    Notebook cells display a figure and then close it immediately, so the
    end-of-cell sweep would find nothing to embed. Capturing at display time
    keeps ``--keep-outputs`` complete and renders each chart exactly once.
    """
    import IPython.display as ipy_display

    captured: list[dict] = []
    real_display = ipy_display.display
    figure_type = matplotlib.figure.Figure

    def display(*objs, **kwargs):
        for obj in objs:
            if isinstance(obj, figure_type):
                captured.append(figure_output(obj, plt))
            else:
                real_display(obj, **kwargs)

    ipy_display.display = display
    return captured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="notebooks/Sensi.ipynb")
    parser.add_argument("--start", help="override NIFTY_SENSI_START")
    parser.add_argument("--end", help="override NIFTY_SENSI_END")
    parser.add_argument("--force", action="store_true", help="override NIFTY_SENSI_FORCE")
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.start:
        os.environ["NIFTY_SENSI_START"] = args.start
    if args.end:
        os.environ["NIFTY_SENSI_END"] = args.end
    if args.force:
        os.environ["NIFTY_SENSI_FORCE"] = "1"

    os.chdir(ROOT)
    path = (ROOT / args.notebook).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import warnings

    warnings.filterwarnings("ignore", message=".*non-interactive.*")

    captured = install_display_hook(matplotlib, plt)
    namespace: dict = {"__name__": "__main__"}
    executed = 0
    figures = 0

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = strip_magics("".join(cell["source"]))
        if not source.strip():
            continue
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, f"<cell {index}>", "exec"), namespace)
        except Exception:
            print(f"FAILED cell {index}\n{source}\n")
            traceback.print_exc()
            return 1

        executed += 1
        images = captured + figure_outputs(matplotlib, plt)
        captured.clear()
        figures += len(images)
        text = buffer.getvalue()
        if args.keep_outputs:
            outputs = []
            if text:
                outputs.append(
                    {"output_type": "stream", "name": "stdout", "text": text.splitlines(True)}
                )
            outputs.extend(images)
            cell["outputs"] = outputs
            cell["execution_count"] = executed
        if text and not args.quiet:
            print(text.rstrip())

    report = ROOT / "artifacts" / "report.html"
    if not report.exists():
        print(f"report not found: {report}")
        return 1
    html = report.read_text(encoding="utf-8")
    embedded = html.count("data:image/png;base64,")
    if embedded < 8:
        print(f"report has only {embedded} embedded figures")
        return 1
    if 'src="http' in html:
        print("report references external images")
        return 1

    if args.keep_outputs:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"notebook OK: {executed} code cells, {figures} inline figures, "
        f"report {embedded} embedded figures ({report})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
