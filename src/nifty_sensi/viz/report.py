"""Self-contained HTML report with embedded PNG figures and tables."""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..paths import atomic_write_text

_CSS = """
:root { --ink:#16202a; --muted:#5b6b7a; --line:#d9e2ec; --accent:#0b6bcb; --bg:#f7f9fc; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
header { border-bottom: 3px solid var(--accent); padding-bottom: 16px; margin-bottom: 24px; }
h1 { margin: 0 0 6px; font-size: 30px; }
h2 { margin: 34px 0 12px; font-size: 21px; border-left: 4px solid var(--accent); padding-left: 10px; }
h3 { margin: 22px 0 8px; font-size: 16px; color: var(--muted); }
p, li { line-height: 1.55; }
.sub { color: var(--muted); font-size: 13px; }
.metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin: 16px 0; }
.metric { background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.metric .k { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.metric .v { font-size:21px; font-weight:650; margin-top:4px; }
figure { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px; margin:16px 0; }
figure img { width:100%; height:auto; display:block; }
figcaption { color:var(--muted); font-size:12px; margin-top:8px; }
table { border-collapse: collapse; width:100%; background:#fff; font-size:12px;
        border:1px solid var(--line); border-radius:10px; overflow:hidden; }
th, td { padding:7px 9px; border-bottom:1px solid var(--line); text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:#eef4fb; font-weight:600; }
tbody tr:nth-child(even) { background:#fbfdff; }
code { background:#eef2f7; padding:1px 5px; border-radius:4px; font-size:12px; }
.note { background:#fff8e6; border:1px solid #f0d58a; border-radius:10px; padding:12px 14px; }
footer { margin-top:40px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:12px; }
"""


def figure_png(fig, dpi: int = 110) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def figure_data_uri(fig, dpi: int = 110) -> str:
    payload = base64.b64encode(figure_png(fig, dpi=dpi)).decode("ascii")
    return f"data:image/png;base64,{payload}"


class ReportBuilder:
    """Accumulate report blocks and write a single portable HTML file."""

    def __init__(self, title: str, subtitle: str | None = None, dpi: int = 110):
        self.title = title
        self.subtitle = subtitle or ""
        self.dpi = dpi
        self.blocks: list[str] = []

    # ------------------------------------------------------------------ blocks
    def add_heading(self, text: str, level: int = 2) -> None:
        self.blocks.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def add_text(self, text: str) -> None:
        self.blocks.append(f"<p>{html.escape(text)}</p>")

    def add_html(self, snippet: str) -> None:
        self.blocks.append(snippet)

    def add_note(self, text: str) -> None:
        self.blocks.append(f'<div class="note">{html.escape(text)}</div>')

    def add_metrics(self, metrics: dict[str, object]) -> None:
        cards = []
        for key, value in metrics.items():
            if isinstance(value, float):
                value = f"{value:,.4g}"
            cards.append(
                f'<div class="metric"><div class="k">{html.escape(str(key))}</div>'
                f'<div class="v">{html.escape(str(value))}</div></div>'
            )
        self.blocks.append(f'<div class="metrics">{"".join(cards)}</div>')

    def add_table(self, frame: pd.DataFrame, title: str | None = None, max_rows: int = 25) -> None:
        if title:
            self.blocks.append(f"<h3>{html.escape(title)}</h3>")
        if frame is None or frame.empty:
            self.blocks.append("<p class='sub'>No rows.</p>")
            return
        display = frame.head(max_rows)
        table = display.to_html(
            index=False, border=0, classes="table", float_format=lambda value: f"{value:,.4g}"
        )
        self.blocks.append(table)
        if len(frame) > max_rows:
            self.blocks.append(
                f"<p class='sub'>Showing {max_rows} of {len(frame):,} rows. "
                "Full data is in the run artifacts.</p>"
            )

    def add_figure(self, fig, title: str | None = None, dpi: int | None = None, close: bool = True) -> None:
        if title:
            self.blocks.append(f"<h3>{html.escape(title)}</h3>")
        uri = figure_data_uri(fig, dpi=dpi or self.dpi)
        caption = html.escape(title) if title else ""
        self.blocks.append(
            f'<figure><img src="{uri}" alt="{caption}"/>'
            f"<figcaption>{caption}</figcaption></figure>"
        )
        if close:
            plt.close(fig)

    # -------------------------------------------------------------------- save
    def to_html(self) -> str:
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        subtitle = f'<p class="sub">{html.escape(self.subtitle)}</p>' if self.subtitle else ""
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'/>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'/>"
            f"<title>{html.escape(self.title)}</title><style>{_CSS}</style></head><body><main>"
            f"<header><h1>{html.escape(self.title)}</h1>{subtitle}"
            f"<p class='sub'>Generated {generated}</p></header>"
            + "\n".join(self.blocks)
            + "<footer>nifty_sensi — Black-76 pricing, documented Greek units, "
            "content-addressed pipeline runs.</footer></main></body></html>"
        )

    def save(self, path: str | Path) -> Path:
        return atomic_write_text(path, self.to_html())
