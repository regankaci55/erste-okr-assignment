"""Colours and small HTML/chart helpers shared by the report pages."""
import html
import json

import pandas as pd

ALL = "All teams"
STATUS_ORDER = ["Done", "On track", "At risk", "Not started"]
STATUS_COLORS = {"Done": "#0ca30c", "On track": "#2a78d6", "At risk": "#d03b3b", "Not started": "#c3c2b7"}
STATUS_TEXT = {"Done": "#ffffff", "On track": "#ffffff", "At risk": "#ffffff", "Not started": "#0b0b0b"}
BLUE, SERIOUS, INK, SECONDARY, MUTED, GRID, BASE, SURFACE = "#2a78d6", "#ec835a", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def esc(value):
    return html.escape("" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))


def pct(part, whole, digits=0):
    return f"{part / whole * 100:.{digits}f}%" if whole else "n/a"


def table(headers, rows, numeric=()):
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{esc(c)}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def base_layout(height, **extra):
    layout = dict(
        template="none", height=height, autosize=True, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=SECONDARY),
        margin=dict(l=8, r=28, t=36, b=36), hoverlabel=dict(bgcolor="#ffffff", font_color=INK, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", traceorder="normal"),
        xaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=BASE, tickfont_color=MUTED, fixedrange=True),
        yaxis=dict(showgrid=False, automargin=True, ticks="", linecolor=BASE, tickfont_color=SECONDARY, fixedrange=True, ticksuffix="  "),
        barcornerradius=4,
    )
    layout.update(extra)
    return layout


def fig_json(figure):
    return json.loads(figure.to_json())
