"""
heatmap.py — Upgraded Heat Map Engine for AI Trade Terminal
===========================================================
Modular, standalone Treemap visualizer optimized for both
desktop wide screens and mobile touch displays.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

_DARK_BG = "#0F172A"
_TILE_LINE = "#020617"

# Symmetric diverging scale: -3% (deep red) -> 0% (slate) -> +3% (deep green)
_CHANGE_COLORSCALE: List[List[Any]] = [
    [0.00, "#7f1d1d"],
    [0.25, "#b91c1c"],
    [0.50, "#1e293b"],
    [0.75, "#047857"],
    [1.00, "#065f46"],
]

_CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "INR": "₹", "HKD": "HK$", "AUD": "A$", "SGD": "S$",
    "KRW": "₩", "CNY": "¥", "CAD": "C$", "CHF": "CHF ", "NZD": "NZ$",
}


def _fmt_price(value: Any, currency: str = "USD") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(num):
        return "—"
    sym = _CURRENCY_SYMBOL.get(str(currency or "USD").upper(), f"{currency} ")
    if abs(num) >= 1000:
        return f"{sym}{num:,.2f}"
    return f"{sym}{num:.2f}"


def _fmt_pct(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(num):
        return "—"
    return f"{num:+.2f}%"


def _wrap(text: Any, width: int = 14, limit_lines: int = 2) -> str:
    name = str(text or "").strip()
    if not name:
        return ""
    lines = textwrap.wrap(name, width=width, max_lines=limit_lines, placeholder="…")
    return "<br>".join(lines)


def build_heatmap_figure(
    frame: pd.DataFrame,
    size_mode: str = "Equal weight",
    color_mode: str = "% change",
    group_mode: str = "Sectors",
    currency: str = "USD",
    max_tiles: int = 120,
    pct_range: float = 3.0,
    is_mobile: bool = False,
) -> Optional[go.Figure]:
    """Builds an adaptive, touch-friendly Plotly Treemap."""
    if frame is None or frame.empty:
        return None

    work = frame.copy()
    if "price" in work.columns:
        work = work[pd.to_numeric(work["price"], errors="coerce").notna()]
    if work.empty:
        return None

    work["change_percent"] = pd.to_numeric(work.get("change_percent"), errors="coerce").fillna(0.0)
    work["market_cap"] = pd.to_numeric(work.get("market_cap"), errors="coerce")
    work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0)

    # Area sizing logic
    if size_mode == "Market cap" and work["market_cap"].notna().any():
        work["size_value"] = work["market_cap"].fillna(work["market_cap"].median() or 1.0).clip(lower=1.0)
    elif size_mode == "Volume":
        work["size_value"] = work["volume"].clip(lower=1.0)
    else:
        work["size_value"] = 1.0

    work = work.sort_values("size_value", ascending=False).head(int(max_tiles))

    # Hierarchy group mapping
    if group_mode == "Industries" and "industry" in work.columns:
        work["group"] = work["industry"].fillna("").replace("", "General")
    elif group_mode == "Sectors" and "sector" in work.columns:
        work["group"] = work["sector"].fillna("").replace("", "General")
    else:
        work["group"] = "All Assets"

    work["color_value"] = work["change_percent"].clip(lower=-pct_range, upper=pct_range)
    cmin, cmax = -float(pct_range), float(pct_range)

    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    colors: List[float] = []
    texts: List[str] = []
    customs: List[List[Any]] = []

    wrap_w = 12 if is_mobile else 18
    for _, row in work.iterrows():
        symbol = str(row.get("symbol", "?"))
        full_name = str(row.get("long_name") or row.get("name") or "").strip()
        curr = str(row.get("currency") or currency or "USD").upper()
        price_txt = _fmt_price(row.get("price"), curr)
        pct_txt = _fmt_pct(row.get("change_percent"))

        name_block = _wrap(full_name, width=wrap_w, limit_lines=2)
        label_block = f"<b>{symbol}</b>"
        if not is_mobile and name_block and name_block.upper() != symbol.upper():
            label_block += f"<br>{name_block}"
        label_block += f"<br>{pct_txt}"

        labels.append(symbol)
        parents.append(str(row["group"]))
        values.append(float(row["size_value"]))
        color_val = float(row["color_value"])
        colors.append(color_val if math.isfinite(color_val) else 0.0)
        texts.append(label_block)
        customs.append([
            full_name or symbol,
            price_txt,
            pct_txt,
            f"{row.get('market_cap', 0):,.0f}" if pd.notna(row.get("market_cap")) else "—",
            f"{row.get('volume', 0):,.0f}" if pd.notna(row.get("volume")) else "—",
            str(row.get("sector") or "—"),
            curr,
        ])

    # Top-level group headers
    for grp in sorted(work["group"].unique().tolist()):
        m = work[work["group"] == grp]
        labels.append(str(grp))
        parents.append("")
        values.append(float(m["size_value"].sum()))
        grp_col = float(m["color_value"].mean()) if len(m) else 0.0
        colors.append(grp_col if math.isfinite(grp_col) else 0.0)
        texts.append(f"<b>{grp}</b>")
        customs.append([grp, "—", "—", "—", "—", grp, currency])

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=texts,
        textinfo="text",
        customdata=customs,
        marker=dict(
            colors=colors,
            cmid=0.0,
            cmin=cmin,
            cmax=cmax,
            colorscale=_CHANGE_COLORSCALE,
            line=dict(width=1.0 if is_mobile else 1.5, color=_TILE_LINE),
        ),
        hovertemplate=(
            "<b>%{label}</b> — %{customdata[0]}<br>"
            "Price: %{customdata[1]} (%{customdata[2]})<br>"
            "Volume: %{customdata[4]}<br>"
            "Sector: %{customdata[5]}<extra></extra>"
        ),
        root_color=_DARK_BG,
        branchvalues="total",
    ))

    plot_height = 420 if is_mobile else 560
    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        height=plot_height,
        paper_bgcolor=_DARK_BG,
        font=dict(color="#f8fafc", size=10 if is_mobile else 12),
    )
    return fig
