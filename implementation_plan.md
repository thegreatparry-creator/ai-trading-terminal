"""
heatmap.py — Upgraded heat map for AI Trade Terminal
=====================================================
Standalone, reusable snippet. Import it into app.py and it replaces the
built-in treemap engine:

    from heatmap import build_heatmap_figure

    fig = build_heatmap_figure(frame)          # works directly with the
    st.plotly_chart(fig, use_container_width=True)  # app's normalized frame

Upgrades over the built-in version
----------------------------------
1. Tiles are coloured STRICTLY by daily % change, centred on 0 with a
   symmetric +/-3% range (deeper red/green for bigger moves).
2. Every tile is labelled with the TICKER SYMBOL in bold plus the FULL
   stock name (word-wrapped), then price and % change underneath.
3. Hover carries the full un-truncated name, sector, industry, market cap,
   volume, currency and data status.
4. Works with the app's normalized frame columns but degrades gracefully
   if a column is missing.

No Streamlit / app imports here — safe to unit-test standalone.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, List

import pandas as pd
import plotly.graph_objects as go

# --- theme (matches the terminal's dark slate palette) ----------------------
_DARK_BG = "#0F172A"
_TILE_LINE = "#020617"

# Symmetric diverging scale centred on 0 %: deep red -> red -> neutral -> green -> deep green
_CHANGE_COLORSCALE: List[List[Any]] = [
    [0.00, "#7f1d1d"],
    [0.25, "#b91c1c"],
    [0.50, "#1e293b"],
    [0.75, "#047857"],
    [1.00, "#065f46"],
]

# Minimal currency-symbol map so the module is standalone (never converts,
# only displays in the instrument's own currency).
_CURRENCY_SYMBOL = {
    "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5",
    "INR": "\u20b9", "HKD": "HK$", "AUD": "A$", "SGD": "S$", "KRW": "\u20a9",
    "CNY": "\u00a5", "CAD": "C$", "CHF": "CHF ", "NZD": "NZ$",
}


def _fmt_price(value: Any, currency: str = "USD") -> str:
    """Format a price in the instrument's own currency."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "\u2014"
    if not math.isfinite(num):
        return "\u2014"
    sym = _CURRENCY_SYMBOL.get(str(currency or "USD").upper(), f"{currency} ")
    if abs(num) >= 1000:
        return f"{sym}{num:,.2f}"
    return f"{sym}{num:.2f}"


def _fmt_pct(value: Any) -> str:
    """Format a percentage change with an explicit +/- sign."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "\u2014"
    if not math.isfinite(num):
        return "\u2014"
    return f"{num:+.2f}%"


def _wrap(text: Any, width: int = 22, limit_lines: int = 3) -> str:
    """Word-wrap a long name into HTML line breaks for the tile label."""
    name = str(text or "").strip()
    if not name:
        return ""
    lines = textwrap.wrap(name, width=width, max_lines=limit_lines,
                          placeholder="\u2026")
    return "<br>".join(lines)


def build_heatmap_figure(frame: pd.DataFrame, size_mode: str = "Market cap",
                         color_mode: str = "% change", group_mode: str = "Sectors",
                         currency: str = "USD", max_tiles: int = 250,
                         pct_range: float = 3.0):
    """Build the upgraded heat map treemap.

    Parameters
    ----------
    frame       : the app's normalized market frame (see normalize_rows()).
    size_mode   : "Market cap" | "Volume" | "Equal weight"  (tile area).
    color_mode  : kept for signature compatibility; colour is ALWAYS the
                  daily % change (symmetric around 0).
    group_mode  : "Sectors" | "Industries" | "Flat".
    currency    : fallback currency when a row has none.
    max_tiles   : maximum number of instrument tiles to render.
    pct_range   : +/- percent mapped to the ends of the colour scale.
    """
    if frame is None or frame.empty:
        return None

    work = frame.copy()
    if "price" in work.columns:
        work = work[pd.to_numeric(work["price"], errors="coerce").notna()]
    if work.empty:
        return None

    # --- numeric coercion ----------------------------------------------------
    work["change_percent"] = pd.to_numeric(work.get("change_percent"),
                                           errors="coerce").fillna(0.0)
    work["market_cap"] = pd.to_numeric(work.get("market_cap"), errors="coerce")
    work["volume"] = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0)

    # --- tile size -----------------------------------------------------------
    if size_mode == "Market cap" and work["market_cap"].notna().any():
        work["size_value"] = work["market_cap"].fillna(
            work["market_cap"].median() or 1.0).clip(lower=1.0)
    elif size_mode == "Volume":
        work["size_value"] = work["volume"].clip(lower=1.0)
    else:
        work["size_value"] = 1.0

    work = work.sort_values("size_value", ascending=False).head(int(max_tiles))

    # --- grouping ------------------------------------------------------------
    if group_mode == "Industries" and "industry" in work.columns:
        work["group"] = work["industry"].fillna("").replace("", "Unclassified")
    elif group_mode == "Sectors" and "sector" in work.columns:
        work["group"] = work["sector"].fillna("").replace("", "Unclassified")
    else:
        work["group"] = "All instruments"

    # --- colour: STRICTLY daily % change, symmetric around zero --------------
    work["color_value"] = work["change_percent"].clip(lower=-pct_range,
                                                      upper=pct_range)
    cmin, cmax = -float(pct_range), float(pct_range)

    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    colors: List[float] = []
    texts: List[str] = []
    customs: List[List[Any]] = []

    for _, row in work.iterrows():
        symbol = str(row.get("symbol", "?"))
        full_name = str(row.get("long_name") or row.get("name") or "").strip()
        currency_row = str(row.get("currency") or currency or "USD").upper()
        price_txt = _fmt_price(row.get("price"), currency_row)
        pct_txt = _fmt_pct(row.get("change_percent"))

        # TILE LABEL: bold ticker + FULL stock name, wrapped; price + % change
        name_block = _wrap(full_name, width=22, limit_lines=3)
        label_block = f"<b>{symbol}</b>"
        if name_block and name_block.upper() != symbol.upper():
            label_block += f"<br>{name_block}"
        label_block += f"<br>{price_txt}  \u00b7  {pct_txt}"

        labels.append(symbol)
        parents.append(str(row["group"]))
        values.append(float(row["size_value"]))
        color_value = float(row["color_value"])
        colors.append(color_value if math.isfinite(color_value) else 0.0)
        texts.append(label_block)
        customs.append([
            full_name or symbol,                       # 0 full (untruncated) name
            price_txt,                                 # 1 price, own currency
            pct_txt,                                   # 2 daily % change
            f"{row.get('market_cap', 0):,.0f}" if pd.notna(row.get("market_cap")) else "\u2014",
            f"{row.get('volume', 0):,.0f}" if pd.notna(row.get("volume")) else "\u2014",
            str(row.get("sector") or "\u2014"),
            str(row.get("industry") or "\u2014"),
            currency_row,
            str(row.get("data_status") or "\u2014"),
            str(row.get("last_bar") or "\u2014"),
        ])

    # group header tiles
    for group in sorted(work["group"].unique().tolist()):
        members = work[work["group"] == group]
        labels.append(str(group))
        parents.append("")
        values.append(float(members["size_value"].sum()))
        group_color = float(members["color_value"].mean()) if len(members) else 0.0
        colors.append(group_color if math.isfinite(group_color) else 0.0)
        texts.append(f"<b>{group}</b><br>{len(members)} names")
        customs.append([group, "\u2014", "\u2014", "\u2014", "\u2014",
                        group, "\u2014", currency, "", ""])

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, text=texts, textinfo="text",
        customdata=customs,
        marker=dict(
            colors=colors, cmid=0.0, cmin=cmin, cmax=cmax,
            colorscale=_CHANGE_COLORSCALE,
            line=dict(width=1.5, color=_TILE_LINE)),
        hovertemplate=(
            "<b>%{label}</b> \u2014 %{customdata[0]}<br>"
            "Price: %{customdata[1]} (\u00b7%{customdata[2]})<br>"
            "Market cap: %{customdata[3]}<br>Volume: %{customdata[4]}<br>"
            "Sector: %{customdata[5]}<br>Industry: %{customdata[6]}<br>"
            "Currency: %{customdata[7]}<br>Data status: %{customdata[8]}<br>"
            "Last bar: %{customdata[9]}<extra></extra>"),
        root_color=_DARK_BG, branchvalues="total",
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=560,
                      paper_bgcolor=_DARK_BG,
                      font=dict(color="#e2e8f0", size=12))
    return fig
