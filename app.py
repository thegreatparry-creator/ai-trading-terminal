"""
AI Trade Terminal - Streamlit Edition (v4.5.0)
==============================================
Main dashboard application file. Imports heatmap.py and watchlist_alerts.py.
Configured with responsive styling for desktop monitors and mobile devices.
"""

from __future__ import annotations

import concurrent.futures as futures
import html
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Import sibling modules
from heatmap import build_heatmap_figure
from watchlist_alerts import AlertEngine, send_telegram

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

APP_VERSION = "4.5.0"
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".cache"
STATE_FILE = CACHE_DIR / "terminal_state.json"

AI_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "Groq — Llama 3.3 70B": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "Groq — GPT-OSS 20B": {"provider": "groq", "model": "openai/gpt-oss-20b"},
}
DEFAULT_AI_LABEL = "Groq — Llama 3.3 70B"

LOG = logging.getLogger("ai_trade_terminal")
if not LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(_h)
LOG.setLevel(logging.INFO)

st.set_page_config(
    page_title="AI Trade Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# RESPONSIVE DESKTOP & MOBILE CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
.stApp { background-color:#020617 !important; color:#f8fafc; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
#MainMenu, footer, header { visibility:hidden; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#0F172A;
    border:1px solid rgba(51,65,85,.45) !important;
    border-radius:14px !important;
    padding: 10px !important;
}

section[data-testid="stSidebar"] {
    background-color:#0B1220 !important;
    border-right:1px solid rgba(51,65,85,.4);
}

.card-header { font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:.1rem; }
.card-sub { font-size:.75rem; color:#94a3b8; margin-bottom:.5rem; }
.metric-label { font-size:.68rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; }
.metric-value { font-size:1.3rem; font-weight:700; color:#f8fafc; font-variant-numeric:tabular-nums; }

.up { color:#34d399 !important; } .down { color:#f87171 !important; } .flat { color:#94a3b8 !important; } .amber { color:#fbbf24 !important; }

.pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.70rem; font-weight:700; }
.pill-open { background:rgba(16,185,129,.18); color:#34d399; border:1px solid rgba(16,185,129,.4); }
.pill-closed { background:rgba(148,163,184,.16); color:#cbd5e1; border:1px solid rgba(148,163,184,.35); }
.pill-delayed { background:rgba(96,165,250,.16); color:#93c5fd; border:1px solid rgba(96,165,250,.4); }

.stButton > button {
    border-radius:10px !important;
    font-weight:600 !important;
    border:1px solid rgba(51,65,85,.55) !important;
    background:#1E293B !important;
    color:#e2e8f0 !important;
    min-height: 42px;
    width: 100%;
}
.stButton > button:hover { border-color:#fbbf24 !important; color:#fbbf24 !important; }

.level-row { display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-radius:9px; margin-bottom:4px; font-size:.85rem; }
.level-support { background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.2); }
.level-resistance { background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.2); }
.level-anchor { background:rgba(251,191,36,.14); border:1px solid rgba(251,191,36,.4); }
.level-current { background:rgba(96,165,250,.14); border:1px solid rgba(96,165,250,.4); }

.plan-card { border-radius:12px; padding:10px 14px; margin-bottom:8px; }
.plan-bull { background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35); }
.plan-bear { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35); }

@media (max-width:768px) {
  .metric-value { font-size:1.1rem; }
  .card-header { font-size:0.95rem; }
  .level-row { font-size:0.78rem; padding:6px 8px; }
  .stButton > button { min-height: 46px; }
}
</style>
"""

# ---------------------------------------------------------------------------
# HELPERS, FORMATTERS & SECRETS
# ---------------------------------------------------------------------------
def secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name, None)
        if val not in (None, ""):
            return str(val).strip()
    except Exception:
        pass
    return (os.environ.get(name, default) or default).strip()

def groq_key() -> str:
    return secret("GROQ_KEY")

def telegram_ready() -> bool:
    return bool(secret("TELEGRAM_BOT_TOKEN") and secret("TELEGRAM_CHAT_ID"))

def _redact(text: Any) -> str:
    out = str(text)
    for n in ("GROQ_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        v = secret(n)
        if v and len(v) >= 6:
            out = out.replace(v, "***")
    out = re.sub(r"gsk_[A-Za-z0-9]+", "gsk_***", out)
    out = re.sub(r"bot\d+:[A-Za-z0-9_\-]+", "bot***", out)
    return out

def safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_redact(exc)}"[:400]

def log_exception(ctx: str, exc: BaseException) -> None:
    LOG.warning("%s | %s", ctx, safe_error(exc))

def _num(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        n = float(str(val).replace(",", "").strip()) if isinstance(val, str) else float(val)
        return n if math.isfinite(n) else None
    except Exception:
        return None

def valid_price(val: Any) -> Optional[float]:
    n = _num(val)
    return n if n is not None and n > 0 else None

CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "HKD": "HK$", "AUD": "A$", "CAD": "C$", "SGD": "S$", "KRW": "₩",
    "CNY": "¥", "CHF": "CHF ", "NZD": "NZ$",
}
SUFFIX_CURRENCY = {
    ".NS": "INR", ".BO": "INR", ".L": "GBP", ".DE": "EUR", ".PA": "EUR",
    ".T": "JPY", ".HK": "HKD", ".AX": "AUD", ".TO": "CAD", ".SI": "SGD", ".KS": "KRW",
}

def guess_currency(symbol: str) -> str:
    raw = (symbol or "").upper()
    if raw.startswith("^NSE") or raw.startswith("^BSE") or raw.startswith("^CNX"):
        return "INR"
    if raw.startswith("^") or raw.endswith("-USD") or raw.endswith("=F"):
        return "USD"
    if raw.endswith("=X"):
        pair = raw[:-2]
        return pair[-3:] if len(pair) >= 6 and pair[-3:] in CURRENCY_SYMBOLS else "USD"
    for sfx, cur in SUFFIX_CURRENCY.items():
        if raw.endswith(sfx):
            return cur
    return "USD"

def _group_indian(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])

def fmt_price(value: Any, currency: str = "USD", decimals: Optional[int] = None) -> str:
    n = _num(value)
    if n is None or n <= 0:
        return "—"
    curr = (currency or "USD").upper()
    if decimals is None:
        decimals = 4 if n < 10 else 2
    if curr == "INR":
        ipart, _, fpart = f"{n:.{decimals}f}".partition(".")
        return "₹" + _group_indian(ipart) + (f".{fpart}" if fpart else "")
    sym = CURRENCY_SYMBOLS.get(curr, f"{curr} ")
    return f"{sym}{n:,.{decimals}f}"

def fmt_pct(value: Any, decimals: int = 2) -> str:
    n = _num(value)
    return "—" if n is None else f"{'+' if n >= 0 else ''}{n:.{decimals}f}%"

def fmt_volume(value: Any) -> str:
    n = _num(value)
    if n is None or n < 0:
        return "—"
    for scale, sfx in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            return f"{n / scale:.2f}{sfx}"
    return f"{n:,.0f}"

def change_class(val: Any) -> str:
    n = _num(val)
    if n is None:
        return "flat"
    return "up" if n > 0 else ("down" if n < 0 else "flat")

def metric_html(label: str, value: str, cls: str = "", sub: str = "") -> str:
    sub_tag = f"<div class='card-sub' style='margin:2px 0 0 0'>{sub}</div>" if sub else ""
    return f"<div class='metric-label'>{label}</div><div class='metric-value {cls}'>{value}</div>{sub_tag}"

def show_df(df: pd.DataFrame, **kw: Any) -> None:
    try:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)
    except TypeError:
        st.dataframe(df, hide_index=True, **kw)

def show_fig(fig: Any, **kw: Any) -> None:
    try:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)
    except TypeError:
        st.plotly_chart(fig, **kw)

# ---------------------------------------------------------------------------
# MARKET CATALOG
# ---------------------------------------------------------------------------
NAMES: Dict[str, str] = {
    "ES=F": "E-mini S&P 500", "NQ=F": "Nasdaq 100 E-mini", "YM=F": "Dow E-mini",
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil WTI", "BZ=F": "Brent Crude",
    "^NSEI": "NIFTY 50", "^NSEBANK": "NIFTY Bank", "^BSESN": "SENSEX", "^CNXIT": "NIFTY IT",
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones",
    "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^N225": "Nikkei 225", "^HSI": "Hang Seng",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
}

def display_name(symbol: str) -> str:
    raw = (symbol or "").upper()
    if raw in NAMES:
        return NAMES[raw]
    if raw.endswith("=X") and len(raw) >= 8:
        return f"{raw[:3]}/{raw[3:6]}"
    for sfx in (".NS", ".BO"):
        if raw.endswith(sfx):
            return raw[:-len(sfx)]
    return raw

N50 = (
    "RELIANCE TCS HDFCBANK ICICIBANK INFY SBIN BHARTIARTL ITC LT HINDUNILVR KOTAKBANK AXISBANK BAJFINANCE "
    "MARUTI ASIANPAINT HCLTECH WIPRO SUNPHARMA TITAN ULTRACEMCO POWERGRID NTPC TATAMOTORS TATASTEEL JSWSTEEL "
    "M&M ADANIENT ADANIPORTS COALINDIA ONGC CIPLA"
).split()

M_IN, M_US, M_CR, M_FX, M_CM, M_FU, M_GI, M_OTC = (
    "India", "United States", "Cryptocurrency", "Forex",
    "Commodities", "Futures", "Global Indices", "OTC / CFD / Synthetic"
)

def build_tree() -> Dict[str, Any]:
    return {
        M_IN: {
            "NSE": {"NIFTY 50": [(f"{s}.NS", s) for s in N50], "All / Custom": []},
            "Indices": {"Benchmark": [("^NSEI", "NIFTY 50"), ("^NSEBANK", "NIFTY Bank"), ("^BSESN", "SENSEX")]},
        },
        M_US: {
            "NASDAQ": {"Mega Tech": [(s, s) for s in "AAPL MSFT NVDA GOOGL AMZN META AVGO AMD TSLA".split()]},
            "NYSE": {"Blue Chips": [(s, s) for s in "JPM BAC WFC XOM CVX JNJ UNH WMT PG".split()]},
        },
        M_CR: {"Crypto": {"Top Coins": [(f"{s}-USD", s) for s in "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK".split()]}},
        M_FX: {"Forex": {"Majors": [(f"{p}=X", f"{p[:3]}/{p[3:]}") for p in "EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD USDINR".split()]}},
        M_CM: {"Commodities": {"Metals & Energy": [("GC=F", "Gold"), ("SI=F", "Silver"), ("CL=F", "Crude Oil"), ("NG=F", "Natural Gas")]}},
        M_FU: {"Futures": {"Equity": [("ES=F", "S&P Futures"), ("NQ=F", "Nasdaq Futures"), ("YM=F", "Dow Futures")]}},
        M_GI: {"Indices": {"Global": [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^FTSE", "FTSE 100"), ("^N225", "Nikkei 225")]}},
        M_OTC: {"Proxies": {"Proxies": [("EURUSD=X", "EUR/USD"), ("GC=F", "Gold"), ("BTC-USD", "Bitcoin")]}},
    }

MARKET_TREE = build_tree()

def normalize_symbol(raw: str, market: str, exchange: str = "") -> str:
    s = (raw or "").strip().upper().replace(" ", "")
    if not s or market == M_OTC:
        return s
    if s.startswith("^") or s.endswith("=F") or s.endswith("=X"):
        return s
    if market == M_IN:
        return s if "." in s else f"{s}.BO" if exchange == "BSE" else f"{s}.NS"
    if market == M_CR:
        return s if s.endswith("-USD") else f"{s.replace('/', '')}-USD"
    if market == M_FX:
        clean = s.replace("/", "")
        return f"{clean}=X" if len(clean) >= 6 else f"{clean}USD=X"
    if market in (M_CM, M_FU):
        return s if s.endswith("=F") else f"{s}=F"
    return s

def market_session(market: str) -> Dict[str, Any]:
    tz_map = {M_IN: ("Asia/Kolkata", (9, 15), (15, 30)), M_US: ("America/New_York", (9, 30), (16, 0))}
    if market in tz_map:
        tz_name, (oh, om), (ch, cm) = tz_map[market]
        tz = ZoneInfo(tz_name) if ZoneInfo else timezone.utc
        now = datetime.now(tz)
        wd = now.weekday()
        mins = now.hour * 60 + now.minute
        is_open = (wd < 5) and (oh * 60 + om <= mins <= ch * 60 + cm)
        status = "OPEN" if is_open else "CLOSED"
        stamp = now.strftime("%H:%M") + f" ({now.tzname() or tz_name})"
    elif market == M_CR:
        status, is_open, stamp = "OPEN", True, "24/7 Continuous"
    else:
        now = datetime.now(timezone.utc)
        is_open = not (now.weekday() == 5 or (now.weekday() == 4 and now.hour >= 22))
        status, stamp = ("OPEN", "24/5 Active") if is_open else ("CLOSED", "Weekend Break")
    return {"status": status, "is_open": is_open, "stamp": stamp}

def status_pill(text: str) -> str:
    css = {"OPEN": "pill-open", "LIVE": "pill-open", "DELAYED": "pill-delayed"}.get(text, "pill-closed")
    return f"<span class='pill {css}'>{text}</span>"

# ---------------------------------------------------------------------------
# MARKET DATA PIPELINE
# ---------------------------------------------------------------------------
def _clean(frame: Any) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns]
    df = df.rename(columns={c: str(c).strip().title() for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    if not all(c in df.columns for c in ("Open", "High", "Low", "Close")):
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])

@st.cache_data(ttl=120, show_spinner=False)
def fetch_history(symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, str]:
    if not symbol:
        return pd.DataFrame(), "No symbol selected."
    try:
        frame = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        df = _clean(frame)
        return (df, "") if not df.empty else (pd.DataFrame(), f"No data for {symbol}.")
    except Exception as exc:
        log_exception(f"history {symbol}", exc)
        return pd.DataFrame(), safe_error(exc)

@st.cache_data(ttl=60, show_spinner=False)
def get_quote(symbol: str) -> Dict[str, Any]:
    q = {"symbol": symbol, "ok": False, "price": None, "change_pct": None, "currency": guess_currency(symbol)}
    df, err = fetch_history(symbol, "5d", "1d")
    if df.empty:
        q["error"] = err
        return q
    last = df.iloc[-1]
    price = valid_price(last["Close"])
    if price is None:
        return q
    prev = valid_price(df["Close"].iloc[-2]) if len(df) > 1 else None
    q.update({
        "ok": True, "price": price, "prev_close": prev,
        "change": (price - prev) if prev else 0.0,
        "change_pct": ((price - prev) / prev * 100) if prev else 0.0,
        "open": valid_price(last["Open"]), "high": valid_price(last["High"]),
        "low": valid_price(last["Low"]), "volume": _num(last["Volume"]),
        "last_bar": str(df.index[-1])[:16],
    })
    return q

@st.cache_data(ttl=120, show_spinner=False)
def fetch_quotes_batch(symbols: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    out = {}
    syms = tuple(dict.fromkeys(s for s in symbols if s))
    if not syms:
        return out
    try:
        raw = yf.download(
            list(syms), period="5d", interval="1d", group_by="ticker",
            auto_adjust=False, progress=False, threads=True, timeout=20
        )
    except Exception as exc:
        log_exception("batch download", exc)
        raw = pd.DataFrame()

    for s in syms:
        row = {"symbol": s, "ok": False, "price": None, "change_pct": None, "currency": guess_currency(s), "name": display_name(s)}
        try:
            df = _clean(raw.xs(s, axis=1, level=0)) if len(syms) > 1 and isinstance(raw.columns, pd.MultiIndex) and s in raw.columns.levels[0] else _clean(raw)
            if not df.empty and valid_price(df["Close"].iloc[-1]):
                p = float(df["Close"].iloc[-1])
                prv = float(df["Close"].iloc[-2]) if len(df) > 1 else p
                row.update({"ok": True, "price": p, "volume": _num(df["Volume"].iloc[-1]), "change_pct": ((p - prv) / prv * 100) if prv else 0.0})
        except Exception:
            pass
        out[s] = row
    return out

# ---------------------------------------------------------------------------
# GANN & GOLDEN POCKET
# ---------------------------------------------------------------------------
GANN_DEGREES = (22.5, 45.0, 67.5, 90.0, 180.0)

def compute_gann_ladder(anchor: Optional[float]) -> List[Dict[str, Any]]:
    a = valid_price(anchor)
    if not a:
        return []
    root = math.sqrt(a)
    ladder = [{"label": "0° Anchor", "degree": 0.0, "side": "0", "value": a}]
    for d in GANN_DEGREES:
        ladder.append({"label": f"R {d:g}°", "degree": d, "side": "R", "value": (root + d / 180.0) ** 2})
        base = root - d / 180.0
        if base > 0:
            ladder.append({"label": f"S {d:g}°", "degree": d, "side": "S", "value": base ** 2})
    ladder.sort(key=lambda x: x["value"])
    return ladder

def compute_golden_pocket(df: pd.DataFrame, length: int = 7, live_price: Optional[float] = None) -> Dict[str, Any]:
    res = {"active": False, "reason": "No confirmed pivot swings detected."}
    if df.empty or len(df) < (2 * length + 3):
        return res
    highs, lows = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    n = len(highs)
    ph, pl = [], []
    for i in range(length, n - length):
        if highs[i] == highs[i - length:i + length + 1].max():
            ph.append(i)
        if lows[i] == lows[i - length:i + length + 1].min():
            pl.append(i)
    bull = [(a, b) for b in reversed(ph) for a in pl if a < b and lows[a] < highs[b]]
    bear = [(a, b) for b in reversed(pl) for a in ph if a < b and highs[a] > lows[b]]
    if not bull and not bear:
        return res
    is_bull = not bear or (bull and bull[0][1] >= bear[0][1])
    a_idx, b_idx = bull[0] if is_bull else bear[0]
    a_p = float(lows[a_idx] if is_bull else highs[a_idx])
    b_p = float(highs[b_idx] if is_bull else lows[b_idx])
    rng = abs(b_p - a_p)
    f50 = b_p - (0.50 * rng) if is_bull else b_p + (0.50 * rng)
    f618 = b_p - (0.618 * rng) if is_bull else b_p + (0.618 * rng)
    top, btm = max(f50, f618), min(f50, f618)
    p = valid_price(live_price) or float(df["Close"].iloc[-1])
    status = "Inside Golden Pocket" if btm <= p <= top else ("Above Golden Pocket" if p > top else "Below Golden Pocket")
    return {
        "active": True, "direction": "Bullish Retracement" if is_bull else "Bearish Retracement",
        "swing_a": a_p, "swing_b": b_p, "fib_500": f50, "fib_618": f618,
        "zone_top": top, "zone_bottom": btm, "price_status": status,
    }

# ---------------------------------------------------------------------------
# NEWS & AI ASSISTANT
# ---------------------------------------------------------------------------
BULL_WORDS = ("surge", "rally", "beat", "growth", "record", "profit", "gain", "strong", "upgrade", "rebound")
BEAR_WORDS = ("fall", "drop", "loss", "miss", "weak", "decline", "crash", "fear", "downgrade", "plunge", "cut")

def news_sentiment(title: str) -> Tuple[str, str, float]:
    low = (title or "").lower()
    score = 5.0 + sum(0.8 for w in BULL_WORDS if w in low) - sum(0.8 for w in BEAR_WORDS if w in low)
    score = max(1.0, min(9.9, round(score, 1)))
    label = "Bullish" if score >= 6.2 else ("Bearish" if score <= 4.2 else "Neutral")
    color = "#34d399" if label == "Bullish" else ("#f87171" if label == "Bearish" else "#fbbf24")
    return label, color, score

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_news(symbol: str) -> List[Dict[str, Any]]:
    items = []
    try:
        raw = yf.Ticker(symbol).news or []
        for entry in raw:
            c = entry.get("content") if isinstance(entry.get("content"), dict) else {}
            title = entry.get("title") or c.get("title") or ""
            link = entry.get("link") or c.get("canonicalUrl", {}).get("url") or ""
            if title:
                items.append({
                    "title": html.unescape(title.strip()),
                    "publisher": entry.get("publisher") or c.get("provider", {}).get("displayName") or "Market Desk",
                    "link": link if isinstance(link, str) and link.startswith("http") else "",
                })
    except Exception:
        pass
    return items[:10]

def call_groq(prompt: str, system: str = "") -> str:
    k = groq_key()
    if not k or Groq is None:
        return "Groq AI key not configured in Secrets."
    try:
        client = Groq(api_key=k)
        model = AI_MODEL_MAP[DEFAULT_AI_LABEL]["model"]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}] if system else [{"role": "user", "content": prompt}]
        res = client.chat.completions.create(model=model, messages=msgs, max_tokens=350, temperature=0.3)
        return res.choices[0].message.content.strip()
    except Exception as exc:
        return f"AI Error: {safe_error(exc)}"

# ---------------------------------------------------------------------------
# PLOTLY CHART
# ---------------------------------------------------------------------------
def build_price_chart(df: pd.DataFrame, ladder: List[Dict[str, Any]], price: Optional[float], cur: str):
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#34d399", decreasing_line_color="#f87171", name="OHLC",
    ))
    for r in ladder:
        clr = "#fbbf24" if r["side"] == "0" else ("#f87171" if r["side"] == "R" else "#34d399")
        fig.add_hline(y=r["value"], line_dash="dot", line_color=clr, line_width=1, annotation_text=r["label"], annotation_position="right")
    if price:
        fig.add_hline(y=price, line_color="#60a5fa", line_width=1.5, annotation_text=f"Live {fmt_price(price, cur)}")
    fig.update_layout(
        height=400, paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
        font=dict(color="#94a3b8", size=10), yaxis=dict(side="right", gridcolor="rgba(51,65,85,.3)"),
    )
    return fig

# ---------------------------------------------------------------------------
# APP INITIALIZATION
# ---------------------------------------------------------------------------
alert_engine = AlertEngine(STATE_FILE)

def init_state() -> None:
    s = st.session_state
    saved_wl, saved_alerts = alert_engine.load_state()
    defaults = {
        "active_market": M_IN, "active_exchange": "NSE", "active_universe": "NIFTY 50",
        "active_symbol": "RELIANCE.NS", "watchlist": saved_wl or ["RELIANCE.NS", "AAPL", "BTC-USD"],
        "alerts": saved_alerts, "page": "Live Analysis", "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in s:
            s[k] = v

init_state()

# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------
def page_live() -> None:
    s = st.session_state
    sym = s.active_symbol
    q = get_quote(sym)
    cur = q.get("currency") or "USD"
    sess = market_session(s.active_market)

    with st.container(border=True):
        c1, c2 = st.columns([3, 1.5])
        with c1:
            st.markdown(f"<div class='card-header'>{display_name(sym)} <span style='color:#64748b'>({sym})</span></div>", unsafe_allow_html=True)
            st.markdown(f"{status_pill(sess['status'])} <span style='color:#94a3b8;font-size:0.8rem'>{sess['stamp']} · {cur}</span>", unsafe_allow_html=True)
        with c2:
            st.markdown(metric_html("Price", fmt_price(q["price"], cur), cls=change_class(q.get("change_pct")), sub=fmt_pct(q.get("change_pct"))), unsafe_allow_html=True)

    df, _ = fetch_history(sym, "5d", "15m")
    anchor_p = q.get("open") or (df["Close"].iloc[0] if not df.empty else q.get("price"))
    ladder = compute_gann_ladder(anchor_p)
    gp = compute_golden_pocket(df, live_price=q["price"])

    t_chart, t_gann, t_pocket, t_plan = st.tabs(["Chart", "Gann Levels", "Golden Pocket", "Scenario"])
    with t_chart:
        if not df.empty and q.get("ok"):
            show_fig(build_price_chart(df, ladder, q["price"], cur))
        else:
            st.info("Chart data loading...")

    with t_gann:
        st.caption(f"0° Reference Anchor: {fmt_price(anchor_p, cur)}")
        p = q.get("price") or 0.0
        for r in sorted(ladder, key=lambda x: x["value"], reverse=True):
            cls = "level-anchor" if r["side"] == "0" else ("level-resistance" if r["side"] == "R" else "level-support")
            diff = ((r["value"] - p) / p) * 100 if p else 0
            st.markdown(f"<div class='level-row {cls}'><span><b>{r['label']}</b> &nbsp; {fmt_price(r['value'], cur)}</span>"
                        f"<span class='{change_class(diff)}'>{diff:+.2f}%</span></div>", unsafe_allow_html=True)

    with t_pocket:
        if gp["active"]:
            st.markdown(f"**Swing:** {gp['direction']} — <span class='pill pill-open'>{gp['price_status']}</span>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(metric_html("50.0% Fib", fmt_price(gp["fib_500"], cur)), unsafe_allow_html=True)
            c2.markdown(metric_html("61.8% Fib", fmt_price(gp["fib_618"], cur)), unsafe_allow_html=True)
        else:
            st.info(gp["reason"])

    with t_plan:
        if q.get("ok") and len(ladder) >= 4:
            sup = [r["value"] for r in ladder if r["value"] < q["price"] and r["side"] != "0"]
            res = [r["value"] for r in ladder if r["value"] > q["price"] and r["side"] != "0"]
            if sup and res:
                st.markdown(f"""
                <div class='plan-card plan-bull'>
                    <b class='up'>▲ Bullish Breakout Idea</b>
                    <div>Entry: <b>{fmt_price(sup[-1], cur)}</b> | Invalidation: <b>{fmt_price(sup[0], cur)}</b></div>
                    <div>Target: <b>{fmt_price(res[0], cur)}</b></div>
                </div>
                <div class='plan-card plan-bear'>
                    <b class='down'>▼ Bearish Rejection Idea</b>
                    <div>Entry: <b>{fmt_price(res[0], cur)}</b> | Invalidation: <b>{fmt_price(res[-1], cur)}</b></div>
                    <div>Target: <b>{fmt_price(sup[-1], cur)}</b></div>
                </div>
                """, unsafe_allow_html=True)

def page_heatmap() -> None:
    s = st.session_state
    sub = MARKET_TREE.get(s.active_market, {}).get(s.active_exchange, {})
    pool = []
    for _, items in sub.items():
        pool.extend(items)
    if not pool:
        pool = [("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia")]

    with st.spinner("Loading heatmap..."):
        batch = fetch_quotes_batch(tuple(sym for sym, _ in pool))
        rows = []
        for sym, name in pool:
            q = batch.get(sym)
            if q and q.get("ok"):
                rows.append({
                    "symbol": sym, "name": name, "price": q["price"],
                    "change_percent": q["change_pct"], "currency": q["currency"],
                })
    if rows:
        frame = pd.DataFrame(rows)
        fig = build_heatmap_figure(frame, is_mobile=True)
        if fig:
            show_fig(fig)
    else:
        st.info("Heatmap feed empty.")

def page_news() -> None:
    s = st.session_state
    st.markdown(f"<div class='card-header'>Live News: {s.active_symbol}</div>", unsafe_allow_html=True)
    items = fetch_stock_news(s.active_symbol)
    if not items:
        st.info("No headlines returned by provider.")
        return
    for idx, it in enumerate(items):
        lbl, colr, scr = news_sentiment(it["title"])
        with st.container(border=True):
            st.markdown(f"<div style='display:flex;justify-content:space-between;'>"
                        f"<span style='font-weight:600'>{it['title']}</span>"
                        f"<span class='pill' style='color:{colr}'>{lbl} ({scr})</span></div>", unsafe_allow_html=True)
            st.caption(f"Source: {it['publisher']}")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Explain Impact", key=f"n_{idx}"):
                    st.info(call_groq(
                        f"Headline: {it['title']}\nSymbol: {s.active_symbol}",
                        system="Provide 2 short sentences on how this headline impacts the stock."
                    ))
            with c2:
                if it["link"]:
                    st.markdown(f"<a href='{it['link']}' target='_blank' style='color:#60a5fa;font-size:0.85rem'>Read Source ↗</a>", unsafe_allow_html=True)

def page_alerts() -> None:
    s = st.session_state
    with st.container(border=True):
        st.markdown("<div class='card-header'>Set Price Alert</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 1.5, 1])
        sym = c1.text_input("Symbol", value=s.active_symbol).upper()
        target = c2.number_input("Target", min_value=0.0, format="%.2f")
        cond = c3.selectbox("Direction", ["Above", "Below"])
        if st.button("Add Alert"):
            ok, msg = AlertEngine.add(s.alerts, sym, target, cond)
            if ok:
                alert_engine.save_state(s.watchlist, s.alerts)
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

    if s.alerts:
        st.markdown("<div class='card-header'>Active Monitors</div>", unsafe_allow_html=True)
        for a in list(s.alerts):
            with st.container(border=True):
                r1, r2 = st.columns([4, 1])
                r1.markdown(f"**{a['symbol']}** {a['condition']} **{a['target']}**")
                if r2.button("Remove", key=f"del_{a['id']}"):
                    s.alerts = [x for x in s.alerts if x["id"] != a["id"]]
                    alert_engine.save_state(s.watchlist, s.alerts)
                    st.rerun()

def page_watchlist() -> None:
    s = st.session_state
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        new_sym = c1.text_input("Add to Watchlist", placeholder="e.g. INFY.NS, TSLA")
        if c2.button("Add") and new_sym:
            clean = new_sym.strip().upper()
            if clean not in s.watchlist:
                s.watchlist.append(clean)
                alert_engine.save_state(s.watchlist, s.alerts)
                st.rerun()

    quotes = fetch_quotes_batch(tuple(s.watchlist))
    for sym in list(s.watchlist):
        q = quotes.get(sym, {})
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.markdown(f"<b>{display_name(sym)}</b><br><span style='color:#64748b;font-size:0.75rem'>{sym}</span>", unsafe_allow_html=True)
            c2.markdown(metric_html(
                "Price", fmt_price(q.get("price"), q.get("currency", "USD")),
                cls=change_class(q.get("change_pct")), sub=fmt_pct(q.get("change_pct"))
            ), unsafe_allow_html=True)
            with c3:
                if st.button("Select", key=f"act_{sym}"):
                    s.active_symbol = sym
                    st.rerun()
                if st.button("🗑", key=f"wdel_{sym}"):
                    s.watchlist = [x for x in s.watchlist if x != sym]
                    alert_engine.save_state(s.watchlist, s.alerts)
                    st.rerun()

def page_ai() -> None:
    s = st.session_state
    st.markdown(f"<div class='card-header'>AI Terminal Chat ({s.active_symbol})</div>", unsafe_allow_html=True)
    for m in s.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    user_p = st.chat_input("Ask about market trends or active levels...")
    if user_p:
        s.chat_history.append({"role": "user", "content": user_p})
        with st.chat_message("user"):
            st.markdown(user_p)
        with st.chat_message("assistant"):
            q = get_quote(s.active_symbol)
            ctx = f"Asset: {s.active_symbol}, Price: {q.get('price')} {q.get('currency')}."
            reply = call_groq(user_p, system=f"Educational market analyzer. Context: {ctx}")
            st.markdown(reply)
            s.chat_history.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    s = st.session_state

    with st.sidebar:
        st.markdown("## ⚡ Terminal Control")
        s.active_market = st.selectbox("Market", list(MARKET_TREE.keys()))
        s.active_exchange = st.selectbox("Exchange", list(MARKET_TREE[s.active_market].keys()))
        s.active_universe = st.selectbox("Universe", list(MARKET_TREE[s.active_market][s.active_exchange].keys()))

        presets = MARKET_TREE[s.active_market][s.active_exchange][s.active_universe]
        options = [f"{sym} — {nm}" for sym, nm in presets]
        if options:
            ch = st.selectbox("Asset", options)
            s.active_symbol = presets[options.index(ch)][0]
        custom = st.text_input("Override Symbol")
        if custom:
            s.active_symbol = normalize_symbol(custom, s.active_market, s.active_exchange)

    # Evaluate alerts in the background
    if s.alerts:
        needed = tuple({a["symbol"] for a in s.alerts})
        quotes_map = fetch_quotes_batch(needed)
        fired = alert_engine.evaluate_all(
            s.alerts,
            lambda sym: quotes_map.get(sym, {}).get("price"),
            notifier=send_telegram if telegram_ready() else None,
        )
        for f in fired:
            st.toast(f)

    PAGES = ["Live Analysis", "Heatmap", "News", "Watchlist", "Alerts", "AI Chat"]
    s.page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="nav")

    router = {
        "Live Analysis": page_live,
        "Heatmap": page_heatmap,
        "News": page_news,
        "Watchlist": page_watchlist,
        "Alerts": page_alerts,
        "AI Chat": page_ai,
    }
    router[s.page]()

if __name__ == "__main__":
    main()
