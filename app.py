"""
AI Trade Terminal - Streamlit Edition (v4.0.0)
==============================================
Multi-market trading terminal: Live Analysis (Gann Square-of-9 + Golden Pocket), dynamic
Market -> Exchange -> Universe -> Symbol selector with ONE global active symbol, market-following
heat map, global RSS news, alerts (optional Telegram), watchlist, history and a Groq AI assistant.

Secrets (Streamlit -> Settings -> Secrets):  GROQ_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Data: Yahoo Finance via yfinance (delayed / not guaranteed real-time). Educational use only.
"""
from __future__ import annotations

import concurrent.futures as futures
import html
import json
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

try:  # optional at import time; a clear UI message is shown if it is missing
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

APP_VERSION = "4.0.0"
HTTP_UA = "Mozilla/5.0 (compatible; AI-Trade-Terminal/4.0; +https://streamlit.io)"
APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".cache"
STATE_FILE = CACHE_DIR / "terminal_state.json"

# ---------------------------------------------------------------------------
# AI MODEL MAPPING - one place, easy to change. Only the verified model is exposed.
# ---------------------------------------------------------------------------
AI_MODEL_MAP: Dict[str, Dict[str, str]] = {
    "Groq — GPT-OSS 20B": {"provider": "groq", "model": "openai/gpt-oss-20b"},
}
DEFAULT_AI_LABEL = "Groq — GPT-OSS 20B"

LOG = logging.getLogger("ai_trade_terminal")
if not LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(_h)
LOG.setLevel(logging.INFO)

st.set_page_config(page_title="AI Trade Terminal", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

_CSS = """
<style>
.stApp { background-color:#020617 !important; color:#f8fafc; }
#MainMenu, footer { visibility:hidden; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#0F172A; border:1px solid rgba(51,65,85,.45) !important; border-radius:14px !important; }
div[data-testid="stExpander"] { background:#0F172A; border:1px solid rgba(51,65,85,.45); border-radius:12px; }
section[data-testid="stSidebar"] { background-color:#0F172A !important; border-right:1px solid rgba(51,65,85,.4); }
.card-header { font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:.1rem; }
.card-sub { font-size:.75rem; color:#64748b; margin-bottom:.5rem; }
.metric-label { font-size:.65rem; font-weight:500; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }
.metric-value { font-size:1.3rem; font-weight:700; color:#f8fafc; font-variant-numeric:tabular-nums; }
.up { color:#34d399 !important; } .down { color:#f87171 !important; } .flat { color:#94a3b8 !important; } .amber { color:#fbbf24 !important; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.68rem; font-weight:700; }
.pill-open { background:rgba(16,185,129,.18); color:#34d399; border:1px solid rgba(16,185,129,.4); }
.pill-closed { background:rgba(148,163,184,.16); color:#cbd5e1; border:1px solid rgba(148,163,184,.35); }
.pill-delayed { background:rgba(96,165,250,.16); color:#93c5fd; border:1px solid rgba(96,165,250,.4); }
.pill-last { background:rgba(167,139,250,.16); color:#c4b5fd; border:1px solid rgba(167,139,250,.4); }
.pill-bad { background:rgba(248,113,113,.16); color:#fca5a5; border:1px solid rgba(248,113,113,.4); }
.stButton > button { border-radius:10px !important; font-weight:600 !important; border:1px solid rgba(51,65,85,.55) !important; background:#1E293B !important; color:#e2e8f0 !important; }
.stButton > button:hover { border-color:#fbbf24 !important; color:#fbbf24 !important; }
.stTabs [data-baseweb="tab"] { color:#94a3b8; }
.level-row { display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-radius:9px; margin-bottom:4px; font-size:.9rem; gap:8px; flex-wrap:wrap; }
.level-support { background:rgba(16,185,129,.08); border:1px solid rgba(16,185,129,.2); }
.level-resistance { background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.2); }
.level-anchor { background:rgba(251,191,36,.14); border:1px solid rgba(251,191,36,.4); }
.level-current { background:rgba(96,165,250,.14); border:1px solid rgba(96,165,250,.4); }
.gp-box { background:rgba(251,146,60,.10); border:1px solid rgba(251,146,60,.45); border-radius:12px; padding:10px 14px; margin-bottom:8px; }
.plan-card { border-radius:12px; padding:12px 14px; margin-bottom:10px; }
.plan-bull { background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35); }
.plan-bear { background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35); }
.news-card { background:#0B1220; border:1px solid rgba(51,65,85,.5); border-radius:12px; padding:10px 14px; margin-bottom:8px; }
@media (max-width:700px) {
  .metric-value { font-size:1.0rem; } .card-header { font-size:.98rem; } .level-row { font-size:.8rem; }
}
</style>
"""


# ===========================================================================
# SECTION 1 - SECRETS, LOGGING, SAFE ERRORS
# ===========================================================================
def secret(name: str, default: str = "") -> str:
    """Streamlit secrets first, then environment. Never raises."""
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name, default) or default).strip()


def groq_key() -> str:
    return secret("GROQ_KEY")


def telegram_ready() -> bool:
    return bool(secret("TELEGRAM_BOT_TOKEN") and secret("TELEGRAM_CHAT_ID"))


def _redact(text: Any) -> str:
    out = str(text)
    for name in ("GROQ_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        value = secret(name)
        if value and len(value) >= 6:
            out = out.replace(value, "***")
    out = re.sub(r"gsk_[A-Za-z0-9]+", "gsk_***", out)
    out = re.sub(r"bot\d+:[A-Za-z0-9_\-]+", "bot***", out)
    return out


def safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {_redact(exc)}"[:400]


def log_exception(context: str, exc: BaseException) -> None:
    LOG.warning("%s | %s", context, safe_error(exc))


# ===========================================================================
# SECTION 2 - NUMBERS, CURRENCY, FORMATTING
# ===========================================================================
CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "HKD": "HK$", "AUD": "A$",
    "CAD": "C$", "SGD": "S$", "KRW": "₩", "CNY": "¥", "CHF": "CHF ", "NZD": "NZ$", "MXN": "MX$",
    "ZAR": "R", "BRL": "R$", "TRY": "₺",
}
SUFFIX_CURRENCY = {".NS": "INR", ".BO": "INR", ".L": "GBP", ".DE": "EUR", ".PA": "EUR", ".T": "JPY",
                   ".HK": "HKD", ".AX": "AUD", ".TO": "CAD", ".SI": "SGD", ".KS": "KRW"}
INDEX_CURRENCY = {"^NSEI": "INR", "^NSEBANK": "INR", "^BSESN": "INR", "^CNXIT": "INR", "^CNXAUTO": "INR",
                  "^CNXFMCG": "INR", "^CNXPHARMA": "INR", "^CNXMETAL": "INR", "^FTSE": "GBP",
                  "^GDAXI": "EUR", "^FCHI": "EUR", "^STOXX50E": "EUR", "^N225": "JPY", "^HSI": "HKD",
                  "^AXJO": "AUD", "^KS11": "KRW", "^STI": "SGD"}


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", "").strip()) if isinstance(value, str) else float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def valid_price(value: Any) -> Optional[float]:
    number = _num(value)
    return number if number is not None and number > 0 else None


def guess_currency(symbol: str) -> str:
    raw = (symbol or "").upper()
    if raw in INDEX_CURRENCY:
        return INDEX_CURRENCY[raw]
    if raw.startswith("^") or raw.endswith("-USD") or raw.endswith("=F"):
        return "USD"
    if raw.endswith("=X"):
        pair = raw[:-2]
        tail = pair[-3:] if len(pair) >= 6 else "USD"
        return tail if tail in CURRENCY_SYMBOLS else "USD"
    for suffix, cur in SUFFIX_CURRENCY.items():
        if raw.endswith(suffix):
            return cur
    return "USD"


def _group_indian(digits: str) -> str:
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts: List[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def fmt_price(value: Any, currency: str = "USD", decimals: Optional[int] = None) -> str:
    """Price in its own currency (INR uses Indian grouping). Non-positive / missing -> dash."""
    number = _num(value)
    if number is None or number <= 0:
        return "—"
    currency = (currency or "USD").upper()
    if decimals is None:
        decimals = 4 if number < 10 else 2
    if currency == "INR":
        int_part, _, frac = f"{number:.{decimals}f}".partition(".")
        return "₹" + _group_indian(int_part) + (f".{frac}" if frac else "")
    return f"{CURRENCY_SYMBOLS.get(currency, currency + ' ')}{number:,.{decimals}f}"


def fmt_delta(value: Any, currency: str = "USD") -> str:
    number = _num(value)
    if number is None:
        return "—"
    if abs(number) > 0:
        body = fmt_price(abs(number), currency)
    else:
        body = f"{CURRENCY_SYMBOLS.get((currency or 'USD').upper(), '')}0.00"
    return f"{'+' if number >= 0 else '−'}{body}"


def fmt_pct(value: Any, decimals: int = 2) -> str:
    number = _num(value)
    return "—" if number is None else f"{'+' if number >= 0 else ''}{number:.{decimals}f}%"


def fmt_volume(value: Any) -> str:
    number = _num(value)
    if number is None or number < 0:
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if number >= scale:
            return f"{number / scale:.2f}{suffix}"
    return f"{number:,.0f}"


def change_class(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "flat"
    return "up" if number > 0 else ("down" if number < 0 else "flat")


def metric_html(label: str, value: str, cls: str = "", sub: str = "") -> str:
    sub_html = f"<div class='card-sub' style='margin:2px 0 0 0'>{sub}</div>" if sub else ""
    return (f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}'>{value}</div>{sub_html}")


# --- Streamlit API compatibility: use_container_width was deprecated in newer releases ---
def _stretch(fn, *args, **kwargs):
    last: Optional[BaseException] = None
    for extra in ({"use_container_width": True}, {"width": "stretch"}, {}):
        try:
            return fn(*args, **kwargs, **extra)
        except (TypeError, ValueError, st.errors.StreamlitAPIException) as exc:
            last = exc
    raise last  # type: ignore[misc]


def show_df(df: pd.DataFrame, **kw: Any) -> None:
    _stretch(st.dataframe, df, hide_index=True, **kw)


def show_fig(fig: Any, **kw: Any) -> None:
    _stretch(st.plotly_chart, fig, **kw)


def wbutton(label: str, **kw: Any) -> bool:
    return bool(_stretch(st.button, label, **kw))


# ===========================================================================
# SECTION 3 - MARKET CATALOG  (Market -> Exchange -> Universe -> symbols)
# ===========================================================================
NAMES: Dict[str, str] = {
    "ES=F": "E-mini S&P 500", "NQ=F": "Nasdaq 100 E-mini", "YM=F": "Dow E-mini", "RTY=F": "Russell 2000 E-mini",
    "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper", "PL=F": "Platinum", "PA=F": "Palladium",
    "CL=F": "Crude Oil WTI", "BZ=F": "Brent Crude", "NG=F": "Natural Gas", "HO=F": "Heating Oil",
    "RB=F": "RBOB Gasoline", "ZC=F": "Corn", "ZW=F": "Wheat", "ZS=F": "Soybeans", "KC=F": "Coffee",
    "SB=F": "Sugar", "CT=F": "Cotton",
    "^NSEI": "NIFTY 50", "^NSEBANK": "NIFTY Bank", "^BSESN": "SENSEX", "^CNXIT": "NIFTY IT",
    "^CNXAUTO": "NIFTY Auto", "^CNXFMCG": "NIFTY FMCG", "^CNXPHARMA": "NIFTY Pharma", "^CNXMETAL": "NIFTY Metal",
    "^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
    "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40", "^STOXX50E": "Euro Stoxx 50",
    "^N225": "Nikkei 225", "^HSI": "Hang Seng", "^AXJO": "ASX 200", "^KS11": "KOSPI", "^STI": "Straits Times",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "BNB", "SOL-USD": "Solana", "XRP-USD": "XRP",
    "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "TRX-USD": "TRON", "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink", "DOT-USD": "Polkadot", "NEAR-USD": "NEAR", "ATOM-USD": "Cosmos",
    "AAVE-USD": "Aave", "MKR-USD": "Maker", "CRV-USD": "Curve", "LDO-USD": "Lido DAO", "SNX-USD": "Synthetix",
    "UNI7083-USD": "Uniswap", "SHIB-USD": "Shiba Inu", "FLOKI-USD": "Floki", "PEPE24478-USD": "Pepe",
    "XAUUSD=X": "Gold spot (XAU/USD)",
}


def display_name(symbol: str) -> str:
    raw = (symbol or "").upper()
    if raw in NAMES:
        return NAMES[raw]
    if raw.endswith("=X") and len(raw) >= 8:
        return f"{raw[:3]}/{raw[3:6]}"
    for suffix in (".NS", ".BO"):
        if raw.endswith(suffix):
            return raw[:-len(suffix)]
    return raw


def _p(symbols: List[str], suffix: str = "") -> List[Tuple[str, str]]:
    out = []
    for s in symbols:
        full = f"{s}{suffix}"
        out.append((full, display_name(full)))
    return out


N50 = ("RELIANCE TCS HDFCBANK ICICIBANK INFY SBIN BHARTIARTL ITC LT HINDUNILVR KOTAKBANK AXISBANK BAJFINANCE "
       "MARUTI ASIANPAINT HCLTECH WIPRO SUNPHARMA TITAN ULTRACEMCO POWERGRID NTPC TATAMOTORS TATASTEEL JSWSTEEL "
       "M&M ADANIENT ADANIPORTS COALINDIA ONGC GRASIM HINDALCO DRREDDY CIPLA DIVISLAB NESTLEIND BRITANNIA TECHM "
       "BAJAJFINSV BAJAJ-AUTO HEROMOTOCO EICHERMOT INDUSINDBK APOLLOHOSP BPCL SBILIFE HDFCLIFE TATACONSUM TRENT").split()
IN_GROUPS: Dict[str, List[str]] = {
    "NIFTY 50": N50,
    "NIFTY Bank": "HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK INDUSINDBK BANKBARODA PNB FEDERALBNK IDFCFIRSTB AUBANK".split(),
    "NIFTY IT": "TCS INFY HCLTECH WIPRO TECHM LTIM PERSISTENT COFORGE MPHASIS".split(),
    "NIFTY Auto": "MARUTI TATAMOTORS M&M BAJAJ-AUTO EICHERMOT HEROMOTOCO TVSMOTOR ASHOKLEY BOSCHLTD".split(),
    "NIFTY FMCG": "HINDUNILVR ITC NESTLEIND BRITANNIA TATACONSUM DABUR GODREJCP MARICO COLPAL".split(),
    "NIFTY Pharma": "SUNPHARMA DRREDDY CIPLA DIVISLAB LUPIN AUROPHARMA TORNTPHARM ZYDUSLIFE".split(),
    "NIFTY Metal": "TATASTEEL JSWSTEEL HINDALCO VEDL SAIL NMDC JINDALSTEL COALINDIA".split(),
}
US_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "NASDAQ": {
        "S&P 500 Technology": "AAPL MSFT NVDA GOOGL AMZN META AVGO AMD ADBE CSCO INTC QCOM".split(),
        "S&P 500 Banking": "FITB HBAN ZION".split(),
        "S&P 500 Healthcare": "AMGN GILD REGN VRTX ISRG MRNA".split(),
        "S&P 500 Consumer": "COST PEP SBUX MDLZ NFLX TSLA".split(),
        "S&P 500 Energy": "FANG".split(),
    },
    "NYSE": {
        "S&P 500 Technology": "ORCL CRM IBM NOW UBER".split(),
        "S&P 500 Banking": "JPM BAC WFC C GS MS USB PNC".split(),
        "S&P 500 Healthcare": "JNJ UNH LLY PFE MRK ABBV TMO ABT".split(),
        "S&P 500 Consumer": "WMT PG KO MCD NKE HD DIS".split(),
        "S&P 500 Energy": "XOM CVX COP SLB EOG OXY".split(),
    },
    "AMEX": {"S&P 500 Energy": "IMO UUUU".split()},
}
# (symbol, exchange, group)
FUTURES: List[Tuple[str, str, str]] = [
    ("ES=F", "CME", "Equity Index Futures"), ("NQ=F", "CME", "Equity Index Futures"),
    ("RTY=F", "CME", "Equity Index Futures"), ("YM=F", "CBOT", "Equity Index Futures"),
    ("GC=F", "COMEX", "Metals Futures"), ("SI=F", "COMEX", "Metals Futures"), ("HG=F", "COMEX", "Metals Futures"),
    ("CL=F", "NYMEX", "Energy Futures"), ("BZ=F", "NYMEX", "Energy Futures"), ("NG=F", "NYMEX", "Energy Futures"),
    ("ZC=F", "CBOT", "Agriculture Futures"), ("ZW=F", "CBOT", "Agriculture Futures"),
    ("ZS=F", "CBOT", "Agriculture Futures"),
]
FUTURES_GROUPS = ["Equity Index Futures", "Metals Futures", "Energy Futures", "Agriculture Futures"]
INDIA_INDEX_SET = {"^NSEI", "^NSEBANK", "^BSESN", "^CNXIT", "^CNXAUTO", "^CNXFMCG", "^CNXPHARMA", "^CNXMETAL"}

M_IN, M_US, M_CR, M_FX, M_CM, M_FU, M_GI, M_OTC = ("India", "United States", "Cryptocurrency", "Forex",
                                                    "Commodities", "Futures", "Global Indices",
                                                    "OTC / CFD / Synthetic")


def build_tree() -> Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]]:
    """market -> exchange -> universe -> [(yahoo symbol, name)].  An EMPTY list marks a custom universe."""
    tree: Dict[str, Dict[str, Dict[str, List[Tuple[str, str]]]]] = {}
    custom = "All / Custom Symbol"
    tree[M_IN] = {
        "NSE": {**{g: _p(s, ".NS") for g, s in IN_GROUPS.items()}, custom: []},
        "BSE": {**{g: _p(s, ".BO") for g, s in IN_GROUPS.items()}, custom: []},
        "Indices": {
            "Benchmark Indices": _p(["^NSEI", "^NSEBANK", "^BSESN"]),
            "Sectoral Indices": _p(["^CNXIT", "^CNXAUTO", "^CNXFMCG", "^CNXPHARMA", "^CNXMETAL"]),
            custom: [],
        },
    }
    tree[M_US] = {ex: {**{g: _p(s) for g, s in groups.items()}, custom: []} for ex, groups in US_GROUPS.items()}
    tree[M_CR] = {"Yahoo Finance / USD Pairs": {
        "Top Crypto": _p("BTC ETH BNB SOL XRP ADA DOGE TRX AVAX LINK".split(), "-USD"),
        "Layer 1": _p("ETH SOL ADA AVAX DOT NEAR ATOM".split(), "-USD"),
        "DeFi": _p("AAVE MKR CRV LDO SNX UNI7083".split(), "-USD"),
        "Memes": _p("DOGE SHIB FLOKI PEPE24478".split(), "-USD"),
        "Custom Symbol": []}}
    tree[M_FX] = {"Yahoo Finance FX": {
        "Major Pairs": _p("EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD".split(), "=X"),
        "Cross Pairs": _p("EURGBP EURJPY GBPJPY AUDJPY EURCHF EURAUD".split(), "=X"),
        "Emerging Market Pairs": _p("USDINR USDMXN USDZAR USDTRY USDBRL USDSGD".split(), "=X"),
        "Custom Symbol": []}}
    tree[M_CM] = {"Futures / Yahoo Finance": {
        "Metals": _p("GC SI HG PL PA".split(), "=F"),
        "Energy": _p("CL BZ NG HO RB".split(), "=F"),
        "Agriculture": _p("ZC ZW ZS KC SB CT".split(), "=F"),
        "Custom Symbol": []}}
    fut: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    for ex in ("CME", "COMEX", "NYMEX", "CBOT"):
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for g in FUTURES_GROUPS:
            items = [(s, display_name(s)) for s, e, gg in FUTURES if e == ex and gg == g]
            if items:
                groups[g] = items
        groups["Custom Contract"] = []
        fut[ex] = groups
    tree[M_FU] = fut
    tree[M_GI] = {"Global Index": {
        "US Indices": _p(["^GSPC", "^IXIC", "^DJI", "^RUT"]),
        "Europe Indices": _p(["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E"]),
        "Asia Indices": _p(["^N225", "^HSI", "^AXJO", "^KS11", "^STI"]),
        "Custom Index": []}}
    tree[M_OTC] = {"Yahoo-Supported Proxy / Custom": {
        "Forex Proxies": _p(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X"]),
        "Commodity Proxies": _p(["GC=F", "SI=F", "CL=F"]),
        "Crypto Proxies": _p(["BTC-USD", "ETH-USD"]),
        "Custom Symbol": []}}
    return tree


MARKET_TREE = build_tree()
OTC_MODES = ["Standard OHLC", "OTC / CFD (User symbol)", "Synthetic session"]
ANCHOR_MODES = ["First available 15-minute candle close", "First available 15-minute candle open",
                "Daily session open", "Custom user-entered anchor"]
GP_TIMEFRAMES: Dict[str, Tuple[str, str]] = {
    "15 minute · 5 days": ("5d", "15m"), "1 hour · 3 months": ("3mo", "1h"), "Daily · 1 year": ("1y", "1d")}


@st.cache_data(ttl=3600, show_spinner=False)
def xau_spot_available() -> bool:
    """XAUUSD=X is offered only if Yahoo really returns data for it (otherwise GC=F is the gold proxy)."""
    try:
        frame = yf.Ticker("XAUUSD=X").history(period="5d", interval="1d", auto_adjust=False)
        return frame is not None and not frame.empty and bool(valid_price(frame["Close"].dropna().iloc[-1]))
    except Exception:
        return False


def universe_symbols(market: str, exchange: str, universe: str) -> List[Tuple[str, str]]:
    groups = MARKET_TREE.get(market, {}).get(exchange, {})
    items = list(groups.get(universe, []))
    if not items:  # custom universe: offer the first real group as a pick-list
        for label, lst in groups.items():
            if lst:
                items = list(lst)
                break
    if market == M_OTC and universe == "Commodity Proxies" and xau_spot_available():
        items = [("XAUUSD=X", NAMES["XAUUSD=X"])] + items
    return items


def is_custom_universe(market: str, exchange: str, universe: str) -> bool:
    return not MARKET_TREE.get(market, {}).get(exchange, {}).get(universe, [1])


def heatmap_groups(market: str, exchange: str) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for universe, lst in MARKET_TREE.get(market, {}).get(exchange, {}).items():
        if lst:
            out[universe] = universe_symbols(market, exchange, universe)
    return out


def normalize_symbol(raw: str, market: str, exchange: str = "") -> str:
    """Yahoo symbol normalisation that never overwrites an already-valid symbol."""
    s = (raw or "").strip().upper().replace(" ", "")
    if not s:
        return ""
    if market == M_OTC:
        return s
    if s.startswith("^") or s.endswith("=F") or s.endswith("=X"):
        return s
    if market == M_IN:
        if exchange == "Indices":
            alias = {"NIFTY": "^NSEI", "NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
            return alias.get(s, s if "." in s else "^" + s)
        if "." in s:
            return s
        return s + (".BO" if exchange == "BSE" else ".NS")
    if market == M_CR:
        if s.endswith("-USD"):
            return s
        s = s.replace("/", "")
        return (s[:-3] if s.endswith("USD") and len(s) > 3 else s) + "-USD"
    if market == M_FX:
        s = s.replace("/", "")
        if len(s) == 3:
            s = "USD" + s if s != "USD" else "EURUSD"
        return s + "=X"
    if market in (M_CM, M_FU):
        return s + "=F"
    if market == M_GI:
        return s if "." in s else "^" + s
    return s


def infer_market_context(symbol: str) -> Tuple[str, str, str]:
    """Which Market / Exchange / Universe a raw Yahoo symbol naturally belongs to."""
    s = (symbol or "").upper()
    if s.endswith(".NS"):
        return M_IN, "NSE", "All / Custom Symbol"
    if s.endswith(".BO"):
        return M_IN, "BSE", "All / Custom Symbol"
    if s.startswith("^"):
        if s in INDIA_INDEX_SET:
            return M_IN, "Indices", "All / Custom Symbol"
        return M_GI, "Global Index", "Custom Index"
    if s.endswith("-USD"):
        return M_CR, "Yahoo Finance / USD Pairs", "Custom Symbol"
    if s.endswith("=X"):
        return M_FX, "Yahoo Finance FX", "Custom Symbol"
    if s.endswith("=F"):
        for sym, ex, _g in FUTURES:
            if sym == s:
                return M_FU, ex, "Custom Contract"
        return M_CM, "Futures / Yahoo Finance", "Custom Symbol"
    return M_US, "NYSE", "All / Custom Symbol"


# ===========================================================================
# SECTION 4 - SESSIONS
# ===========================================================================
def _tz(name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    return timezone.utc


def market_session(market: str) -> Dict[str, Any]:
    """OPEN / CLOSED per market convention (exchange holiday calendars are not bundled)."""
    if market == M_IN:
        tz, o, c, kind = "Asia/Kolkata", (9, 15), (15, 30), "regular"
    elif market == M_US:
        tz, o, c, kind = "America/New_York", (9, 30), (16, 0), "regular"
    elif market == M_CR:
        tz, o, c, kind = "UTC", (0, 0), (23, 59), "continuous"
    else:
        tz, o, c, kind = "UTC", (0, 0), (23, 59), "24x5"
    now = datetime.now(_tz(tz))
    minutes = now.hour * 60 + now.minute
    wd = now.weekday()
    if kind == "continuous":
        status, reason = "OPEN", "Continuous 24/7 market."
    elif kind == "24x5":
        closed = wd == 5 or (wd == 4 and now.hour >= 22) or (wd == 6 and now.hour < 22)
        status, reason = ("CLOSED", "Weekend break of the 24x5 market.") if closed else ("OPEN", "24x5 session.")
    elif wd >= 5:
        status, reason = "CLOSED", "Weekend."
    elif o[0] * 60 + o[1] <= minutes <= c[0] * 60 + c[1]:
        status, reason = "OPEN", "Inside regular session hours."
    else:
        status, reason = "CLOSED", "Outside regular session hours."
    return {"status": status, "reason": reason, "is_open": status == "OPEN",
            "local_time": now.strftime("%a %d %b %H:%M") + f" ({now.tzname() or tz})"}


def status_pill(text: str) -> str:
    css = {"OPEN": "pill-open", "LIVE": "pill-open", "DELAYED": "pill-delayed", "LAST SESSION": "pill-last",
           "CLOSED": "pill-closed", "UNAVAILABLE": "pill-bad"}.get(text, "pill-closed")
    return f"<span class='pill {css}'>{text}</span>"


# ===========================================================================
# SECTION 5 - MARKET DATA (yfinance, cached, every call guarded)
# ===========================================================================
OHLCV = ("Open", "High", "Low", "Close", "Volume")


def _clean(frame: Any) -> pd.DataFrame:
    """Flatten whatever column layout yfinance returned into plain OHLCV columns."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in df.columns]
    rename = {c: str(c).strip().title() for c in df.columns}
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    if not all(c in df.columns for c in ("Open", "High", "Low", "Close")):
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[list(OHLCV)].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=120, show_spinner=False)
def fetch_history(symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, str]:
    """(bars, error). Never raises; the error string is safe to show."""
    if not symbol:
        return pd.DataFrame(), "No symbol selected."
    try:
        frame = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
    except Exception as exc:
        log_exception(f"history {symbol} {period}/{interval}", exc)
        return pd.DataFrame(), safe_error(exc)
    df = _clean(frame)
    if df.empty:
        return df, f"Yahoo Finance returned no {interval} bars for {symbol}."
    return df, ""


@st.cache_data(ttl=60, show_spinner=False)
def get_quote(symbol: str) -> Dict[str, Any]:
    """Latest quote for ONE symbol from daily bars. Missing data stays missing - nothing is invented."""
    q: Dict[str, Any] = {"symbol": symbol, "ok": False, "price": None, "prev_close": None, "change": None,
                         "change_pct": None, "open": None, "high": None, "low": None, "volume": None,
                         "currency": guess_currency(symbol), "name": display_name(symbol),
                         "last_bar": "", "error": ""}
    df, err = fetch_history(symbol, "5d", "1d")
    if df.empty:
        q["error"] = err or "No data."
        return q
    last = df.iloc[-1]
    price = valid_price(last["Close"])
    prev = valid_price(df["Close"].iloc[-2]) if len(df) > 1 else None
    if price is None:
        q["error"] = "Provider returned a zero/invalid price."
        return q
    q.update({"ok": True, "price": price, "prev_close": prev,
              "change": (price - prev) if prev else None,
              "change_pct": ((price - prev) / prev * 100) if prev else None,
              "open": valid_price(last["Open"]), "high": valid_price(last["High"]),
              "low": valid_price(last["Low"]), "volume": _num(last["Volume"]),
              "last_bar": str(df.index[-1])[:19]})
    return q


def _slice_symbol(frame: pd.DataFrame, symbol: str, many: bool) -> pd.DataFrame:
    """One symbol's bars from a batch download (handles single- and multi-symbol layouts)."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols = frame.columns
    if isinstance(cols, pd.MultiIndex):
        for level in range(cols.nlevels):
            if symbol in cols.get_level_values(level):
                try:
                    return _clean(frame.xs(symbol, axis=1, level=level))
                except Exception:
                    continue
        return pd.DataFrame()
    return pd.DataFrame() if many else _clean(frame)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_quotes_batch(symbols: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """One batched download for many symbols (fast path for heat map / watchlist / alerts)."""
    out: Dict[str, Dict[str, Any]] = {}
    symbols = tuple(dict.fromkeys(s for s in symbols if s))
    if not symbols:
        return out
    error = ""
    frame = pd.DataFrame()
    try:
        frame = yf.download(list(symbols), period="5d", interval="1d", group_by="ticker",
                            auto_adjust=False, progress=False, threads=True, timeout=25)
    except Exception as exc:
        log_exception("batch download", exc)
        error = safe_error(exc)
    for sym in symbols:
        row: Dict[str, Any] = {"symbol": sym, "ok": False, "price": None, "change_pct": None, "volume": None,
                               "currency": guess_currency(sym), "name": display_name(sym), "error": error}
        try:
            df = _slice_symbol(frame, sym, len(symbols) > 1)
            price = valid_price(df["Close"].iloc[-1]) if not df.empty else None
            if price is None:
                row["error"] = error or "No data returned by Yahoo Finance."
            else:
                prev = valid_price(df["Close"].iloc[-2]) if len(df) > 1 else None
                row.update({"ok": True, "price": price, "volume": _num(df["Volume"].iloc[-1]),
                            "change_pct": ((price - prev) / prev * 100) if prev else 0.0, "error": ""})
        except Exception as exc:
            row["error"] = safe_error(exc)
        out[sym] = row
    return out


def data_status(symbol: str, market: str) -> Tuple[str, str]:
    """LIVE only if the market is open AND the newest 15m bar is recent; never guessed upward."""
    intraday, _ = fetch_history(symbol, "5d", "15m")
    if intraday.empty:
        daily, _ = fetch_history(symbol, "5d", "1d")
        return ("LAST SESSION", str(daily.index[-1])[:10]) if not daily.empty else ("UNAVAILABLE", "")
    last_ts = intraday.index[-1]
    try:
        age = (datetime.now(timezone.utc) - last_ts.tz_convert("UTC").to_pydatetime()).total_seconds() / 60
    except Exception:
        age = 1e9
    stamp = str(last_ts)[:16]
    if market_session(market)["is_open"]:
        return ("LIVE", stamp) if age <= 20 else ("DELAYED", stamp)
    return "LAST SESSION", stamp


# ===========================================================================
# SECTION 6 - GLOBAL NEWS (scan every RSS source, dedupe, sort, cache 5+ min)
# ===========================================================================
NEWS_FEEDS: List[Tuple[str, str]] = [
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Finance", "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("The Guardian Business", "https://www.theguardian.com/uk/business/rss"),
    ("Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Livemint Markets", "https://www.livemint.com/rss/markets"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Investing.com", "https://www.investing.com/rss/news.rss"),
]
BULL_WORDS = ("surge", "rally", "beat", "growth", "record", "profit", "gain", "strong", "upgrade", "rebound",
              "jump", "soar", "boost", "rise", "optimism", "cut rates", "rate cut")
BEAR_WORDS = ("fall", "drop", "loss", "miss", "weak", "decline", "crash", "fear", "downgrade", "selloff",
              "plunge", "slump", "war", "default", "recession", "tariff", "sanction", "inflation", "layoff")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _parse_date(text: str) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    for parser in (parsedate_to_datetime, lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))):
        try:
            dt = parser(text)
            if dt is not None:
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _parse_feed(content: bytes, source: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(content)
    items: List[Dict[str, Any]] = []
    for node in root.iter():
        if _local(node.tag) not in ("item", "entry"):
            continue
        title = link = date = summary = ""
        for child in node:
            name = _local(child.tag)
            if name == "title":
                title = (child.text or "").strip()
            elif name == "link":
                link = (child.attrib.get("href") or child.text or "").strip()
            elif name in ("pubdate", "published", "updated", "date"):
                date = date or (child.text or "")
            elif name in ("description", "summary"):
                summary = re.sub(r"<[^>]+>", "", child.text or "").strip()[:300]
        if title:
            items.append({"title": html.unescape(title), "link": link if link.startswith("http") else "",
                          "source": source, "published": _parse_date(date), "summary": html.unescape(summary)})
    return items


def _fetch_feed(name: str, url: str) -> Tuple[str, List[Dict[str, Any]], str]:
    try:
        resp = requests.get(url, headers={"User-Agent": HTTP_UA, "Accept": "application/rss+xml, */*"}, timeout=8)
        resp.raise_for_status()
        return name, _parse_feed(resp.content[:3_000_000], name), ""
    except Exception as exc:
        return name, [], safe_error(exc)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_global_news() -> Dict[str, Any]:
    """Scans ALL feeds in parallel. One failing feed never breaks the others (or the app)."""
    items: List[Dict[str, Any]] = []
    status: List[Dict[str, Any]] = []
    try:
        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(_fetch_feed, n, u) for n, u in NEWS_FEEDS]
            for job in jobs:
                try:
                    name, got, err = job.result(timeout=25)
                except Exception as exc:
                    name, got, err = "unknown feed", [], safe_error(exc)
                items.extend(got)
                status.append({"source": name, "articles": len(got), "ok": not err, "error": err})
    except Exception as exc:
        log_exception("global news", exc)
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        key = re.sub(r"[^a-z0-9 ]", "", item["title"].lower())[:100]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    floor = datetime.min.replace(tzinfo=timezone.utc)
    unique.sort(key=lambda i: i["published"] or floor, reverse=True)
    return {"items": unique[:80], "status": status, "fetched_at": datetime.now(timezone.utc).isoformat()}


def age_label(published: Optional[datetime]) -> str:
    if not isinstance(published, datetime):
        return "time unknown"
    seconds = (datetime.now(timezone.utc) - published).total_seconds()
    if seconds < 0:
        return "just now"
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d ago"
    return published.strftime("%d %b %Y")


def news_sentiment(title: str) -> Tuple[str, str]:
    """Transparent keyword heuristic (label, colour)."""
    low = (title or "").lower()
    score = sum(w in low for w in BULL_WORDS) - sum(w in low for w in BEAR_WORDS)
    if score > 0:
        return "Bullish tone", "#34d399"
    if score < 0:
        return "Bearish tone", "#f87171"
    return "Neutral", "#fbbf24"


def headline_lines(limit: int = 8) -> List[str]:
    try:
        return [f"{i['title']} ({i['source']}, {age_label(i['published'])})"
                for i in fetch_global_news().get("items", [])[:limit]]
    except Exception as exc:
        log_exception("headline lines", exc)
        return []


# ===========================================================================
# SECTION 7 - GANN SQUARE-OF-9 (anchor modes, ladder, position)
# ===========================================================================
GANN_DEGREES: Tuple[float, ...] = (22.5, 45.0, 67.5, 90.0, 180.0)


def gann_levels(anchor: Optional[float]) -> Dict[str, Optional[float]]:
    """Unchanged Square-of-9 formula:
         Resistance = (sqrt(anchor) + degree / 180.0) ** 2
         Support    = (sqrt(anchor) - degree / 180.0) ** 2
    A support whose base would go negative (very low prices) is reported as unavailable instead of
    being squared into a misleading positive price."""
    a = valid_price(anchor)
    if a is None:
        return {}
    root = math.sqrt(a)
    levels: Dict[str, Optional[float]] = {"0": a}
    for d in GANN_DEGREES:
        levels[f"R{d:g}"] = (root + d / 180.0) ** 2
        base = root - d / 180.0
        levels[f"S{d:g}"] = base ** 2 if base > 0 else None
    return levels


def gann_ladder(levels: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    """Sorted (ascending) list: S180, S90, S67.5, S45, S22.5, 0°, R22.5, R45, R67.5, R90, R180."""
    if not levels:
        return []
    rows: List[Dict[str, Any]] = []
    for d in reversed(GANN_DEGREES):
        rows.append({"label": f"S {d:g}°", "degree": d, "side": "S", "value": levels.get(f"S{d:g}")})
    rows.append({"label": "0° Anchor", "degree": 0.0, "side": "0", "value": levels.get("0")})
    for d in GANN_DEGREES:
        rows.append({"label": f"R {d:g}°", "degree": d, "side": "R", "value": levels.get(f"R{d:g}")})
    rows = [r for r in rows if r["value"] is not None]
    rows.sort(key=lambda r: r["value"])
    return rows


def resolve_anchor(symbol: str, mode: str, custom_value: Any = None, utc_sessions: bool = False) -> Dict[str, Any]:
    """The 0° anchor. Every degree level is derived from this single price."""
    res: Dict[str, Any] = {"ok": False, "price": None, "type": mode, "time": "", "note": ""}
    if mode == ANCHOR_MODES[3]:
        price = valid_price(custom_value)
        res.update({"ok": price is not None, "price": price, "time": "user-entered",
                    "note": "" if price else "Enter a custom anchor price above zero."})
        return res
    if mode in ANCHOR_MODES[:2]:
        intraday, err = fetch_history(symbol, "5d", "15m")
        if not intraday.empty:
            try:
                idx = intraday.index
                if utc_sessions and getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert("UTC")
                dates = np.array([d for d in idx.date])
                day = intraday[dates == dates[-1]]
                first = day.iloc[0]
                price = valid_price(first["Close"] if mode == ANCHOR_MODES[0] else first["Open"])
                if price:
                    res.update({"ok": True, "price": price, "time": str(day.index[0])[:16],
                                "note": "Latest session's first 15-minute candle"
                                        + (" (UTC session)" if utc_sessions else "")})
                    return res
            except Exception as exc:
                log_exception("anchor 15m", exc)
        res["note"] = f"15-minute bars unavailable ({err or 'no data'}); using the latest daily bar instead."
    daily, derr = fetch_history(symbol, "10d", "1d")
    if daily.empty:
        res["note"] = (res["note"] + " " if res["note"] else "") + (derr or "No daily data.")
        return res
    row = daily.iloc[-1]
    use_open = mode in (ANCHOR_MODES[1], ANCHOR_MODES[2])
    price = valid_price(row["Open"] if use_open else row["Close"])
    res.update({"ok": price is not None, "price": price, "time": str(daily.index[-1])[:10],
                "note": (res["note"] + " " if res["note"] else "") +
                        ("Daily session open." if mode == ANCHOR_MODES[2] else "Fallback: latest daily bar.")})
    return res


def gann_position(price: Optional[float], ladder: List[Dict[str, Any]], currency: str = "USD") -> Dict[str, Any]:
    """Where the current price sits between the nearest two Gann levels."""
    out: Dict[str, Any] = {"text": "Current price unavailable.", "lower": None, "upper": None,
                           "next_up": None, "next_down": None}
    p = valid_price(price)
    lv = [r for r in ladder if r["value"] is not None]
    if p is None or not lv:
        return out
    lv.sort(key=lambda r: r["value"])
    out["next_up"] = next((r for r in lv if r["value"] > p), None)
    below = [r for r in lv if r["value"] < p]
    out["next_down"] = below[-1] if below else None
    pt = fmt_price(p, currency)
    if p > lv[-1]["value"]:
        out["text"] = f"Current price {pt} is above {lv[-1]['label']} {fmt_price(lv[-1]['value'], currency)}"
        out["lower"] = lv[-1]
        return out
    if p < lv[0]["value"]:
        out["text"] = f"Current price {pt} is below {lv[0]['label']} {fmt_price(lv[0]['value'], currency)}"
        out["upper"] = lv[0]
        return out
    lower = [r for r in lv if r["value"] <= p][-1]
    upper = out["next_up"]
    out["lower"], out["upper"] = lower, upper
    if upper is None:
        out["text"] = f"Current price {pt} is at {lower['label']} {fmt_price(lower['value'], currency)}"
    else:
        out["text"] = (f"Current price {pt} is between {lower['label']} {fmt_price(lower['value'], currency)} "
                       f"and {upper['label']} {fmt_price(upper['value'], currency)}")
    return out


def classic_pivots(high: Any, low: Any, close: Any) -> Dict[str, float]:
    h, l, c = valid_price(high), valid_price(low), valid_price(close)
    if not (h and l and c):
        return {}
    p = (h + l + c) / 3.0
    span = h - l
    return {"Pivot": p, "R1": 2 * p - l, "R2": p + span, "R3": h + 2 * (p - l),
            "S1": 2 * p - h, "S2": p - span, "S3": l - 2 * (h - p)}


def generate_suggestions(price: Optional[float], ladder: List[Dict[str, Any]],
                         atr: Optional[float] = None) -> List[Dict[str, Any]]:
    """Both directions, built only from the real Gann ladder (no invented signals)."""
    p = valid_price(price)
    lv = sorted([r for r in ladder if r["value"] is not None and r["side"] != "0"], key=lambda r: r["value"])
    if p is None or not lv:
        return []
    sup = [r for r in lv if r["value"] < p][::-1]
    res = [r for r in lv if r["value"] > p]
    plans: List[Dict[str, Any]] = []
    if len(sup) >= 2 and res:
        risk = abs(sup[0]["value"] - sup[1]["value"]) or (atr or 0)
        if risk > 0:
            t2 = res[1]["value"] if len(res) > 1 else res[0]["value"]
            plans.append({"direction": "bullish", "entry": sup[0]["value"], "stop": sup[1]["value"],
                          "t1": res[0]["value"], "t2": t2,
                          "rr1": round(abs(res[0]["value"] - sup[0]["value"]) / risk, 2),
                          "rr2": round(abs(t2 - sup[0]["value"]) / risk, 2),
                          "basis": f"{sup[0]['label']} entry / {sup[1]['label']} stop / {res[0]['label']} target"})
    if len(res) >= 2 and sup:
        risk = abs(res[1]["value"] - res[0]["value"]) or (atr or 0)
        if risk > 0:
            t2 = sup[1]["value"] if len(sup) > 1 else sup[0]["value"]
            plans.append({"direction": "bearish", "entry": res[0]["value"], "stop": res[1]["value"],
                          "t1": sup[0]["value"], "t2": t2,
                          "rr1": round(abs(res[0]["value"] - sup[0]["value"]) / risk, 2),
                          "rr2": round(abs(res[0]["value"] - t2) / risk, 2),
                          "basis": f"{res[0]['label']} entry / {res[1]['label']} stop / {sup[0]['label']} target"})
    return plans


# ===========================================================================
# SECTION 8 - GOLDEN POCKET (confirmed pivot swings, Pine-style symmetric pivots)
# ===========================================================================
NO_GP_MESSAGE = ("No confirmed Golden Pocket yet. Wait for a valid pivot low → pivot high or "
                 "pivot high → pivot low swing.")


def find_pivots(highs: np.ndarray, lows: np.ndarray, length: int) -> Tuple[List[int], List[int]]:
    """Confirmed symmetric pivots. Index i is a pivot high when High[i] is the maximum of
    High[i-length .. i+length] (leftmost bar wins a tie). Only i <= n-1-length is examined, so a
    pivot is always confirmed by `length` later candles - unconfirmed pivots are never returned."""
    n = len(highs)
    ph: List[int] = []
    pl: List[int] = []
    for i in range(length, n - length):
        wh = highs[i - length:i + length + 1]
        if highs[i] == wh.max() and int(np.argmax(wh)) == length:
            ph.append(i)
        wl = lows[i - length:i + length + 1]
        if lows[i] == wl.min() and int(np.argmin(wl)) == length:
            pl.append(i)
    return ph, pl


def calculate_golden_pocket(df: pd.DataFrame, pivot_len: int = 7,
                            current_price: Optional[float] = None) -> Dict[str, Any]:
    """Latest valid Golden Pocket (bullish or bearish, whichever swing completed most recently).

    Bullish  : A = confirmed pivot low, B = later confirmed pivot high (B > A)
               fib_500 = B - 0.500*(B-A), fib_618 = B - 0.618*(B-A)
    Bearish  : A = confirmed pivot high, B = later confirmed pivot low (A > B)
               fib_500 = B + 0.500*(A-B), fib_618 = B + 0.618*(A-B)
    """
    result: Dict[str, Any] = {"active": False, "direction": None, "bias": None, "swing_a": None,
                              "swing_b": None, "bar_a": None, "bar_b": None, "time_a": None, "time_b": None,
                              "fib_500": None, "fib_618": None, "zone_top": None, "zone_bottom": None,
                              "current_price": None, "price_status": None, "distance_to_50": None,
                              "distance_to_618": None, "next_level": None, "next_level_label": None,
                              "pivot_len": None, "reason": NO_GP_MESSAGE}
    try:
        length = max(2, int(pivot_len))
        result["pivot_len"] = length
        if df is None or df.empty or not all(c in df.columns for c in ("High", "Low", "Close")):
            result["reason"] = "No OHLC data available. " + NO_GP_MESSAGE
            return result
        work = df[["High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 2 * length + 3:
            result["reason"] = (f"Only {len(work)} bars available; at least {2 * length + 3} are needed for a "
                                f"pivot length of {length}. " + NO_GP_MESSAGE)
            return result
        highs, lows = work["High"].to_numpy(float), work["Low"].to_numpy(float)
        ph, pl = find_pivots(highs, lows, length)
        bull = bear = None
        for b in reversed(ph):
            cand = [a for a in pl if a < b and lows[a] < highs[b]]
            if cand:
                bull = (cand[-1], b)
                break
        for b in reversed(pl):
            cand = [a for a in ph if a < b and highs[a] > lows[b]]
            if cand:
                bear = (cand[-1], b)
                break
        if bull is None and bear is None:
            return result
        if bear is None or (bull is not None and bull[1] >= bear[1]):
            a_i, b_i = bull  # type: ignore[misc]
            a_p, b_p = float(lows[a_i]), float(highs[b_i])
            rng = b_p - a_p
            f50, f618 = b_p - 0.5 * rng, b_p - 0.618 * rng
            direction, bias = "Bullish Retracement", "Bullish"
        else:
            a_i, b_i = bear
            a_p, b_p = float(highs[a_i]), float(lows[b_i])
            rng = a_p - b_p
            f50, f618 = b_p + 0.5 * rng, b_p + 0.618 * rng
            direction, bias = "Bearish Retracement", "Bearish"
        top, bottom = max(f50, f618), min(f50, f618)
        price = valid_price(current_price) or float(work["Close"].iloc[-1])   # live quote when supplied
        if price > top:
            status, nxt, label = "Above Golden Pocket", top, ("50.0%" if top == f50 else "61.8%")
        elif price < bottom:
            status, nxt, label = "Below Golden Pocket", bottom, ("50.0%" if bottom == f50 else "61.8%")
        else:
            status = "Inside Golden Pocket"
            nxt = f50 if abs(price - f50) <= abs(price - f618) else f618
            label = "50.0%" if nxt == f50 else "61.8%"
        result.update({"active": True, "direction": direction, "bias": bias, "swing_a": a_p, "swing_b": b_p,
                       "bar_a": int(a_i), "bar_b": int(b_i), "time_a": work.index[a_i], "time_b": work.index[b_i],
                       "fib_500": f50, "fib_618": f618, "zone_top": top, "zone_bottom": bottom,
                       "current_price": price, "price_status": status, "distance_to_50": price - f50,
                       "distance_to_618": price - f618, "next_level": nxt, "next_level_label": label,
                       "reason": ""})
        return result
    except Exception as exc:
        log_exception("golden pocket", exc)
        result["reason"] = f"Golden Pocket could not be calculated: {safe_error(exc)}"
        return result


# ===========================================================================
# SECTION 9 - INDICATORS
# ===========================================================================
def technical_indicators(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "atr": None, "rsi": None, "ema9": None, "ema50": None,
                           "trend": "", "swing_high": None, "swing_low": None, "note": ""}
    daily, err = fetch_history(symbol, "6mo", "1d")
    if daily.empty or len(daily) < 15:
        out["note"] = err or "Insufficient data (fewer than 15 daily bars)."
        return out
    try:
        close = daily["Close"]
        prev = close.shift(1)
        tr = pd.concat([(daily["High"] - daily["Low"]).abs(), (daily["High"] - prev).abs(),
                        (daily["Low"] - prev).abs()], axis=1).max(axis=1)
        out["atr"] = _num(tr.rolling(14).mean().iloc[-1])
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        out["rsi"] = 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))
        out["ema9"] = _num(close.ewm(span=9, adjust=False).mean().iloc[-1])
        out["ema50"] = _num(close.ewm(span=50, adjust=False).mean().iloc[-1])
        last = float(close.iloc[-1])
        if out["ema9"] and out["ema50"]:
            out["trend"] = ("Uptrend" if last > out["ema9"] > out["ema50"] else
                            "Downtrend" if last < out["ema9"] < out["ema50"] else "Sideways / mixed")
        out["swing_high"] = _num(daily["High"].tail(60).max())
        out["swing_low"] = _num(daily["Low"].tail(60).min())
        out["ok"] = True
    except Exception as exc:
        log_exception("indicators", exc)
        out["note"] = safe_error(exc)
    return out


# ===========================================================================
# SECTION 10 - PERSISTENCE, GLOBAL ACTIVE-SYMBOL STATE
# ===========================================================================
def load_state_file() -> Dict[str, Any]:
    try:
        if STATE_FILE.exists():
            with STATE_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        log_exception("state read", exc)
    return {}


def persist_state() -> None:
    """Best-effort save of the watchlist and alerts (Streamlit Cloud storage is ephemeral)."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        s = st.session_state
        with STATE_FILE.open("w", encoding="utf-8") as handle:
            json.dump({"watchlist": list(s.get("watchlist", [])), "alerts": list(s.get("alerts", []))},
                      handle, default=str)
    except Exception as exc:
        log_exception("state write", exc)


def init_state() -> None:
    s = st.session_state
    saved = load_state_file() if "watchlist" not in s else {}
    defaults: Dict[str, Any] = {
        "active_market": M_IN, "active_exchange": "NSE", "active_universe": "NIFTY 50",
        "active_symbol": "", "active_display_name": "", "active_price": None, "active_change_pct": None,
        "active_currency": "USD", "chat_messages": [], "ai_suggestion": None, "_alert_prefill": True,
        "watchlist": saved.get("watchlist") or ["RELIANCE.NS", "AAPL", "BTC-USD"],
        "alerts": saved.get("alerts") or [], "page": "Live Analysis",
    }
    for key, value in defaults.items():
        if key not in s:
            s[key] = value


def set_active_symbol(market: str, exchange: str, universe: str, symbol: str, display: str,
                      price: Optional[float], change_pct: Optional[float], currency: str = "USD") -> None:
    """Single source of truth. Every page, the AI context and the alert form read these values."""
    s = st.session_state
    changed = s.get("active_symbol") != symbol
    s.active_market, s.active_exchange, s.active_universe = market, exchange, universe
    s.active_symbol, s.active_display_name = symbol, display
    s.active_price, s.active_change_pct, s.active_currency = price, change_pct, currency
    if changed:
        s.chat_messages = []          # chat is per-symbol: never show stale answers about another symbol
        s.ai_suggestion = None
        s._alert_prefill = True       # new-alert form re-fills on its next render


def open_symbol_callback(symbol: str) -> None:
    """on_click helper (runs before the next rerun): jump the sidebar to a symbol's own market."""
    market, exchange, universe = infer_market_context(symbol)
    s = st.session_state
    s["sel_market"] = market
    s[f"sel_exchange_{market}"] = exchange
    s[f"sel_universe_{market}_{exchange}"] = universe
    s[f"sel_custom_{market}"] = symbol


def no_data_message(market: str, symbol: str, error: str = "") -> str:
    if market == M_OTC:
        return ("No Yahoo Finance data is available for this OTC/CFD/synthetic symbol. Enter a valid "
                "Yahoo-supported symbol or connect a dedicated broker/data API.")
    return (f"No Yahoo Finance data is available for {symbol}. Check the spelling / suffix"
            + (f" ({error})" if error else "") + ".")


# ===========================================================================
# SECTION 11 - ANALYSIS BUNDLE (Gann + Golden Pocket for the ACTIVE symbol)
# ===========================================================================
def analysis_bundle() -> Dict[str, Any]:
    s = st.session_state
    symbol = s.get("active_symbol") or ""
    bundle: Dict[str, Any] = {"symbol": symbol, "ok": False}
    if not symbol:
        return bundle
    q = get_quote(symbol)
    currency = q.get("currency") or "USD"
    price = q.get("price")
    utc = s.get("active_market") == M_OTC and s.get("otc_marker_mode") == OTC_MODES[2]
    anchor = resolve_anchor(symbol, s.get("anchor_mode", ANCHOR_MODES[0]), s.get("custom_anchor"), utc)
    levels = gann_levels(anchor["price"]) if anchor["ok"] else {}
    ladder = gann_ladder(levels)
    gp: Optional[Dict[str, Any]] = None
    if s.get("gp_enabled", True):
        period, interval = GP_TIMEFRAMES.get(s.get("gp_tf", "1 hour · 3 months"), ("3mo", "1h"))
        gdf, gerr = fetch_history(symbol, period, interval)
        gp = calculate_golden_pocket(gdf, int(s.get("gp_pivot", 7) or 7), price)
        gp["fetch_error"] = gerr
    bundle.update({"ok": bool(q.get("ok")), "quote": q, "currency": currency, "price": price, "anchor": anchor,
                   "levels": levels, "ladder": ladder, "position": gann_position(price, ladder, currency),
                   "gp": gp})
    return bundle


def build_ai_context() -> str:
    """Everything the AI is allowed to see: selected market/symbol, price, anchor, Gann, Golden Pocket, news."""
    s = st.session_state
    b = analysis_bundle()
    lines = ["CURRENT TERMINAL CONTEXT (educational analysis only)",
             f"- Selected market: {s.get('active_market')} | exchange: {s.get('active_exchange')} | "
             f"universe: {s.get('active_universe')}",
             f"- Full symbol: {s.get('active_symbol') or 'none'} ({s.get('active_display_name') or ''})"]
    if b.get("symbol"):
        cur = b.get("currency", "USD")
        q = b.get("quote", {})
        lines.append(f"- Current price: {fmt_price(b.get('price'), cur)} ({fmt_pct(q.get('change_pct'))}) "
                     f"in {cur}" if b.get("price") else "- Current price: unavailable (no data returned)")
        anc = b.get("anchor", {})
        if anc.get("ok"):
            lines.append(f"- Anchor type: {anc['type']} | 0° Gann anchor: {fmt_price(anc['price'], cur)} "
                         f"({anc.get('time')})")
            lines.append(f"- Price position: {b['position']['text']}")
            nu, nd = b["position"].get("next_up"), b["position"].get("next_down")
            if nu:
                lines.append(f"- Next upside Gann level: {nu['label']} {fmt_price(nu['value'], cur)}")
            if nd:
                lines.append(f"- Next downside Gann level: {nd['label']} {fmt_price(nd['value'], cur)}")
            lines.append("- Gann levels: " + " | ".join(f"{r['label']}={fmt_price(r['value'], cur)}"
                                                        for r in b["ladder"]))
        else:
            lines.append(f"- Gann anchor unavailable: {anc.get('note', '')}")
        gp = b.get("gp")
        if gp and gp.get("active"):
            lines.append(f"- Golden Pocket ({gp['direction']}): swing A {fmt_price(gp['swing_a'], cur)} → "
                         f"swing B {fmt_price(gp['swing_b'], cur)}; 50.0% {fmt_price(gp['fib_500'], cur)}, "
                         f"61.8% {fmt_price(gp['fib_618'], cur)}; zone {fmt_price(gp['zone_bottom'], cur)} – "
                         f"{fmt_price(gp['zone_top'], cur)}; price is: {gp['price_status']}")
        elif gp is not None:
            lines.append("- Golden Pocket: no confirmed swing yet.")
    heads = headline_lines(8)
    lines.append("- Global market headlines (NOT stock-specific): " + (" || ".join(heads) if heads else "none available"))
    if s.get("active_market") == M_OTC:
        lines.append(f"- Data caveat: OTC/CFD/synthetic symbols use Yahoo-supported proxy data "
                     f"(mode: {s.get('otc_marker_mode', OTC_MODES[0])}); this is not a broker feed.")
    return "\n".join(lines)


# ===========================================================================
# SECTION 12 - AI (Groq only, verified model)
# ===========================================================================
class AIUnavailable(RuntimeError):
    pass


AI_SYSTEM = ("You are an educational market-analysis assistant inside a trading terminal. Discuss the "
             "instrument and market context supplied below; you may also answer general questions. Use only "
             "the supplied numbers for levels, never invent prices or news, quote the currency, and say when "
             "data is missing. Present scenarios and invalidation levels, never certainty or guaranteed "
             "returns. This is education, not financial advice.")


def friendly_ai_error(exc: BaseException) -> str:
    low = str(exc).lower()
    if any(k in low for k in ("401", "invalid api key", "invalid_api_key", "unauthorized")):
        return "the Groq key was rejected (401). Re-check GROQ_KEY in Streamlit Secrets."
    if any(k in low for k in ("403", "forbidden", "permission")):
        return "access was refused (403); the key may lack access to this model."
    if any(k in low for k in ("429", "rate limit", "quota", "too many requests")):
        return "rate limit or quota reached. Wait a moment and retry."
    if any(k in low for k in ("404", "not found", "decommission", "does not exist")):
        return f"model {AI_MODEL_MAP[DEFAULT_AI_LABEL]['model']} is not available for this key."
    if any(k in low for k in ("timeout", "timed out", "connection", "network")):
        return "network timeout while contacting Groq. Try again."
    return _redact(str(exc))[:240] or type(exc).__name__


def call_groq(system: str, messages: List[Dict[str, str]], max_tokens: int = 1500,
              temperature: float = 0.4) -> str:
    key = groq_key()
    if not key:
        raise AIUnavailable("Groq API key is not configured. Add GROQ_KEY to Streamlit Secrets.")
    if Groq is None:
        raise AIUnavailable("the `groq` package is not installed (add it to requirements.txt).")
    model = AI_MODEL_MAP[DEFAULT_AI_LABEL]["model"]
    payload = [{"role": "system", "content": system}]
    payload += [{"role": m["role"], "content": m["content"]} for m in messages
                if m.get("role") in ("user", "assistant") and m.get("content")]
    kwargs: Dict[str, Any] = dict(model=model, messages=payload, temperature=temperature, max_tokens=max_tokens)
    try:
        client = Groq(api_key=key)
        try:
            resp = client.chat.completions.create(reasoning_effort="low", **kwargs)
        except Exception as first:
            if isinstance(first, TypeError) or "reasoning" in str(first).lower():
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
    except AIUnavailable:
        raise
    except Exception as exc:
        log_exception("groq call", exc)
        raise AIUnavailable(friendly_ai_error(exc)) from exc
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        text = ""
    if not text:
        raise AIUnavailable("the model returned an empty response.")
    return text


def ai_trading_suggestion() -> str:
    context = build_ai_context()
    prompt = ("Using ONLY the context above, write an educational trading-scenario note for the active symbol:\n"
              "1) Bias and where price sits versus the Gann 0° anchor and nearest levels\n"
              "2) Golden Pocket read (direction, zone, whether price is inside/above/below) if active\n"
              "3) A bullish scenario and a bearish scenario, each with trigger, invalidation/stop idea and "
              "targets taken from the listed levels\n"
              "4) How the global headlines could matter (they are market-wide, not stock-specific)\n"
              "5) Risks and what the data does NOT establish. Keep it under 300 words. No guarantees.")
    return call_groq(AI_SYSTEM + "\n\n" + context, [{"role": "user", "content": prompt}], max_tokens=1800)


# ===========================================================================
# SECTION 13 - ALERTS (optional Telegram, no duplicate sends)
# ===========================================================================
def send_telegram(text: str) -> Tuple[bool, str]:
    token, chat_id = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return False, "Telegram is not configured."
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json={"chat_id": chat_id, "text": text}, timeout=8)
        if resp.status_code != 200:
            return False, f"Telegram returned HTTP {resp.status_code}."
        return True, ""
    except Exception as exc:
        return False, _redact(safe_error(exc))


def add_alert(symbol: str, target: float, condition: str) -> Tuple[bool, str]:
    s = st.session_state
    if not symbol or target <= 0:
        return False, "Enter a symbol and a target price above zero."
    for a in s.alerts:
        if (a["symbol"] == symbol and abs(a["target"] - target) < 1e-9 and a["condition"] == condition
                and a["status"] == "Active"):
            return False, "An identical active alert already exists."
    name = s.active_display_name if symbol == s.get("active_symbol") else display_name(symbol)
    s.alerts.append({"id": f"{int(time.time() * 1000)}", "symbol": symbol, "name": name, "target": float(target),
                     "condition": condition, "status": "Active", "created": datetime.now().strftime("%d %b %H:%M"),
                     "triggered_at": "", "telegram_sent": False, "telegram_attempted": False})
    persist_state()
    return True, ""


def evaluate_alerts() -> List[str]:
    """Runs on every rerun using ONE cached batch quote call. Each alert notifies at most once."""
    s = st.session_state
    active = [a for a in s.get("alerts", []) if a.get("status") == "Active"]
    if not active:
        return []
    quotes = fetch_quotes_batch(tuple(sorted({a["symbol"] for a in active})))
    fired: List[str] = []
    changed = False
    for a in active:
        price = (quotes.get(a["symbol"]) or {}).get("price")
        if price is None:
            continue
        hit = price >= a["target"] if a["condition"] == "Above" else price <= a["target"]
        if not hit:
            continue
        a["status"], a["triggered_at"] = "Triggered", datetime.now().strftime("%d %b %H:%M")
        changed = True
        cur = guess_currency(a["symbol"])
        fired.append(f"{a['symbol']} {a['condition']} {fmt_price(a['target'], cur)} (now {fmt_price(price, cur)})")
        if telegram_ready() and not a.get("telegram_attempted"):
            a["telegram_attempted"] = True
            ok, _err = send_telegram(f"ALERT: {a['symbol']} is {a['condition'].lower()} "
                                     f"{fmt_price(a['target'], cur)} - now {fmt_price(price, cur)}")
            a["telegram_sent"] = ok
    if changed:
        persist_state()
    return fired


# ===========================================================================
# SECTION 14 - CHARTS
# ===========================================================================
_LAYOUT = dict(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", font=dict(color="#94a3b8", size=11),
               margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, showlegend=False)


def _candles(df: pd.DataFrame) -> Any:
    return go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                          increasing_line_color="#34d399", decreasing_line_color="#f87171", name="Price")


def analysis_chart(symbol: str, ladder: List[Dict[str, Any]], price: Optional[float], currency: str):
    df, err = fetch_history(symbol, "5d", "15m")
    label = "15-minute bars · last 5 sessions"
    if df.empty:
        df, err = fetch_history(symbol, "1mo", "1d")
        label = "Daily bars · last month (intraday unavailable)"
    if df.empty:
        return None, err or "No candle data available."
    fig = go.Figure(_candles(df))
    for r in ladder:
        colour = "#fbbf24" if r["side"] == "0" else ("#f87171" if r["side"] == "R" else "#34d399")
        fig.add_hline(y=r["value"], line_dash="dot", line_color=colour, line_width=1,
                      annotation_text=r["label"], annotation_position="right",
                      annotation_font_color=colour, annotation_font_size=9)
    if valid_price(price):
        fig.add_hline(y=float(price), line_color="#60a5fa", line_width=1.5,
                      annotation_text=f"Price {fmt_price(price, currency)}", annotation_position="left",
                      annotation_font_color="#60a5fa", annotation_font_size=10)
    fig.update_layout(height=420, yaxis=dict(side="right", gridcolor="rgba(51,65,85,.3)"),
                      xaxis=dict(gridcolor="rgba(51,65,85,.3)"), **_LAYOUT)
    return fig, label


def golden_pocket_figure(df: pd.DataFrame, gp: Dict[str, Any], ext_bars: int, price: Optional[float],
                         currency: str):
    fig = go.Figure(_candles(df))
    last_ts = df.index[-1]
    delta = df.index.to_series().diff().median() if len(df) > 2 else pd.Timedelta(days=1)
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        delta = pd.Timedelta(days=1)
    x1 = last_ts + delta * max(0, int(ext_bars))
    fig.add_shape(type="rect", x0=gp["time_b"], x1=x1, y0=gp["zone_bottom"], y1=gp["zone_top"],
                  fillcolor="rgba(255,165,0,0.25)", line=dict(color="rgba(255,165,0,0.8)", width=1), layer="below")
    fig.add_hline(y=gp["fib_500"], line_dash="dash", line_color="#fb923c", line_width=1.2,
                  annotation_text=f"50.0%  {fmt_price(gp['fib_500'], currency)}", annotation_position="top left",
                  annotation_font_color="#fb923c", annotation_font_size=10)
    fig.add_hline(y=gp["fib_618"], line_dash="dash", line_color="#f59e0b", line_width=1.2,
                  annotation_text=f"61.8%  {fmt_price(gp['fib_618'], currency)}", annotation_position="bottom left",
                  annotation_font_color="#f59e0b", annotation_font_size=10)
    if valid_price(price):
        fig.add_hline(y=float(price), line_color="#60a5fa", line_width=1.5,
                      annotation_text=f"Current {fmt_price(price, currency)}", annotation_position="top right",
                      annotation_font_color="#60a5fa", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=[gp["time_a"], gp["time_b"]], y=[gp["swing_a"], gp["swing_b"]], mode="markers+text",
                             text=["A", "B"], textposition="top center", marker=dict(size=8, color="#fbbf24"),
                             hoverinfo="skip"))
    fig.update_layout(height=430, yaxis=dict(side="right", gridcolor="rgba(51,65,85,.3)"),
                      xaxis=dict(gridcolor="rgba(51,65,85,.3)", range=[df.index[0], x1]), **_LAYOUT)
    return fig


def heatmap_figure(groups: Dict[str, List[Dict[str, Any]]]):
    """Treemap: colour = signed intensity (0-10, red = down, green = up); tiles are equal weight."""
    ids: List[str] = []
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    colours: List[float] = []
    texts: List[str] = []
    hovers: List[str] = []
    for group, rows in groups.items():
        if not rows:
            continue
        ids.append(f"g:{group}")
        labels.append(group)
        parents.append("")
        values.append(0)
        avg = float(np.mean([r["signed_intensity"] for r in rows]))
        colours.append(avg)
        texts.append(f"avg intensity {avg:+.1f}/10")
        hovers.append(f"{group}<br>{len(rows)} instruments")
        for r in rows:
            ids.append(f"s:{group}:{r['symbol']}")
            labels.append(r["short"])
            parents.append(f"g:{group}")
            values.append(1)
            colours.append(r["signed_intensity"])
            texts.append(f"{fmt_pct(r['change_pct'])} · {r['intensity']:.1f}/10")
            hovers.append(f"{r['name']} ({r['symbol']})<br>{fmt_price(r['price'], r['currency'])}<br>"
                          f"{fmt_pct(r['change_pct'])}<br>intensity {r['intensity']:.1f}/10")
    if not ids:
        return None
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, text=texts, textinfo="label+text",
        hovertext=hovers, hoverinfo="text",
        marker=dict(colors=colours, colorscale=[[0, "#b91c1c"], [0.5, "#334155"], [1, "#15803d"]],
                    cmin=-10, cmax=10, line=dict(width=1, color="#0F172A"),
                    colorbar=dict(title="Intensity", tickvals=[-10, -5, 0, 5, 10])),
        branchvalues="remainder", tiling=dict(pad=2)))
    fig.update_layout(height=560, paper_bgcolor="#0F172A", font=dict(color="#f8fafc"),
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


def heatmap_rows(symbols: List[Tuple[str, str]], scale: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Rows only for symbols Yahoo actually returned; skipped symbols are reported, never faked."""
    quotes = fetch_quotes_batch(tuple(s for s, _ in symbols))
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for sym, name in symbols:
        q = quotes.get(sym) or {}
        if not q.get("ok") or q.get("change_pct") is None:
            skipped.append(sym)
            continue
        pct = float(q["change_pct"])
        inten = round(min(10.0, abs(pct) / scale * 10.0), 1)
        rows.append({"symbol": sym, "short": sym.replace(".NS", "").replace(".BO", "").replace("=F", "")
                     .replace("=X", "").replace("-USD", ""), "name": name, "price": q["price"],
                     "currency": q["currency"], "change_pct": pct, "intensity": inten,
                     "signed_intensity": inten if pct >= 0 else -inten})
    return rows, skipped


# ===========================================================================
# SECTION 15 - SIDEBAR  (Market -> Exchange -> Universe -> Find Symbol, anchor + Golden Pocket controls)
# ===========================================================================
def render_sidebar() -> None:
    s = st.session_state
    with st.sidebar:
        st.markdown("## ⚡ AI Trade Terminal")
        market = st.selectbox("🌎 Market", list(MARKET_TREE.keys()), key="sel_market")
        exchange = st.selectbox("Exchange", list(MARKET_TREE[market].keys()), key=f"sel_exchange_{market}")
        universe = st.selectbox("Universe", list(MARKET_TREE[market][exchange].keys()),
                                key=f"sel_universe_{market}_{exchange}")
        pairs = universe_symbols(market, exchange, universe)
        options = [f"{sym} — {name}" for sym, name in pairs]
        pick_label = {M_FU: "Futures Contract", M_OTC: "Example Yahoo symbols"}.get(market, "Find Symbol / Contract")
        picked = ""
        if options:
            choice = st.selectbox(pick_label, options, key=f"sel_symbol_{market}_{exchange}_{universe}")
            picked = pairs[options.index(choice)][0]
        custom_label = ("OTC / CFD / Synthetic Yahoo Symbol" if market == M_OTC else "Or type a custom symbol")
        custom = st.text_input(custom_label, key=f"sel_custom_{market}",
                               placeholder="e.g. EURUSD=X, GC=F, BTC-USD" if market == M_OTC else
                               "Type a symbol — overrides the list until cleared")
        if market == M_OTC:
            st.info("Yahoo Finance / yfinance does not provide a universal OTC / CFD / synthetic symbol feed. "
                    "Enter a valid Yahoo-supported symbol (the examples above are commonly available), or "
                    "connect a dedicated broker/data API. No prices are ever invented.")
            st.selectbox("OTC / Synthetic marker mode", OTC_MODES, key="otc_marker_mode",
                         help="Changes labels, session logic and warnings only - never the market data.")
        if is_custom_universe(market, exchange, universe) and not custom:
            st.caption("Custom universe: type a symbol above, or use the pick-list.")

        resolved = normalize_symbol((custom or "").strip() or picked, market, exchange)
        quote = get_quote(resolved) if resolved else {"ok": False, "price": None, "change_pct": None,
                                                      "currency": "USD", "error": "No symbol entered."}
        set_active_symbol(market, exchange, universe, resolved, display_name(resolved),
                          quote.get("price") if quote.get("ok") else None,
                          quote.get("change_pct") if quote.get("ok") else None,
                          quote.get("currency") or "USD")
        if quote.get("ok"):
            st.success(f"Active: **{resolved}** — {display_name(resolved)}  \n"
                       f"{fmt_price(quote['price'], quote['currency'])} ({fmt_pct(quote.get('change_pct'))})")
        elif resolved:
            st.error(no_data_message(market, resolved, quote.get("error", "")))
        st.markdown("---")

        st.markdown("**Gann anchor / session**")
        mode = st.selectbox("Anchor / Session Mode", ANCHOR_MODES, index=0, key="anchor_mode")
        if mode == ANCHOR_MODES[3]:
            st.number_input("Custom 0° anchor price", min_value=0.0, value=float(quote.get("price") or 1.0),
                            format="%.4f", key="custom_anchor")
        st.markdown("**Golden Pocket**")
        st.checkbox("Enable Golden Pocket", value=True, key="gp_enabled")
        st.number_input("Pivot Lookback Length", min_value=2, max_value=50, value=7, step=1, key="gp_pivot")
        st.selectbox("Golden Pocket timeframe", list(GP_TIMEFRAMES.keys()), index=1, key="gp_tf")
        st.number_input("Golden Pocket extension bars", min_value=0, max_value=500, value=20, step=5,
                        key="gp_ext", help="How many bars the orange band extends beyond the latest bar.")
        st.markdown("---")
        if wbutton("⟳ Refresh market data", key="refresh_btn"):
            try:
                st.cache_data.clear()
            except Exception as exc:
                log_exception("cache clear", exc)
            st.rerun()
        st.caption("Data: Yahoo Finance via yfinance — may be delayed. Educational use only.")


# ===========================================================================
# SECTION 16 - PAGE: LIVE ANALYSIS
# ===========================================================================
def _level_row(css: str, colour: str, label: str, value: str, pct: Any) -> str:
    return (f"<div class='level-row {css}'><span><b style='color:{colour}'>{label}</b> &nbsp; {value}</span>"
            f"<span class='{change_class(pct)}'>{fmt_pct(pct)}</span></div>")


def page_live() -> None:
    s = st.session_state
    symbol = s.active_symbol
    if not symbol:
        st.info("Choose a market and symbol in the sidebar.")
        return
    b = analysis_bundle()
    q, cur, price = b["quote"], b["currency"], b["price"]
    market = s.active_market
    status, stamp = data_status(symbol, market)
    sess = market_session(market)

    with st.container(border=True):
        st.markdown(f"<div class='card-header'>{html.escape(s.active_display_name or symbol)} "
                    f"<span style='color:#64748b;font-size:.85rem'>{html.escape(symbol)}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:6px'>"
                    f"{s.active_market} · {s.active_exchange} · {html.escape(str(s.active_universe))} "
                    f"{status_pill(sess['status'])} {status_pill(status)} "
                    f"<span style='color:#64748b'>as of {stamp or '—'} · {sess['local_time']} · {cur}</span></div>",
                    unsafe_allow_html=True)
        if market == M_OTC:
            mode = s.get("otc_marker_mode", OTC_MODES[0])
            if mode == OTC_MODES[1]:
                st.warning("OTC / CFD (user symbol) mode: these are Yahoo Finance prices for the symbol you "
                           "entered — not a broker or CFD feed. Spreads and broker prices can differ.")
            elif mode == OTC_MODES[2]:
                st.warning("Synthetic session mode: session boundaries use UTC days. Data are still real "
                           "Yahoo Finance bars; no synthetic prices are generated.")
        if not q.get("ok"):
            st.error(no_data_message(market, symbol, q.get("error", "")))
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_html("Price", fmt_price(price, cur), change_class(q.get("change_pct")),
                                f"{fmt_pct(q.get('change_pct'))} · {cur}"), unsafe_allow_html=True)
        c2.markdown(metric_html("Day range", f"{fmt_price(q.get('low'), cur)} – {fmt_price(q.get('high'), cur)}",
                                sub=f"open {fmt_price(q.get('open'), cur)}"), unsafe_allow_html=True)
        c3.markdown(metric_html("Previous close", fmt_price(q.get("prev_close"), cur),
                                sub=f"change {fmt_delta(q.get('change'), cur)}"), unsafe_allow_html=True)
        c4.markdown(metric_html("Volume", fmt_volume(q.get("volume")), sub=f"last bar {q.get('last_bar') or '—'}"),
                    unsafe_allow_html=True)

    fig, note = analysis_chart(symbol, b["ladder"], price, cur)
    with st.container(border=True):
        st.markdown(f"<div class='card-header'>Price action · Gann levels</div>"
                    f"<div class='card-sub'>{note}</div>", unsafe_allow_html=True)
        if fig is not None:
            show_fig(fig)
        else:
            st.info(note)

    # ---------------- Gann anchor + degree levels ----------------
    anchor, ladder, pos = b["anchor"], b["ladder"], b["position"]
    with st.container(border=True):
        st.markdown("<div class='card-header'>Square-of-9 Gann anchor &amp; degree levels</div>"
                    "<div class='card-sub'>Resistance = (√anchor + degree/180)² · Support = (√anchor − degree/180)² "
                    "· every level uses the same 0° anchor</div>", unsafe_allow_html=True)
        if not anchor.get("ok"):
            st.warning("Gann anchor unavailable: " + (anchor.get("note") or "no data") +
                       (" Enter a custom anchor in the sidebar." if s.get("anchor_mode") != ANCHOR_MODES[3] else ""))
        else:
            a1, a2, a3 = st.columns(3)
            a1.markdown(metric_html("0° Anchor Price", fmt_price(anchor["price"], cur), "amber",
                                    anchor.get("time", "")), unsafe_allow_html=True)
            a2.markdown(metric_html("Anchor Type", html.escape(anchor["type"])), unsafe_allow_html=True)
            a3.markdown(metric_html("Current price", fmt_price(price, cur), change_class(q.get("change_pct"))),
                        unsafe_allow_html=True)
            if anchor.get("note"):
                st.caption(anchor["note"])
            st.markdown(f"<div class='level-row level-current'><b>📍 {html.escape(pos['text'])}</b></div>",
                        unsafe_allow_html=True)
            nu, nd = pos.get("next_up"), pos.get("next_down")
            n1, n2 = st.columns(2)
            n1.markdown(f"<div class='level-row level-resistance'><span><b>Next upside</b> "
                        f"{html.escape(nu['label']) if nu else '—'} {fmt_price(nu['value'], cur) if nu else ''}</span>"
                        f"<span class='up'>{fmt_pct((nu['value'] - price) / price * 100) if nu else '—'}</span></div>",
                        unsafe_allow_html=True)
            n2.markdown(f"<div class='level-row level-support'><span><b>Next downside</b> "
                        f"{html.escape(nd['label']) if nd else '—'} {fmt_price(nd['value'], cur) if nd else ''}</span>"
                        f"<span class='down'>{fmt_pct((nd['value'] - price) / price * 100) if nd else '—'}</span></div>",
                        unsafe_allow_html=True)
            placed = False
            for r in sorted(ladder, key=lambda x: x["value"], reverse=True):
                if not placed and price >= r["value"]:
                    st.markdown(_level_row("level-current", "#60a5fa", "▶ Current price", fmt_price(price, cur), 0.0),
                                unsafe_allow_html=True)
                    placed = True
                pct = (r["value"] - price) / price * 100
                if r["side"] == "0":
                    st.markdown(_level_row("level-anchor", "#fbbf24", r["label"], fmt_price(r["value"], cur), pct),
                                unsafe_allow_html=True)
                elif r["side"] == "R":
                    st.markdown(_level_row("level-resistance", "#f87171", r["label"], fmt_price(r["value"], cur), pct),
                                unsafe_allow_html=True)
                else:
                    st.markdown(_level_row("level-support", "#34d399", r["label"], fmt_price(r["value"], cur), pct),
                                unsafe_allow_html=True)
            if not placed:
                st.markdown(_level_row("level-current", "#60a5fa", "▶ Current price", fmt_price(price, cur), 0.0),
                            unsafe_allow_html=True)
            missing = [f"S {d:g}°" for d in GANN_DEGREES if b["levels"].get(f"S{d:g}") is None]
            if missing:
                st.caption("Not shown (√anchor − degree/180 would be negative at this price): " + ", ".join(missing))
            with st.expander("Ladder as a table"):
                show_df(pd.DataFrame([{"Level": r["label"], "Price": round(r["value"], 6),
                                       "% vs current": round((r["value"] - price) / price * 100, 3)}
                                      for r in sorted(ladder, key=lambda x: x["value"], reverse=True)]))

    # ---------------- Golden Pocket ----------------
    with st.container(border=True):
        st.markdown("<div class='card-header'>🟠 Golden Pocket Zone</div>"
                    "<div class='card-sub'>Confirmed pivot swings only · 50.0%–61.8% retracement band</div>",
                    unsafe_allow_html=True)
        gp = b.get("gp")
        if not s.get("gp_enabled", True):
            st.info("Golden Pocket is switched off in the sidebar.")
        elif not gp or not gp.get("active"):
            st.info(NO_GP_MESSAGE)
            if gp and gp.get("fetch_error"):
                st.caption("Data note: " + gp["fetch_error"])
            elif gp and gp.get("reason") and gp["reason"] != NO_GP_MESSAGE:
                st.caption(gp["reason"])
        else:
            css = {"Inside Golden Pocket": "pill-open", "Above Golden Pocket": "pill-delayed",
                   "Below Golden Pocket": "pill-bad"}[gp["price_status"]]
            st.markdown(f"<div class='gp-box'><b>{gp['direction']}</b> &nbsp; "
                        f"<span class='pill {css}'>{gp['price_status']}</span> &nbsp; "
                        f"<span style='color:#94a3b8'>pivot length {gp['pivot_len']}</span></div>",
                        unsafe_allow_html=True)
            g1, g2, g3, g4 = st.columns(4)
            g1.markdown(metric_html("Swing A", fmt_price(gp["swing_a"], cur), sub=str(gp["time_a"])[:16]),
                        unsafe_allow_html=True)
            g2.markdown(metric_html("Swing B", fmt_price(gp["swing_b"], cur), sub=str(gp["time_b"])[:16]),
                        unsafe_allow_html=True)
            g3.markdown(metric_html("50.0%", fmt_price(gp["fib_500"], cur), "amber"), unsafe_allow_html=True)
            g4.markdown(metric_html("61.8%", fmt_price(gp["fib_618"], cur), "amber"), unsafe_allow_html=True)
            h1, h2, h3 = st.columns(3)
            h1.markdown(metric_html("Zone range", f"{fmt_price(gp['zone_bottom'], cur)} – "
                                                  f"{fmt_price(gp['zone_top'], cur)}"), unsafe_allow_html=True)
            h2.markdown(metric_html("Distance to 50.0%", fmt_delta(gp["distance_to_50"], cur),
                                    sub=fmt_pct(gp["distance_to_50"] / gp["fib_500"] * 100)), unsafe_allow_html=True)
            h3.markdown(metric_html("Distance to 61.8%", fmt_delta(gp["distance_to_618"], cur),
                                    sub=fmt_pct(gp["distance_to_618"] / gp["fib_618"] * 100)), unsafe_allow_html=True)
            st.caption(f"Nearest pocket level: {gp['next_level_label']} at {fmt_price(gp['next_level'], cur)} · "
                       "positive distance = price is above that level.")
            period, interval = GP_TIMEFRAMES.get(s.get("gp_tf", "1 hour · 3 months"), ("3mo", "1h"))
            gdf, _err = fetch_history(symbol, period, interval)
            if not gdf.empty:
                show_fig(golden_pocket_figure(gdf, gp, int(s.get("gp_ext", 20)), price, cur))

    # ---------------- Plans, indicators, AI ----------------
    ind = technical_indicators(symbol)
    with st.container(border=True):
        st.markdown("<div class='card-header'>⚡ Trade plans from the Gann levels</div>"
                    "<div class='card-sub'>Built only from the real level set — educational, not advice</div>",
                    unsafe_allow_html=True)
        plans = generate_suggestions(price, ladder, ind.get("atr"))
        for plan in plans:
            bull = plan["direction"] == "bullish"
            st.markdown(f"<div class='plan-card {'plan-bull' if bull else 'plan-bear'}'>"
                        f"<b class='{'up' if bull else 'down'}'>{'▲ BULLISH' if bull else '▼ BEARISH'} PLAN</b>"
                        f"<div>Entry <b>{fmt_price(plan['entry'], cur)}</b> · Stop <b>{fmt_price(plan['stop'], cur)}</b></div>"
                        f"<div>T1 <b>{fmt_price(plan['t1'], cur)}</b> (R:R {plan['rr1']}) · "
                        f"T2 <b>{fmt_price(plan['t2'], cur)}</b> (R:R {plan['rr2']})</div>"
                        f"<div style='font-size:.72rem;color:#94a3b8'>{plan['basis']}</div></div>",
                        unsafe_allow_html=True)
        if not plans:
            st.info("Not enough levels around the current price to build a plan.")
        if ind.get("ok"):
            with st.expander("Technical read"):
                show_df(pd.DataFrame([
                    ("Trend", ind.get("trend") or "—"),
                    ("EMA 9 / 50", f"{fmt_price(ind.get('ema9'), cur)} / {fmt_price(ind.get('ema50'), cur)}"),
                    ("RSI(14)", f"{ind['rsi']:.1f}" if ind.get("rsi") is not None else "—"),
                    ("ATR(14)", fmt_price(ind.get("atr"), cur)),
                    ("Swing high / low (60d)", f"{fmt_price(ind.get('swing_high'), cur)} / "
                                               f"{fmt_price(ind.get('swing_low'), cur)}")],
                    columns=["Metric", "Value"]))
        st.markdown("**🤖 AI trading suggestion** (uses Gann + Golden Pocket + global headlines)")
        if wbutton("Generate AI suggestion", key="ai_sugg_btn"):
            with st.spinner("Asking Groq…"):
                try:
                    s.ai_suggestion = {"symbol": symbol, "text": ai_trading_suggestion(), "error": ""}
                except AIUnavailable as exc:
                    s.ai_suggestion = {"symbol": symbol, "text": "", "error": str(exc)}
                except Exception as exc:
                    s.ai_suggestion = {"symbol": symbol, "text": "", "error": friendly_ai_error(exc)}
        sugg = s.get("ai_suggestion")
        if sugg and sugg.get("symbol") == symbol:
            if sugg.get("text"):
                st.markdown(sugg["text"])
            else:
                st.warning(f"Groq AI is temporarily unavailable: {sugg.get('error')}")
        st.caption("Educational analysis only — no guarantee of returns.")

    # ---------------- Headlines (after Gann + Golden Pocket) ----------------
    with st.container(border=True):
        st.markdown("<div class='card-header'>📰 Global market headlines</div>"
                    "<div class='card-sub'>Market-wide news from all configured sources — not specific to "
                    "this symbol</div>", unsafe_allow_html=True)
        items = fetch_global_news().get("items", [])[:6]
        if not items:
            st.info("No headlines could be loaded right now (see the Global News page for feed status).")
        for it in items:
            sentiment = news_sentiment(it["title"], it.get("summary", ""))
            tone = sentiment.get("label", "Neutral")
            colour = sentiment.get("colour", "#fbbf24")
            link = f"<a href='{html.escape(it['link'])}' target='_blank' rel='noopener'>open ↗</a>" if it["link"] else ""
            st.markdown(f"<div class='news-card'><b>{html.escape(it['title'])}</b><div class='card-sub' "
                        f"style='margin:4px 0 0 0'>{html.escape(it['source'])} · {age_label(it['published'])} · "
                        f"<span style='color:{colour}'>{tone}</span> {link}</div></div>", unsafe_allow_html=True)


# ===========================================================================
# SECTION 17 - PAGE: HEAT MAP (follows the active market / exchange / universe)
# ===========================================================================
def page_heatmap() -> None:
    s = st.session_state
    market, exchange = s.active_market, s.active_exchange
    groups = heatmap_groups(market, exchange)
    if not groups:
        st.warning("No heat-map groups are defined for this market.")
        return
    with st.container(border=True):
        st.markdown(f"<div class='card-header'>Market heat map — {html.escape(market)} · {html.escape(exchange)}</div>"
                    "<div class='card-sub'>Follows the sidebar Market / Exchange / Universe. Colour = daily % change "
                    "as an intensity from 0 to 10 (green up, red down). Data only where Yahoo Finance returned it.</div>",
                    unsafe_allow_html=True)
        opts = ["All groups"] + list(groups.keys())
        default = s.active_universe if s.active_universe in groups else "All groups"
        c1, c2 = st.columns(2)
        with c1:
            group = st.selectbox("Sector group", opts, index=opts.index(default),
                                 key=f"hm_group_{market}_{exchange}_{s.active_universe}")
        with c2:
            default_scale = {M_CR: 8.0, M_FX: 1.0}.get(market, 3.0)
            scale = st.slider("% move that equals intensity 10", 0.5, 15.0, default_scale, 0.5,
                              key=f"hm_scale_{market}")
        chosen = groups if group == "All groups" else {group: groups[group]}
        if market == M_OTC:
            st.caption("OTC/CFD proxies: only symbols for which Yahoo Finance returns data are shown. "
                       "This is not a broker feed.")
        with st.spinner("Loading heat-map data…"):
            built: Dict[str, List[Dict[str, Any]]] = {}
            skipped: List[str] = []
            for g, syms in chosen.items():
                rows, sk = heatmap_rows(syms, scale)
                built[g] = rows
                skipped += sk
        fig = heatmap_figure(built)
        if fig is None:
            st.warning("Yahoo Finance returned no data for this group right now. Nothing is shown rather than "
                       "showing invented values.")
        else:
            show_fig(fig)
        if skipped:
            st.caption("No data returned for: " + ", ".join(sorted(set(skipped))))
        table = [{"Group": g, "Symbol": r["symbol"], "Name": r["name"],
                  "Price": fmt_price(r["price"], r["currency"]), "Change %": round(r["change_pct"], 2),
                  "Intensity (0-10)": r["intensity"]} for g, rows in built.items() for r in rows]
        if table:
            with st.expander("Data table"):
                show_df(pd.DataFrame(table))
            allrows = [r for rows in built.values() for r in rows]
            up = sum(1 for r in allrows if r["change_pct"] > 0)
            dn = sum(1 for r in allrows if r["change_pct"] < 0)
            st.caption(f"Breadth: {up} up / {dn} down / {len(allrows) - up - dn} flat "
                       f"({len(allrows)} instruments)")


# ===========================================================================
# ===========================================================================
# ===========================================================================
# SECTION 18a - STOCK NEWS + IMPACT (Script 1 News implementation)
# ===========================================================================
# ===========================================================================
# SECTION 9 - NEWS SERVICE (crash-proof parser)
# ---------------------------------------------------------------------------
# ROOT CAUSE OF THE ORIGINAL AttributeError (traceback line 945):
#     link = n.get("link") or (n.get("content") or {}).get("clickThroughUrl", {}).get("url") or "#"
# Yahoo returns BOTH shapes and sometimes junk:
#     {"title", "publisher", "link", "providerPublishTime"}                      (legacy)
#     {"id", "content": {"title", "provider": {"displayName"},
#                        "clickThroughUrl": {"url"}, ...}}                       (current)
# and `content` can be a str / list / None / int. Calling .get() on a str raised
# AttributeError and killed the whole script.
# Every field now goes through a typed accessor - one malformed article can
# never take down the app, and missing pieces degrade to "—"/"#".
# ===========================================================================
def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        for entry in value:
            text = _as_str(entry)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "title", "name", "url", "value"):
            text = _as_str(value.get(key))
            if text:
                return text
    return ""


def _first_url(*candidates: Any) -> str:
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("url", "href", "link"):
                url = _as_str(candidate.get(key))
                if url.startswith(("http://", "https://")):
                    return url
        elif isinstance(candidate, (list, tuple)):
            nested = _first_url(*candidate)
            if nested:
                return nested
        else:
            url = _as_str(candidate)
            if url.startswith(("http://", "https://")):
                return url
    return ""


def safe_news_link(item: Any) -> str:
    """The robust replacement for the line that crashed. Never raises."""
    if not isinstance(item, dict):
        return ""
    direct = _as_str(item.get("link"))
    if direct.startswith(("http://", "https://")):
        return direct
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("clickThroughUrl", "canonicalUrl", "previewUrl", "providerUrl"):
            url = _first_url(content.get(key))
            if url:
                return url
        url = _first_url(content.get("url"))
        if url:
            return url
    for key in ("clickThroughUrl", "canonicalUrl", "url"):
        url = _first_url(item.get(key))
        if url:
            return url
    return ""


def safe_news_image(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    content = _as_dict(item.get("content"))
    thumb = _as_dict(content.get("thumbnail"))
    resolutions = thumb.get("resolutions")
    if isinstance(resolutions, list):
        for entry in resolutions:
            url = _first_url(entry)
            if url:
                return url
    for key in ("thumbnail", "image", "img"):
        url = _first_url(item.get(key))
        if url:
            return url
    return ""


def parse_news_items(raw: Any, symbol: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """Normalize ANY news payload into a list of safe dicts. Never raises."""
    if isinstance(raw, dict):
        candidates: Any = raw.get("news") or raw.get("items") or raw.get("data") or []
    else:
        candidates = raw
    if not isinstance(candidates, (list, tuple)):
        return []
    items: List[Dict[str, Any]] = []
    for entry in candidates:
        try:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content") if isinstance(entry.get("content"), dict) else {}
            title = (_as_str(entry.get("title")) or _as_str(content.get("title"))
                     or _as_str(content.get("headline")) or _as_str(entry.get("headline")))
            summary = (_as_str(entry.get("summary")) or _as_str(entry.get("description"))
                       or _as_str(content.get("summary")) or _as_str(content.get("description")))
            publisher = (_as_str(entry.get("publisher")) or _as_str(entry.get("provider"))
                         or _as_str(_as_dict(content.get("provider")).get("displayName"))
                         or _as_str(_as_dict(content.get("provider")).get("name"))
                         or _as_str(entry.get("source")) or "Unknown source")
            link = safe_news_link(entry)
            image = safe_news_image(entry)
            stamp = (entry.get("providerPublishTime") or entry.get("published")
                     or content.get("pubDate") or content.get("datePublished")
                     or content.get("displayTime")
                     or entry.get("pubDate") or entry.get("time"))
            published = None
            if isinstance(stamp, datetime):
                # v3.1.1 FIX: already a normalized datetime (fetch_news re-parses
                # its own parsed items) - the old int/float/str-only chain dropped
                # it and every headline showed "time unknown".
                published = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
            elif isinstance(stamp, (int, float)):
                try:
                    published = datetime.fromtimestamp(float(stamp), tz=timezone.utc)
                except Exception:
                    published = None
            elif isinstance(stamp, str) and stamp.strip():
                text = stamp.strip().replace("Z", "+00:00")
                for parser in (lambda t: datetime.fromisoformat(t),
                               lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z"),
                               lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S")):
                    try:
                        published = parser(text)
                        break
                    except Exception:
                        continue
            if published is not None and published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if not title:
                continue
            items.append({
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "link": link,
                "image": image,
                "published": published,
                "age_label": _relative_age(published),
                "symbol": symbol,
            })
        except Exception as exc:      # a single bad article is logged, never fatal
            log_exception("parse news item", exc)
            continue
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        key = item["title"].lower()[:110]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda i: i["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return unique[:limit]


def _relative_age(published: Optional[datetime]) -> str:
    if not isinstance(published, datetime):
        return "time unknown"
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    seconds = (now - published).total_seconds()
    if seconds < 0:
        return "just now"
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d ago"
    return published.strftime("%d %b %Y")


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(query: str, yf_symbol: str = "", limit: int = 20) -> Dict[str, Any]:
    """
    Real news only. Provider order:
      1. yfinance ticker.news  (per-instrument)
      2. Yahoo Finance search endpoint with newsCount (per query / per symbol)
    If both are empty the UI says so - no fabricated headlines, ever.
    """
    items: List[Dict[str, Any]] = []
    sources: List[str] = []
    errors: List[str] = []

    if yf_symbol:
        try:
            raw = yf.Ticker(yf_symbol).news
            parsed = parse_news_items(raw, symbol=yf_symbol, limit=limit)
            if parsed:
                items.extend(parsed)
                sources.append("yfinance ticker.news")
        except Exception as exc:
            log_exception(f"ticker.news {yf_symbol}", exc)
            errors.append(safe_error(exc))

    for search_term in [t for t in (query, yf_symbol) if t]:
        if len(items) >= limit:
            break
        try:
            response = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": search_term, "quotesCount": 0, "newsCount": min(limit, 20),
                        "enableFuzzyQuery": "false", "newsQueryId": "news_cie_vespa"},
                headers={"User-Agent": HTTP_UA, "Accept": "application/json"}, timeout=10)
            response.raise_for_status()
            payload = response.json() if response.content else {}
            parsed = parse_news_items(payload, symbol=yf_symbol or search_term, limit=limit)
            if parsed:
                items.extend(parsed)
                sources.append(f"Yahoo news search ('{search_term}')")
        except Exception as exc:
            log_exception(f"news search {search_term}", exc)
            errors.append(safe_error(exc))

    combined = parse_news_items(items, symbol=yf_symbol, limit=limit)
    return {"items": combined, "sources": sources, "errors": errors,
            "ok": bool(combined), "fetched_at": datetime.now(timezone.utc).isoformat()}


# Two lexicons: general business words plus market-specific event words
# (results, guidance, fundraise, order book ...) so a headline like
# "Q2 results beat estimates" is not scored Neutral just because it lacks the word "surge".
GENERAL_BULLISH = ("surge", "rally", "beat", "beats", "growth", "record", "profit", "gain", "gains",
                   "strong", "expansion", "bullish", "rise", "rises", "high", "upbeat", "upgrade",
                   "outperform", "positive", "approval", "wins", "jump", "soar", "boost", "buyback",
                   "dividend", "partnership", "recovery", "rebound", "optimism", "inflow", "buy")
GENERAL_BEARISH = ("fall", "falls", "drop", "drops", "loss", "losses", "miss", "misses", "cut", "cuts",
                   "weak", "down", "decline", "crash", "fear", "bearish", "selloff", "sell-off",
                   "concern", "concerns", "downgrade", "probe", "fine", "penalty", "lawsuit", "fraud",
                   "default", "resign", "layoff", "layoffs", "slump", "plunge", "halt", "recall",
                   "sell", "outflow", "pessimism")
MARKET_BULLISH = ("results beat", "beats estimates", "profit rises", "profit jumps", "revenue up",
                  "margin expansion", "order win", "order book", "new order", "wins contract", "bags order",
                  "fundraise", "fund raising", "capital raise", "qip", "ipo", "stake sale", "buyback",
                  "bonus issue", "stock split", "target raised", "price target raised", "upgrade",
                  "initiates coverage", "accumulate", "add rating", "all-time high", "52-week high",
                  "capex", "expansion plan", "capacity addition", "approval received", "regulatory approval",
                  "tie-up", "tie up", "joint venture", "acquisition", "to acquire", "demerger", "inflow")
MARKET_BEARISH = ("results miss", "misses estimates", "profit falls", "profit drops", "revenue down",
                  "margin pressure", "guidance cut", "cuts guidance", "target cut", "price target cut",
                  "downgrade", "reduce rating", "block deal", "bulk deal", "promoter sells", "pledge",
                  "insider selling", "auditor resigns", "delisting", "default", "insolvency", "bankruptcy",
                  "regulatory action", "show-cause", "sebi probe", "tax raid", "penalty", "order cancelled",
                  "contract terminated", "recall", "outage", "strike", "halt", "trading halt",
                  "data breach", "impairment", "write-off", "writedown", "outflow")


def _lexicon_hits(text: str, words: Iterable[str]) -> List[str]:
    return [word for word in words if word in text]


def news_sentiment(title: str, extra_text: str = "") -> Dict[str, Any]:
    """
    Keyword sentiment over a general + market-event lexicon. Deliberately transparent:
    it is labelled a heuristic in the UI, and the matched words are returned so the
    score can be audited. The news page's "Summary & impact" action adds a model read.
    """
    text = f"{title or ''} {extra_text or ''}".lower()
    bullish = _lexicon_hits(text, GENERAL_BULLISH) + _lexicon_hits(text, MARKET_BULLISH)
    bearish = _lexicon_hits(text, GENERAL_BEARISH) + _lexicon_hits(text, MARKET_BEARISH)
    if not text.strip():
        return {"score": None, "label": "No headline text", "colour": "#94a3b8",
                "bullish_terms": [], "bearish_terms": []}
    score = 5.5 + 0.55 * len(bullish) - 0.55 * len(bearish)
    score = max(1.5, min(9.5, round(score, 1)))
    if score >= 6.5:
        label, colour = "Bullish", "#34d399"
    elif score <= 4.5:
        label, colour = "Bearish", "#f87171"
    else:
        label, colour = "Neutral", "#fbbf24"
    return {"score": score, "label": label, "colour": colour,
            "bullish_terms": bullish[:6], "bearish_terms": bearish[:6]}




def ai_impact_summary(title: str, symbol: str, currency: str = "USD") -> Tuple[Optional[str], str]:
    """Keep Script 2's verified Groq connection, but use Script 1's exact two-sentence prompt behavior."""
    system = ("You are a concise financial news explainer. In exactly two short sentences: (1) what this headline "
              "means, (2) how it could plausibly affect the named instrument. Hedge appropriately and never "
              f"invent figures that are not in the headline. Prices for this instrument are quoted in {currency}. "
              "Educational only, not advice.")
    try:
        return call_groq(system, [{"role": "user", "content": f"Headline: {title}\nInstrument: {symbol}"}],
                         max_tokens=700, temperature=0.3), ""
    except AIUnavailable as exc:
        return None, str(exc)
    except Exception as exc:
        return None, friendly_ai_error(exc)

def page_news() -> None:
    state = st.session_state
    yf_symbol = state.get("active_symbol", "")
    snapshot = {
        "name": state.get("active_display_name") or yf_symbol,
        "currency": state.get("active_currency") or "USD",
    }
    symbol = state.get("active_display_name") or (snapshot.get("name") or yf_symbol)
    currency = snapshot.get("currency") or "USD"

    with st.container(border=True):
        st.markdown("<div class='card-header'>News &amp; sentiment</div>"
                    "<div class='card-sub'>Headlines come straight from the configured provider. Nothing is invented — "
                    "when the provider returns nothing, this page says so.</div>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            topic = st.text_input("Search news", value=snapshot.get("name") or symbol,
                                  placeholder="e.g. Reliance Industries, semiconductor tariffs, RBI policy",
                                  key="news_topic")
        with col_b:
            st.write("")
            only_symbol = st.checkbox("Instrument feed only", value=False, key="news_instrument_only",
                                      help="Query the instrument's own news feed instead of the free-text topic.")

    query = yf_symbol if (only_symbol and yf_symbol) else (topic or yf_symbol)
    with st.spinner("Loading news…"):
        payload = fetch_news(query, "" if only_symbol else yf_symbol, limit=12)
    items = payload.get("items", [])

    st.markdown("<div class='card-sub'>"
                + (f"Source: {', '.join(payload.get('sources', []))}" if payload.get("sources")
                   else "No provider source returned data")
                + f" · fetched {datetime.now(timezone.utc).strftime('%H:%M UTC')}</div>",
                unsafe_allow_html=True)

    if not items:
        st.info("No news available from the provider for this query right now. "
                "This build never substitutes fabricated headlines.")
        if payload.get("errors"):
            with st.expander("Provider errors (technical)"):
                st.code("\n".join(payload["errors"]))
        return

    if "news_open" not in state:
        state.news_open = None
    state.setdefault("news_impact", {})

    for index, item in enumerate(items):
        sentiment = news_sentiment(item["title"], item.get("summary", ""))
        score_text = sentiment["score"] if sentiment["score"] is not None else "–"
        matched = (sentiment.get("bullish_terms") or []) + (sentiment.get("bearish_terms") or [])
        audit = ("matched: " + ", ".join(matched[:4])) if matched else "no keyword matched"
        st.markdown(f"""
        <div style="background:#0B1220;border:1px solid rgba(51,65,85,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px;display:flex;gap:14px;align-items:flex-start;">
            <div style="min-width:44px;height:44px;border-radius:12px;background:rgba(30,41,59,0.9);color:{sentiment['colour']};font-weight:700;font-size:1.05rem;display:flex;align-items:center;justify-content:center;">{score_text}</div>
            <div style="flex:1;">
                <div style="color:#f1f5f9;font-weight:600;font-size:0.95rem;line-height:1.35;">{html.escape(item['title'])}</div>
                <div style="margin-top:6px;font-size:0.78rem;color:#64748b;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span>{html.escape(item['publisher'])}</span><span>·</span><span>{item['age_label']}</span>
                    <span style="background:rgba(148,163,184,0.14);color:{sentiment['colour']};padding:2px 9px;border-radius:999px;font-weight:600;font-size:0.72rem;">{sentiment['label']} (keyword heuristic)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if item.get("summary"):
            st.caption(item["summary"][:400])
        action_left, action_right = st.columns([1, 1])
        with action_left:
            if st.button("Summary & impact", key=f"news_sum_{index}", use_container_width=True):
                state.news_open = index if state.news_open != index else None
                st.rerun()
        with action_right:
            if item.get("link"):
                st.markdown(f"<a href='{item['link']}' target='_blank' rel='noopener' "
                            f"style='display:block;text-align:center;padding:0.4rem 0.6rem;border-radius:8px;"
                            f"background:#1E293B;border:1px solid rgba(51,65,85,0.6);color:#e2e8f0;"
                            f"text-decoration:none;font-size:0.85rem;'>Open full article ↗</a>",
                            unsafe_allow_html=True)
            else:
                st.caption("No external link in the provider payload")

        if state.news_open == index:
            with st.container(border=True):
                st.markdown("<div class='card-sub'>Sentiment read</div>", unsafe_allow_html=True)
                st.caption(f"Keyword heuristic: {score_text}/10 · {sentiment['label']} · {audit}")
                st.markdown("<div class='card-sub'>Summary &amp; impact (generated by the configured AI model)</div>",
                            unsafe_allow_html=True)
                cache_key = f"{yf_symbol}|{item['title']}"
                if cache_key not in state.get("news_impact", {}):
                    state.news_impact[cache_key] = ai_impact_summary(
                        item["title"], symbol or yf_symbol, currency
                    )
                summary, ai_error = state.news_impact[cache_key]
                if summary:
                    st.markdown(summary)
                    st.caption("Model-generated interpretation — not a verified fact about the company.")
                else:
                    st.warning(f"AI summary unavailable: {ai_error or 'model not configured, or the provider returned an error.'} "
                               "The headline above is unchanged provider data.")


# SECTION 18 - PAGE: GLOBAL NEWS
# ===========================================================================
def page_global_news() -> None:
    """Global feed using the exact same News & sentiment card/interaction system."""
    state = st.session_state

    with st.container(border=True):
        st.markdown("<div class='card-header'>News &amp; sentiment · Global market</div>"
                    "<div class='card-sub'>Headlines come from the configured global RSS sources. "
                    "Nothing is invented — newest headlines are deduplicated and shown with the same impact "
                    "rating, Summary &amp; impact action, and article links as the stock News &amp; sentiment feed.</div>",
                    unsafe_allow_html=True)

    with st.spinner("Scanning global news sources…"):
        data = fetch_global_news()
    items = data.get("items", [])
    status = data.get("status", [])

    ok = sum(1 for x in status if x.get("ok"))
    st.markdown("<div class='card-sub'>"
                f"Source status: {ok}/{len(status)} feeds responded"
                f" · fetched {datetime.now(timezone.utc).strftime('%H:%M UTC')}</div>",
                unsafe_allow_html=True)

    if not items:
        st.warning("No news could be loaded from any global source right now. Try again in a few minutes.")
        if status:
            with st.expander("Source status / provider errors"):
                show_df(pd.DataFrame([
                    {"Source": x.get("source", ""),
                     "Articles": x.get("articles", 0),
                     "Status": "OK" if x.get("ok") else "Failed",
                     "Detail": x.get("error") or "—"}
                    for x in status
                ]))
        return

    state.setdefault("news_open", None)
    state.setdefault("news_impact", {})

    # Global news deliberately uses the same 0-10 heuristic, colour, badge,
    # Summary & impact toggle, AI explanation and full-article link as page_news().
    for index, item in enumerate(items[:40]):
        sentiment = news_sentiment(item.get("title", ""), item.get("summary", ""))
        score_text = sentiment["score"] if sentiment["score"] is not None else "–"
        matched = (sentiment.get("bullish_terms") or []) + (sentiment.get("bearish_terms") or [])
        audit = ("matched: " + ", ".join(matched[:4])) if matched else "no keyword matched"
        publisher = item.get("source") or "Unknown source"
        published = item.get("published")
        age = age_label(published)

        st.markdown(f"""
        <div style="background:#0B1220;border:1px solid rgba(51,65,85,0.5);border-radius:14px;padding:14px 16px;margin-bottom:8px;display:flex;gap:14px;align-items:flex-start;">
            <div style="min-width:44px;height:44px;border-radius:12px;background:rgba(30,41,59,0.9);color:{sentiment['colour']};font-weight:700;font-size:1.05rem;display:flex;align-items:center;justify-content:center;">{score_text}</div>
            <div style="flex:1;">
                <div style="color:#f1f5f9;font-weight:600;font-size:0.95rem;line-height:1.35;">{html.escape(item.get('title', ''))}</div>
                <div style="margin-top:6px;font-size:0.78rem;color:#64748b;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span>{html.escape(publisher)}</span><span>·</span><span>{html.escape(age)}</span>
                    <span style="background:rgba(148,163,184,0.14);color:{sentiment['colour']};padding:2px 9px;border-radius:999px;font-weight:600;font-size:0.72rem;">{sentiment['label']} (keyword heuristic)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if item.get("summary"):
            st.caption(str(item["summary"])[:400])

        action_left, action_right = st.columns([1, 1])
        with action_left:
            if st.button("Summary & impact", key=f"global_news_sum_{index}", use_container_width=True):
                state.news_open = index if state.news_open != index else None
                st.rerun()
        with action_right:
            link = item.get("link") or ""
            if link:
                st.markdown(f"<a href='{html.escape(link)}' target='_blank' rel='noopener' "
                            f"style='display:block;text-align:center;padding:0.4rem 0.6rem;border-radius:8px;"
                            f"background:#1E293B;border:1px solid rgba(51,65,85,0.6);color:#e2e8f0;"
                            f"text-decoration:none;font-size:0.85rem;'>Open full article ↗</a>",
                            unsafe_allow_html=True)
            else:
                st.caption("No external link in the provider payload")

        if state.news_open == index:
            with st.container(border=True):
                st.markdown("<div class='card-sub'>Sentiment read</div>", unsafe_allow_html=True)
                st.caption(f"Keyword heuristic: {score_text}/10 · {sentiment['label']} · {audit}")
                st.markdown("<div class='card-sub'>Summary &amp; impact (generated by the configured AI model)</div>",
                            unsafe_allow_html=True)

                cache_key = f"global|{item.get('title', '')}"
                if cache_key not in state.get("news_impact", {}):
                    with st.spinner("AI is analysing this headline…"):
                        state.news_impact[cache_key] = ai_impact_summary(
                            item.get("title", ""), "Global markets", "USD"
                        )
                summary, ai_error = state.news_impact[cache_key]
                if summary:
                    st.markdown(summary)
                    st.caption("Model-generated interpretation — not a verified fact about the market.")
                else:
                    st.warning(f"AI summary unavailable: {ai_error or 'model not configured, or the provider returned an error.'} "
                               "The headline above is unchanged provider data.")

    if status:
        with st.expander("Source status"):
            show_df(pd.DataFrame([
                {"Source": x.get("source", ""), "Articles": x.get("articles", 0),
                 "Status": "OK" if x.get("ok") else "Failed", "Detail": x.get("error") or "—"}
                for x in status
            ]))


# ===========================================================================
# SECTION 19 - PAGE: AI CHAT
# ===========================================================================
def page_ai() -> None:
    s = st.session_state
    st.markdown("<div class='card-header'>🤖 AI Chat</div>", unsafe_allow_html=True)
    st.selectbox("AI model", list(AI_MODEL_MAP.keys()), key="ai_model_label")
    key_ok = bool(groq_key())
    if not key_ok:
        st.warning("Groq API key is not configured. Add GROQ_KEY to Streamlit Secrets.")
    st.caption(f"Chatting about: **{s.active_symbol or 'no symbol'}** ({s.active_market}). "
               "Changing the symbol starts a fresh conversation.")
    with st.expander("Context the AI receives"):
        st.code(build_ai_context(), language="text")
    prompt = st.chat_input("Ask about the active symbol, the Gann levels, Golden Pocket or the news…",
                           disabled=not key_ok)
    if prompt and prompt.strip():
        s.chat_messages.append({"role": "user", "content": prompt.strip()[:4000]})
        with st.spinner("Groq is thinking…"):
            try:
                reply = call_groq(AI_SYSTEM + "\n\n" + build_ai_context(), s.chat_messages[-12:])
            except AIUnavailable as exc:
                reply = f"Groq AI is temporarily unavailable: {exc}"
            except Exception as exc:
                reply = f"Groq AI is temporarily unavailable: {friendly_ai_error(exc)}"
        s.chat_messages.append({"role": "assistant", "content": reply})
    if not s.chat_messages:
        st.info("Ask a question to begin. Educational analysis only — no guarantees.")
    for m in s.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if s.chat_messages and wbutton("🗑 Clear chat", key="chat_clear"):
        s.chat_messages = []
        st.rerun()


# ===========================================================================
# SECTION 20 - PAGE: HISTORY
# ===========================================================================
def page_history() -> None:
    s = st.session_state
    symbol = s.active_symbol
    if not symbol:
        st.info("Choose a symbol in the sidebar.")
        return
    cur = guess_currency(symbol)
    df, err = fetch_history(symbol, "3mo", "1d")
    st.markdown(f"<div class='card-header'>Historical data — {html.escape(display_name(symbol))} "
                f"({html.escape(symbol)})</div>", unsafe_allow_html=True)
    if df.empty or len(df) < 2:
        st.warning(err or "Not enough historical data (at least two daily bars are required).")
        return
    fig = go.Figure(_candles(df))
    fig.update_layout(height=340, yaxis=dict(side="right", gridcolor="rgba(51,65,85,.3)"),
                      xaxis=dict(gridcolor="rgba(51,65,85,.3)"), **_LAYOUT)
    show_fig(fig)
    d1, d2 = df.iloc[-2], df.iloc[-1]
    col1, col2 = st.columns(2)
    for col, label, row, idx in ((col1, "Previous session", d1, df.index[-2]), (col2, "Latest session", d2, df.index[-1])):
        with col.container(border=True):
            st.markdown(f"<div class='card-header'>{label} — {pd.Timestamp(idx).strftime('%d %b %Y')}</div>",
                        unsafe_allow_html=True)
            st.markdown(metric_html("Close", fmt_price(row["Close"], cur),
                                    sub=f"O {fmt_price(row['Open'], cur)} · H {fmt_price(row['High'], cur)} · "
                                        f"L {fmt_price(row['Low'], cur)} · Vol {fmt_volume(row['Volume'])}"),
                        unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='card-header'>Range &amp; momentum</div>", unsafe_allow_html=True)
        st.write(f"- Latest high **{'broke' if d2['High'] > d1['High'] else 'did not break'}** the previous high "
                 f"({fmt_price(d1['High'], cur)}); low **{'broke' if d2['Low'] < d1['Low'] else 'did not break'}** "
                 f"the previous low ({fmt_price(d1['Low'], cur)}).")
        chg = (d2["Close"] - d1["Close"]) / d1["Close"] * 100
        st.write(f"- Net change between the two closes: **{fmt_pct(chg)}**.")
        closes = df["Close"]
        if len(closes) >= 6:
            ma_now, ma_prev = closes.tail(5).mean(), closes.tail(6).head(5).mean()
            st.write(f"- 5-period average is **{'rising' if ma_now > ma_prev else 'falling'}** "
                     f"({fmt_price(float(ma_now), cur)}).")
    table = df.tail(15).iloc[::-1].copy()
    out = pd.DataFrame({"Date": [pd.Timestamp(i).strftime("%d %b %Y") for i in table.index]})
    for c in ("Open", "High", "Low", "Close"):
        out[c] = [fmt_price(v, cur) for v in table[c]]
    out["Volume"] = [fmt_volume(v) for v in table["Volume"]]
    show_df(out)
    st.caption(f"Prices are shown in {cur}, the instrument's own currency.")


# ===========================================================================
# SECTION 21 - PAGE: ALERTS (prefilled from the active symbol)
# ===========================================================================
def use_active_callback() -> None:
    s = st.session_state
    s["alert_symbol_w"] = s.get("active_symbol", "")
    s["alert_price_w"] = float(s.get("active_price") or 0.0)


def page_alerts() -> None:
    s = st.session_state
    if not telegram_ready():
        st.warning("Telegram is not configured — alerts still work inside the app. To get notifications add "
                   "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to Streamlit Secrets.")
    if s.get("_alert_prefill") or "alert_symbol_w" not in s:
        if not s.get("_alert_prefill") and "_alert_saved" in s:
            sym, prc, cond = s["_alert_saved"]
        else:
            sym, prc, cond = s.get("active_symbol", ""), s.get("active_price") or 0.0, "Above"
        s["alert_symbol_w"], s["alert_price_w"], s["alert_cond_w"] = sym, float(prc or 0.0), cond
        s["_alert_prefill"] = False

    with st.container(border=True):
        st.markdown("<div class='card-header'>🔔 Create new alert</div>", unsafe_allow_html=True)
        banner = st.empty()
        c1, c2, c3 = st.columns([2, 2, 1.4])
        with c1:
            st.text_input("Symbol / Ticker", key="alert_symbol_w")
        with c2:
            st.number_input("Alert price (editable)", min_value=0.0, step=0.01, format="%.4f", key="alert_price_w")
        with c3:
            st.selectbox("Condition", ["Above", "Below"], key="alert_cond_w")
        sym = normalize_symbol(s["alert_symbol_w"], s.active_market, s.active_exchange) or ""
        live = get_quote(sym) if sym else {"ok": False}
        cur = live.get("currency") or s.get("active_currency", "USD")
        if live.get("ok"):
            banner.markdown(f"**Creating alert for: {sym} — {fmt_price(live['price'], cur)}**  \n"
                            f"Current market price: {fmt_price(live['price'], cur)} ({fmt_pct(live.get('change_pct'))})")
        else:
            banner.warning(f"No live price available for {sym or 'this symbol'}.")
        b1, b2 = st.columns(2)
        with b1:
            st.button("↺ Use Active Symbol", key="alert_use_active", on_click=use_active_callback)
        with b2:
            if wbutton("＋ Set alert", key="alert_set"):
                ok, msg = add_alert(sym, float(s["alert_price_w"]), s["alert_cond_w"])
                (st.success("Alert saved.") if ok else st.warning(msg))
        s["_alert_saved"] = (s["alert_symbol_w"], s["alert_price_w"], s["alert_cond_w"])

    alerts = s.get("alerts", [])
    with st.container(border=True):
        st.markdown("<div class='card-header'>Your alerts</div>", unsafe_allow_html=True)
        if not alerts:
            st.caption("No alerts yet.")
        quotes = fetch_quotes_batch(tuple(sorted({a["symbol"] for a in alerts}))) if alerts else {}
        for a in list(alerts):
            q = quotes.get(a["symbol"]) or {}
            acur = q.get("currency") or guess_currency(a["symbol"])
            live_p = q.get("price")
            dist = (a["target"] - live_p) if live_p else None
            dist_pct = (dist / live_p * 100) if live_p else None
            state = ("Triggered" if a["status"] == "Triggered" else "Active")
            with st.container(border=True):
                st.markdown(f"**{html.escape(a['symbol'])}** — {html.escape(a.get('name') or '')} "
                            f"{status_pill('OPEN' if state == 'Active' else 'DELAYED')} {state}"
                            + (" · acknowledged" if a.get("acknowledged") else ""), unsafe_allow_html=True)
                st.caption(f"Live {fmt_price(live_p, acur)} · target {fmt_price(a['target'], acur)} · "
                           f"condition: {a['condition']} · distance to target {fmt_delta(dist, acur)} "
                           f"({fmt_pct(dist_pct)}) · created {a.get('created', '')}"
                           + (f" · triggered {a['triggered_at']}" if a.get("triggered_at") else "")
                           + (" · Telegram sent" if a.get("telegram_sent") else ""))
                k1, k2, k3 = st.columns(3)
                with k1:
                    if state == "Triggered" and not a.get("acknowledged"):
                        if wbutton("Acknowledge", key=f"ack_{a['id']}"):
                            a["acknowledged"] = True
                            persist_state()
                            st.rerun()
                with k2:
                    if state == "Triggered" and wbutton("Re-arm", key=f"rearm_{a['id']}"):
                        a.update({"status": "Active", "acknowledged": False, "telegram_sent": False,
                                  "telegram_attempted": False, "triggered_at": ""})
                        persist_state()
                        st.rerun()
                with k3:
                    if wbutton("Delete", key=f"del_{a['id']}"):
                        s.alerts = [x for x in s.alerts if x["id"] != a["id"]]
                        persist_state()
                        st.rerun()
    st.caption("Existing alerts are never changed by switching the active symbol. Alerts are evaluated whenever "
               "the app reruns; each alert sends at most one Telegram message.")


# ===========================================================================
# SECTION 22 - PAGE: WATCHLIST, GLOBAL, DIAGNOSTICS
# ===========================================================================
def open_and_go(symbol: str) -> None:
    open_symbol_callback(symbol)
    st.session_state["page"] = "Live Analysis"


def page_watchlist() -> None:
    s = st.session_state
    with st.container(border=True):
        st.markdown("<div class='card-header'>⭐ Watchlist</div><div class='card-sub'>Your saved list is never "
                    "changed by the sidebar selection.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1])
        with c1:
            new = st.text_input("Add symbol", key="watch_add", placeholder="RELIANCE.NS, AAPL, BTC-USD, ^GSPC, GC=F")
        with c2:
            st.write("")
            if wbutton("＋ Add", key="watch_add_btn") and new.strip():
                raw = new.strip().upper()
                sym = raw if any(t in raw for t in (".", "^", "=", "-")) else \
                    normalize_symbol(raw, s.active_market, s.active_exchange)
                if sym and sym not in s.watchlist:
                    s.watchlist.append(sym)
                    persist_state()
                st.rerun()
        quotes = fetch_quotes_batch(tuple(s.watchlist)) if s.watchlist else {}
        if not s.watchlist:
            st.caption("Your watchlist is empty.")
        for sym in list(s.watchlist):
            q = quotes.get(sym) or {}
            cur = q.get("currency") or guess_currency(sym)
            l, m, r = st.columns([4, 1.4, 1])
            l.markdown(f"**{html.escape(sym)}** — {html.escape(display_name(sym))}  \n"
                       f"{fmt_price(q.get('price'), cur)} "
                       f"<span class='{change_class(q.get('change_pct'))}'>{fmt_pct(q.get('change_pct'))}</span>"
                       + ("" if q.get("ok") else " <span class='down'>no data</span>"), unsafe_allow_html=True)
            m.button("Open", key=f"wopen_{sym}", on_click=open_and_go, args=(sym,))
            if r.button("🗑", key=f"wdel_{sym}"):
                s.watchlist = [x for x in s.watchlist if x != sym]
                persist_state()
                st.rerun()


def page_global() -> None:
    probes = [("^NSEI", "NIFTY 50"), ("^BSESN", "SENSEX"), ("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq Composite"),
              ("^DJI", "Dow Jones"), ("^FTSE", "FTSE 100"), ("^GDAXI", "DAX"), ("^N225", "Nikkei 225"),
              ("^HSI", "Hang Seng"), ("^AXJO", "ASX 200")]
    st.markdown("<div class='card-header'>🌐 Cross-market index overview</div>", unsafe_allow_html=True)
    quotes = fetch_quotes_batch(tuple(p for p, _ in probes))
    rows = []
    for sym, label in probes:
        q = quotes.get(sym) or {}
        rows.append({"Index": label, "Symbol": sym, "Level": fmt_price(q.get("price"), q.get("currency") or "USD"),
                     "% change": fmt_pct(q.get("change_pct")), "Status": "ok" if q.get("ok") else "no data"})
    show_df(pd.DataFrame(rows))
    st.caption("Index levels are points, not tradable prices. Dash / 'no data' means Yahoo returned nothing.")


def page_diagnostics() -> None:
    s = st.session_state
    st.markdown("<div class='card-header'>🛠 Diagnostics</div>", unsafe_allow_html=True)
    rows = [("App version", APP_VERSION), ("groq package installed", str(Groq is not None)),
            ("GROQ_KEY configured", str(bool(groq_key()))), ("AI model", AI_MODEL_MAP[DEFAULT_AI_LABEL]["model"]),
            ("Telegram configured", str(telegram_ready())), ("yfinance version", getattr(yf, "__version__", "?")),
            ("Active symbol", s.get("active_symbol") or "—"), ("Active market / exchange / universe",
                                                              f"{s.get('active_market')} / {s.get('active_exchange')} / "
                                                              f"{s.get('active_universe')}"),
            ("News sources configured", str(len(NEWS_FEEDS))), ("State file", str(STATE_FILE))]
    show_df(pd.DataFrame(rows, columns=["Item", "Value"]))
    if wbutton("🧪 Test Groq connection", key="diag_groq"):
        try:
            st.success("Groq replied: " + call_groq("Reply with OK only.", [{"role": "user", "content": "ping"}],
                                                    max_tokens=200)[:80])
        except AIUnavailable as exc:
            st.error(f"Groq AI is temporarily unavailable: {exc}")
    st.caption("Market data: Yahoo Finance via yfinance — not guaranteed real-time; LIVE is only shown when the "
               "market is open and the newest 15-minute bar is recent.")


# ===========================================================================
# SECTION 23 - MAIN
# ===========================================================================
PAGES = ["Live Analysis", "Heatmap", "News", "AI Chat", "History", "Alerts", "Watchlist", "Global",
         "Diagnostics"]


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    init_state()
    try:
        render_sidebar()
    except Exception as exc:
        log_exception("sidebar", exc)
        st.sidebar.error("The sidebar hit an unexpected error and was contained.")
        with st.sidebar.expander("Technical detail"):
            st.code(safe_error(exc))
    s = st.session_state
    try:
        fired = evaluate_alerts()
        for text in fired:
            st.toast(f"🔔 Alert triggered: {text}")
    except Exception as exc:
        log_exception("alerts", exc)
    title = f"{s.get('active_symbol') or 'Terminal'} · AI Trade Terminal"
    try:
        components.html(f"<script>window.parent.document.title={json.dumps(title)};</script>", height=0)
    except Exception:
        pass
    left, right = st.columns([3, 2])
    left.markdown(f"### ⚡ {html.escape(s.get('active_display_name') or 'AI Trade Terminal')} "
                  f"<span style='color:#64748b;font-size:.9rem'>{html.escape(s.get('active_symbol') or '')}</span>",
                  unsafe_allow_html=True)
    right.markdown(f"<div style='text-align:right' class='metric-value {change_class(s.get('active_change_pct'))}'>"
                   f"{fmt_price(s.get('active_price'), s.get('active_currency', 'USD'))} "
                   f"<span style='font-size:.85rem'>{fmt_pct(s.get('active_change_pct'))}</span></div>",
                   unsafe_allow_html=True)
    if s.get("page") not in PAGES:
        s["page"] = PAGES[0]
    page = st.radio("Page", PAGES, horizontal=True, label_visibility="collapsed", key="page")
    handlers = {"Live Analysis": page_live, "Heatmap": page_heatmap, "News": page_news, "AI Chat": page_ai,
                "History": page_history, "Alerts": page_alerts, "Watchlist": page_watchlist,
                "Global": page_global, "Diagnostics": page_diagnostics}
    try:
        handlers[page]()
    except Exception as exc:
        log_exception(f"page {page}", exc)
        st.error("This section hit an unexpected error and was contained. The rest of the terminal still works.")
        with st.expander("Technical detail (secrets redacted)"):
            st.code(safe_error(exc))
    st.markdown("<div style='text-align:center;color:#475569;font-size:.75rem;margin-top:1.5rem'>"
                f"AI Trade Terminal v{APP_VERSION} · Yahoo Finance via yfinance (may be delayed) · AI: Groq · "
                "educational tool, not investment advice.</div>", unsafe_allow_html=True)


if os.environ.get("TERMINAL_SKIP_MAIN") != "1":
    main()
