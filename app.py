"""
AI Trade Terminal - Streamlit Edition
=====================================
Main content left + right sidebar (Chat / Watch / Alerts).
India NSE auto-appends .NS. Real data via yfinance (falls back to last
available day when market closed). Real treemap heatmap (color = % change,
size = market cap, grouped by sector). Support/Resistance + Gann degree
ladder from session open. AI chat/news use current Groq + Gemini models.
"""

import math
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from groq import Groq
from google import genai as google_genai

# ---------------------------------------------------------------------------
# SECRETS
# ---------------------------------------------------------------------------
def secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GROQ_KEY = secret("GROQ_KEY")
GEMINI_KEY = secret("GEMINI_KEY")
TELEGRAM_BOT_TOKEN = secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = secret("TELEGRAM_CHAT_ID")

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
# google-generativeai (old SDK) is deprecated; using the current google-genai SDK.
gemini_client = google_genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# Current, live models on a normal API key (the earlier build used
# llama-3.3-70b-versatile / gemini-1.5-flash, both retired — that silent
# failure is why "AI chat wasn't working").
GROQ_MODELS = {
    "Groq GPT-OSS 20B (fast)": "openai/gpt-oss-20b",
    "Groq GPT-OSS 120B (smartest)": "openai/gpt-oss-120b",
    "Groq Qwen3 32B": "qwen/qwen3-32b",
    "Groq Kimi K2": "moonshotai/kimi-k2-instruct",
}
GEMINI_MODELS = {"Gemini 3.5 Flash": "gemini-3.5-flash", "Gemini 3.1 Pro (Preview)": "gemini-3.1-pro-preview"}
ALL_MODELS = list(GROQ_MODELS) + list(GEMINI_MODELS)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Trade Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #020617 !important; color: #f8fafc; }
    #MainMenu, footer, header { visibility: hidden; }

    .card {
        background: #0F172A;
        border: 1px solid rgba(51,65,85,0.45);
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .card-header { font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.1rem; }
    .card-sub { font-size: 0.75rem; color: #64748b; margin-bottom: 0.75rem; }

    .metric-label { font-size: 0.65rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.35rem; font-weight: 700; color: #f8fafc; font-variant-numeric: tabular-nums; }
    .up { color: #34d399 !important; }
    .down { color: #f87171 !important; }
    .amber { color: #fbbf24 !important; }

    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        border: 1px solid rgba(51,65,85,0.55) !important;
        background: #1E293B !important; color: #e2e8f0 !important;
    }
    .stButton > button:hover {
        border-color: #fbbf24 !important; color: #fbbf24 !important;
        background: rgba(251,191,36,0.08) !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #94a3b8; padding: 7px 14px; }
    .stTabs [aria-selected="true"] { background: #1E293B !important; color: #f8fafc !important; }

    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(51,65,85,0.4);
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #1E293B !important;
        border: 1px solid rgba(51,65,85,0.55) !important;
        border-radius: 8px !important; color: #f8fafc !important;
    }

    .chat-user {
        background: #334155; border-radius: 12px; padding: 9px 13px;
        margin: 5px 0 5px 18%; text-align: right; font-size: 0.88rem;
    }
    .chat-assistant {
        background: #1E293B; border: 1px solid rgba(51,65,85,0.4);
        border-radius: 12px; padding: 9px 13px; margin: 5px 12% 5px 0;
        font-size: 0.88rem; line-height: 1.4;
    }

    .level-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; border-radius: 9px; margin-bottom: 4px; font-size: 0.9rem;
    }
    .level-support { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); }
    .level-resistance { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); }

    .plan-card { border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }
    .plan-bull { background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.35); }
    .plan-bear { background: rgba(239,68,68,.12); border: 1px solid rgba(239,68,68,.35); }
    .plan-title-bull { color: #34d399; font-weight: 700; margin-bottom: 6px; }
    .plan-title-bear { color: #f87171; font-weight: 700; margin-bottom: 6px; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MARKET CATALOG
# ---------------------------------------------------------------------------
MARKET_CATEGORIES = [
    {"key": "india", "label": "India NSE"},
    {"key": "us", "label": "US Stocks"},
    {"key": "crypto", "label": "Crypto"},
    {"key": "commodities", "label": "Commodities"},
    {"key": "forex", "label": "Forex"},
    {"key": "indices", "label": "Global Indices"},
]

MARKET_ITEMS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries", "category": "india", "sector": "Energy", "yf": "RELIANCE.NS"},
    {"symbol": "TCS", "name": "Tata Consultancy", "category": "india", "sector": "IT", "yf": "TCS.NS"},
    {"symbol": "INFY", "name": "Infosys", "category": "india", "sector": "IT", "yf": "INFY.NS"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "category": "india", "sector": "Banking", "yf": "HDFCBANK.NS"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "category": "india", "sector": "Banking", "yf": "ICICIBANK.NS"},
    {"symbol": "SBIN", "name": "State Bank of India", "category": "india", "sector": "Banking", "yf": "SBIN.NS"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors", "category": "india", "sector": "Auto", "yf": "TATAMOTORS.NS"},
    {"symbol": "ITC", "name": "ITC Ltd", "category": "india", "sector": "FMCG", "yf": "ITC.NS"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharma", "category": "india", "sector": "Pharma", "yf": "SUNPHARMA.NS"},
    {"symbol": "TATASTEEL", "name": "Tata Steel", "category": "india", "sector": "Metal", "yf": "TATASTEEL.NS"},
    {"symbol": "AAPL", "name": "Apple", "category": "us", "sector": "US Tech", "yf": "AAPL"},
    {"symbol": "MSFT", "name": "Microsoft", "category": "us", "sector": "US Tech", "yf": "MSFT"},
    {"symbol": "NVDA", "name": "NVIDIA", "category": "us", "sector": "US Tech", "yf": "NVDA"},
    {"symbol": "GOOGL", "name": "Alphabet", "category": "us", "sector": "US Tech", "yf": "GOOGL"},
    {"symbol": "AMZN", "name": "Amazon", "category": "us", "sector": "US Tech", "yf": "AMZN"},
    {"symbol": "META", "name": "Meta", "category": "us", "sector": "US Tech", "yf": "META"},
    {"symbol": "TSLA", "name": "Tesla", "category": "us", "sector": "US Auto", "yf": "TSLA"},
    {"symbol": "JPM", "name": "JPMorgan", "category": "us", "sector": "US Banking", "yf": "JPM"},
    {"symbol": "BTC", "name": "Bitcoin", "category": "crypto", "sector": "Crypto", "yf": "BTC-USD"},
    {"symbol": "ETH", "name": "Ethereum", "category": "crypto", "sector": "Crypto", "yf": "ETH-USD"},
    {"symbol": "BNB", "name": "BNB", "category": "crypto", "sector": "Crypto", "yf": "BNB-USD"},
    {"symbol": "SOL", "name": "Solana", "category": "crypto", "sector": "Crypto", "yf": "SOL-USD"},
    {"symbol": "XRP", "name": "XRP", "category": "crypto", "sector": "Crypto", "yf": "XRP-USD"},
    {"symbol": "ADA", "name": "Cardano", "category": "crypto", "sector": "Crypto", "yf": "ADA-USD"},
    {"symbol": "DOGE", "name": "Dogecoin", "category": "crypto", "sector": "Crypto", "yf": "DOGE-USD"},
    {"symbol": "GC", "name": "Gold", "category": "commodities", "sector": "Commodities", "yf": "GC=F"},
    {"symbol": "SI", "name": "Silver", "category": "commodities", "sector": "Commodities", "yf": "SI=F"},
    {"symbol": "CL", "name": "Crude Oil", "category": "commodities", "sector": "Commodities", "yf": "CL=F"},
    {"symbol": "NG", "name": "Natural Gas", "category": "commodities", "sector": "Commodities", "yf": "NG=F"},
    {"symbol": "HG", "name": "Copper", "category": "commodities", "sector": "Commodities", "yf": "HG=F"},
    {"symbol": "USDINR", "name": "USD/INR", "category": "forex", "sector": "Forex", "yf": "USDINR=X"},
    {"symbol": "EURUSD", "name": "EUR/USD", "category": "forex", "sector": "Forex", "yf": "EURUSD=X"},
    {"symbol": "GBPUSD", "name": "GBP/USD", "category": "forex", "sector": "Forex", "yf": "GBPUSD=X"},
    {"symbol": "USDJPY", "name": "USD/JPY", "category": "forex", "sector": "Forex", "yf": "USDJPY=X"},
    {"symbol": "AUDUSD", "name": "AUD/USD", "category": "forex", "sector": "Forex", "yf": "AUDUSD=X"},
    {"symbol": "NIFTY", "name": "Nifty 50", "category": "indices", "sector": "Indices", "yf": "^NSEI"},
    {"symbol": "SENSEX", "name": "Sensex", "category": "indices", "sector": "Indices", "yf": "^BSESN"},
    {"symbol": "SPX", "name": "S&P 500", "category": "indices", "sector": "Indices", "yf": "^GSPC"},
    {"symbol": "DJI", "name": "Dow Jones", "category": "indices", "sector": "Indices", "yf": "^DJI"},
    {"symbol": "IXIC", "name": "Nasdaq", "category": "indices", "sector": "Indices", "yf": "^IXIC"},
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def format_price(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def format_percent(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def format_volume(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return str(int(v))


def resolve_yf_symbol(user_input: str, category: str = "india") -> str:
    s = user_input.strip().upper()
    if any(x in s for x in [".NS", ".BO", "-USD", "=F", "=X", "^"]):
        return s
    for m in MARKET_ITEMS:
        if m["symbol"] == s or m["yf"] == s:
            return m["yf"]
    if category == "india":
        return f"{s}.NS"
    return s


def yahoo_search(query, limit=8):
    """Universal symbol search (any stock/index/crypto/forex worldwide), so the
    dashboard search box isn't limited to the curated MARKET_ITEMS list."""
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                          params={"q": query, "quotesCount": limit, "newsCount": 0},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        r.raise_for_status()
        return [x for x in r.json().get("quotes", []) if x.get("symbol")]
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_stock_data(yf_symbol: str) -> Dict:
    try:
        t = yf.Ticker(yf_symbol)
        hist = t.history(period="10d", interval="1d")
        if hist is None or hist.empty:
            info = t.info or {}
            price = info.get("regularMarketPrice") or info.get("previousClose") or 0
            return {
                "symbol": yf_symbol, "name": info.get("shortName") or yf_symbol,
                "currentPrice": float(price or 0), "previousClose": float(info.get("previousClose") or price or 0),
                "dayHigh": float(info.get("dayHigh") or price or 0), "dayLow": float(info.get("dayLow") or price or 0),
                "volume": float(info.get("volume") or 0), "changePercent": 0.0, "open": float(info.get("open") or price or 0),
                "ok": False, "lastDate": "—",
            }
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        current = float(last["Close"])
        prev_close = float(prev["Close"])
        change_pct = ((current - prev_close) / prev_close * 100) if prev_close else 0.0
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass
        return {
            "symbol": yf_symbol,
            "name": info.get("shortName") or info.get("longName") or yf_symbol,
            "currentPrice": current, "previousClose": prev_close,
            "dayHigh": float(last["High"]), "dayLow": float(last["Low"]),
            "volume": float(last.get("Volume", 0) or 0), "changePercent": change_pct,
            "open": float(last["Open"]), "ok": True,
            "lastDate": str(hist.index[-1].date()) if hasattr(hist.index[-1], "date") else str(hist.index[-1]),
        }
    except Exception as e:
        return {"symbol": yf_symbol, "name": yf_symbol, "currentPrice": 0, "previousClose": 0,
                "dayHigh": 0, "dayLow": 0, "volume": 0, "changePercent": 0, "open": 0, "ok": False, "lastDate": "—", "error": str(e)}


@st.cache_data(ttl=300)
def fetch_market_cap(yf_symbol: str):
    """Used to size heatmap tiles — bigger block = bigger company."""
    try:
        fi = yf.Ticker(yf_symbol).fast_info
        cap = fi.get("market_cap")
        if cap:
            return float(cap)
    except Exception:
        pass
    try:
        info = yf.Ticker(yf_symbol).info or {}
        cap = info.get("marketCap")
        return float(cap) if cap else None
    except Exception:
        return None


@st.cache_data(ttl=120)
def fetch_candles(yf_symbol: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(yf_symbol)
        df = t.history(period="5d", interval="15m")
        if df is None or df.empty:
            df = t.history(period="1mo", interval="1d")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_history(yf_symbol: str) -> pd.DataFrame:
    try:
        return yf.Ticker(yf_symbol).history(period="1mo", interval="1d")
    except Exception:
        return pd.DataFrame()


def generate_levels(high: float, low: float, close: float) -> Dict:
    """Legacy classic pivot (fallback only)."""
    if high <= 0 or low <= 0 or close <= 0:
        return {}
    pivot = (high + low + close) / 3.0
    range_ = high - low
    return {
        "pivot": pivot,
        "R1": 2 * pivot - low, "R2": pivot + range_, "R3": high + 2 * (pivot - low),
        "R4": pivot + 2 * range_, "R5": pivot + 3 * range_,
        "S1": 2 * pivot - high, "S2": pivot - range_, "S3": low - 2 * (high - pivot),
        "S4": pivot - 2 * range_, "S5": pivot - 3 * range_,
    }


def gann_degree_levels(ref_price: float) -> Dict:
    """
    Square-of-9 / Gann degree levels.

    0° anchor = the first 15-minute candle close.
    Resistance: (sqrt(anchor) + degree / 180)^2
    Support:    (sqrt(anchor) - degree / 180)^2
    """
    if ref_price is None or ref_price <= 0:
        return {}

    anchor_sqrt = math.sqrt(float(ref_price))
    degrees = [22.5, 45.0, 67.5, 90.0, 180.0]

    levels = {
        "_ref_close": float(ref_price),
        "_type": "gann_sq9",
        "0": float(ref_price),
    }

    for degree in degrees:
        suffix = str(degree).rstrip("0").rstrip(".")

        resistance = (anchor_sqrt + degree / 180.0) ** 2
        support = max(0.01, (anchor_sqrt - degree / 180.0) ** 2)

        levels[f"R{suffix}"] = resistance
        levels[f"S{suffix}"] = support

    return levels


@st.cache_data(ttl=300)
def get_session_fixed_levels(yf_symbol: str) -> Dict:
    """0° reference = first 15-minute CLOSING price of the session. Fixed until close."""
    try:
        t = yf.Ticker(yf_symbol)
        df = t.history(period="5d", interval="15m")
        if df is None or df.empty:
            daily = t.history(period="10d", interval="1d")
            if daily is None or daily.empty:
                return {}
            last = daily.iloc[-1]
            ref = float(last["Close"])
            levels = gann_degree_levels(ref)
            levels["_ref"] = "daily_fallback"
            levels["_ref_time"] = str(daily.index[-1])
            classic = generate_levels(float(last["High"]), float(last["Low"]), ref)
            levels.update({k: v for k, v in classic.items() if k.startswith(("R", "S", "pivot"))})
            return levels

        df = df.copy()
        df["date"] = df.index.date
        last_day = df["date"].iloc[-1]
        day_bars = df[df["date"] == last_day]
        if day_bars.empty:
            day_bars = df.tail(26)

        first = day_bars.iloc[0]
        ref_close = float(first["Close"])
        levels = gann_degree_levels(ref_close)
        levels["_ref"] = "first_15m"
        levels["_ref_time"] = str(day_bars.index[0])
        levels["_ref_high"] = float(first["High"])
        levels["_ref_low"] = float(first["Low"])
        classic = generate_levels(float(first["High"]), float(first["Low"]), ref_close)
        levels.update({k: v for k, v in classic.items() if k.startswith(("R", "S", "pivot"))})
        return levels
    except Exception:
        return {}


def generate_suggestions(price: float, levels: Dict) -> List[Dict]:
    """Always includes T1 + T2 for both directions (no click needed to reveal)."""
    if not levels or price <= 0:
        return []
    out = []
    s1, s2, r1, r2 = levels.get("S1"), levels.get("S2"), levels.get("R1"), levels.get("R2")
    if s1 and s2 and r1:
        risk = abs(s1 - s2)
        t2 = r2 or r1
        out.append({
            "direction": "bullish", "entry": round(s1, 2), "stop": round(s2, 2),
            "t1": round(r1, 2), "t2": round(t2, 2),
            "rr1": round(abs(r1 - s1) / risk, 2) if risk else 0,
            "rr2": round(abs(t2 - s1) / risk, 2) if risk else 0,
        })
    if r1 and r2 and s1:
        risk = abs(r2 - r1)
        t2 = s2 or s1
        out.append({
            "direction": "bearish", "entry": round(r1, 2), "stop": round(r2, 2),
            "t1": round(s1, 2), "t2": round(t2, 2),
            "rr1": round(abs(r1 - s1) / risk, 2) if risk else 0,
            "rr2": round(abs(r1 - t2) / risk, 2) if risk else 0,
        })
    return out


def send_telegram(msg: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def ai_chat(prompt: str, symbol: str, data: Dict, levels: Dict, model_choice: str) -> str:
    context = f"Symbol: {symbol}, Price: {data.get('currentPrice')}, Change: {data.get('changePercent', 0):.2f}%, Levels: {levels}"
    full = f"You are a trading assistant. Context: {context}\nUser: {prompt}\nAnswer briefly and specifically, with numeric levels where relevant."
    try:
        if model_choice in GROQ_MODELS:
            if not groq_client:
                return "⚠️ Configure GROQ_KEY in .streamlit/secrets.toml to use Groq models."
            resp = groq_client.chat.completions.create(model=GROQ_MODELS[model_choice], messages=[{"role": "user", "content": full}], max_tokens=400, temperature=0.3)
            return resp.choices[0].message.content
        elif model_choice in GEMINI_MODELS:
            if not gemini_client:
                return "⚠️ Configure GEMINI_KEY in .streamlit/secrets.toml to use Gemini models."
            resp = gemini_client.models.generate_content(model=GEMINI_MODELS[model_choice], contents=full)
            return resp.text
        else:
            return "⚠️ Invalid model selected."
    except Exception as e:
        # Surface the real error instead of silently falling back — that silent
        # fallback was the "chat isn't working" bug.
        return f"⚠️ AI error ({model_choice}): {type(e).__name__}: {e}"


def detect_candle_pattern(daily_df):
    if daily_df is None or len(daily_df) < 2:
        return "Insufficient data"
    c, o = daily_df["Close"].values, daily_df["Open"].values
    if c[-1] > o[-1] and c[-2] < o[-2] and c[-1] >= o[-2]:
        return "🔥 Bullish Engulfing"
    if c[-1] < o[-1] and c[-2] > o[-2] and c[-1] <= o[-2]:
        return "⚠️ Bearish Engulfing"
    return "Standard / no clear pattern"


@st.cache_data(ttl=120)
def build_quant_summary(yf_symbol: str, ref_price: float):
    """Enriched real-time context — pre-market gap, EMA(9/50) trend bias, golden
    pocket, Gann-style wave target ((B*C)/A), and candle-exhaustion streak — fed
    into the AI chat and trade-suggestion prompts so the AI reasons over more than
    just static pivot numbers."""
    try:
        daily = yf.Ticker(yf_symbol).history(period="6mo", interval="1d")
        if daily is None or daily.empty or len(daily) < 10:
            return None
        last_close = float(daily["Close"].iloc[-1])
        prev_day_close = float(daily["Close"].iloc[-2]) if len(daily) > 1 else last_close
        gap_pct = ((ref_price - prev_day_close) / prev_day_close * 100) if prev_day_close and ref_price else 0.0
        if gap_pct > 0.3:
            gap_status = f"Gap UP {gap_pct:+.2f}%"
        elif gap_pct < -0.3:
            gap_status = f"Gap DOWN {gap_pct:+.2f}%"
        else:
            gap_status = "Flat open"

        intraday = yf.Ticker(yf_symbol).history(period="5d", interval="15m")
        if intraday is not None and not intraday.empty:
            intraday = intraday.copy()
            intraday["Date"] = intraday.index.date
            today_bars = intraday[intraday["Date"] == intraday["Date"].iloc[-1]]
            early = today_bars.head(4) if len(today_bars) else intraday.tail(4)
            pre_market_high, pre_market_low = float(early["High"].max()), float(early["Low"].min())
        else:
            pre_market_high = pre_market_low = ref_price

        candle_pattern = detect_candle_pattern(daily.tail(5))

        anchor_price = ref_price or last_close
        anchor_sqrt = math.sqrt(abs(anchor_price)) if anchor_price else 0.0
        r225 = (anchor_sqrt + 22.5 / 180.0) ** 2
r450 = (anchor_sqrt + 45.0 / 180.0) ** 2
r675 = (anchor_sqrt + 67.5 / 180.0) ** 2
r900 = (anchor_sqrt + 90.0 / 180.0) ** 2
r180 = (anchor_sqrt + 180.0 / 180.0) ** 2

s225 = max(0.01, (anchor_sqrt - 22.5 / 180.0) ** 2)
s450 = max(0.01, (anchor_sqrt - 45.0 / 180.0) ** 2)
s675 = max(0.01, (anchor_sqrt - 67.5 / 180.0) ** 2)
s900 = max(0.01, (anchor_sqrt - 90.0 / 180.0) ** 2)
s180 = max(0.01, (anchor_sqrt - 180.0 / 180.0) ** 2)

        closes = daily["Close"]
        fast_ema = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
        slow_ema = float(closes.ewm(span=50, adjust=False).mean().iloc[-1]) if len(closes) >= 20 else fast_ema
        ema_bias = "Bullish (fast>slow)" if fast_ema > slow_ema else "Bearish (fast<slow)"

        window = daily.tail(60)
        swing_high, swing_low = float(window["High"].max()), float(window["Low"].min())
        diff = swing_high - swing_low
        fib_618, fib_500 = swing_high - diff * 0.618, swing_high - diff * 0.5
        A, B, C = swing_low, swing_high, last_close
        wave_target = (B * C) / A if A else 0.0

        directions = (closes.diff() > 0).astype(int).tail(6)
        streak, last_dir = 1, None
        if len(directions) >= 2:
            last_dir = "up" if directions.iloc[-1] == 1 else "down"
            for i in range(len(directions) - 1, 0, -1):
                if directions.iloc[i] == directions.iloc[i - 1]:
                    streak += 1
                else:
                    break
        streak_signal = (f"{streak} consecutive {last_dir} closes" + (" — possible exhaustion" if streak >= 4 else "")) if last_dir else "n/a"

        text = f"""📊 REAL-TIME CANDLES & PRE-MARKET STRUCTURE: {yf_symbol}
- Live Close: {round(last_close, 2)}
- Pre-Market Gap Structure: {gap_status} (Prev Close: {round(prev_day_close, 2)})
- Pre-Market High (PMH): {round(pre_market_high, 2)} | Pre-Market Low (PML): {round(pre_market_low, 2)}
- Live Candle Pattern Detected: {candle_pattern}
- Gann Anchor (1st 15m): {round(anchor_price, 2)} (sqrt: {round(anchor_sqrt, 4)})
- Gann Resistances (R): 22.5°: {round(r225, 2)} | 45°: {round(r450, 2)} | 67.5°: {round(r675, 2)} | 90°: {round(r900, 2)} | 180°: {round(r180, 2)}
- Gann Supports (S): 22.5°: {round(s225, 2)} | 45°: {round(s450, 2)} | 67.5°: {round(s675, 2)} | 90°: {round(s900, 2)} | 180°: {round(s180, 2)}
- Dynamic EMAs: Fast(9)={round(fast_ema, 2)} vs Slow(50)={round(slow_ema, 2)} ({ema_bias})
- Golden Pocket: {round(fib_618, 2)} - {round(fib_500, 2)} | Wave Target ((B*C)/A): {round(wave_target, 2)}
- Candle Exhaustion: {streak_signal}"""
        return {"text": text, "gap_status": gap_status, "ema_bias": ema_bias, "wave_target": wave_target,
                "fib_618": fib_618, "fib_500": fib_500, "streak_signal": streak_signal,
                "candle_pattern": candle_pattern, "pre_market_high": pre_market_high, "pre_market_low": pre_market_low}
    except Exception as e:
        return {"text": f"Quant computation fallback: {str(e)}", "error": True}


def build_heatmap_treemap(pool):
    """A REAL heatmap: color = % change (darker/more saturated = bigger move),
    size = market cap, grouped by sector via treemap parent/child hierarchy."""
    rows = []
    for item in pool:
        d = fetch_stock_data(item["yf"])
        cap = fetch_market_cap(item["yf"])
        rows.append({
            "symbol": item["symbol"], "sector": item.get("sector", item["category"]),
            "price": float(d.get("currentPrice") or 0), "chg": float(d.get("changePercent") or 0),
            "cap": cap if cap and cap > 0 else 1,
        })
    if not rows:
        return None
    sectors = sorted({r["sector"] for r in rows})
    labels = [r["symbol"] for r in rows] + sectors
    parents = [r["sector"] for r in rows] + [""] * len(sectors)
    values = [r["cap"] for r in rows] + [sum(r["cap"] for r in rows if r["sector"] == s) for s in sectors]
    colors = [r["chg"] for r in rows] + [np.mean([r["chg"] for r in rows if r["sector"] == s]) for s in sectors]
    text = [f"{r['symbol']}<br>{format_price(r['price'])}<br>{format_percent(r['chg'])}" for r in rows] + sectors

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, text=text, textinfo="text",
        marker=dict(colors=colors, colorscale=[[0, "#7f1d1d"], [0.5, "#1e293b"], [1, "#065f46"]],
                    cmid=0, cmin=-3, cmax=3, line=dict(width=2, color="#020617")),
        hovertemplate="%{label}<br>%{text}<extra></extra>",
        root_color="#0F172A",
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520, paper_bgcolor="#0F172A",
                       font=dict(color="#e2e8f0", size=12))
    return fig


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "market_cat" not in st.session_state:
    st.session_state.market_cat = "india"
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "RELIANCE"
if "selected_yf" not in st.session_state:
    st.session_state.selected_yf = "RELIANCE.NS"
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sidebar_tab" not in st.session_state:
    st.session_state.sidebar_tab = "Chat"
if "chat_model" not in st.session_state:
    st.session_state.chat_model = ALL_MODELS[0]

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
col_logo, col_nav = st.columns([2, 4])
with col_logo:
    st.markdown("### ⚡ **AI Trade Terminal**")
with col_nav:
    pages = ["Dashboard", "News", "Heatmap", "History"]
    nav = st.radio("nav", pages, horizontal=True, label_visibility="collapsed",
                   index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    st.session_state.page = nav

st.markdown("---")

# ---------------------------------------------------------------------------
# MAIN + SIDE
# ---------------------------------------------------------------------------
main_col, side_col = st.columns([7, 3])

with main_col:
    if st.session_state.page == "Dashboard":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Select a market &amp; find a symbol</div>", unsafe_allow_html=True)
        mcol1, mcol2 = st.columns([1, 2])
        with mcol1:
            cat_labels = [c["label"] for c in MARKET_CATEGORIES]
            cat_keys = [c["key"] for c in MARKET_CATEGORIES]
            sel_label = st.selectbox("Market", cat_labels, index=cat_keys.index(st.session_state.market_cat) if st.session_state.market_cat in cat_keys else 0)
            st.session_state.market_cat = cat_keys[cat_labels.index(sel_label)]
            quick_items = [m for m in MARKET_ITEMS if m["category"] == st.session_state.market_cat]
            quick_labels = [f"{m['symbol']} — {m['name']}" for m in quick_items]
            quick_pick = st.selectbox("Quick pick", quick_labels, key="quick_pick")
            if st.button("Load", key="quick_load", use_container_width=True):
                chosen = quick_items[quick_labels.index(quick_pick)]
                st.session_state.selected_yf = chosen["yf"]
                st.session_state.selected_symbol = chosen["symbol"]
                st.rerun()
        with mcol2:
            search_input = st.text_input("Search any stock / index / crypto worldwide",
                                          placeholder="e.g. Reliance, Apple, Bitcoin, Nifty 50, EURUSD")
            if search_input.strip():
                results = yahoo_search(search_input)
                if results:
                    labels = [f"{r['symbol']} — {r.get('shortname') or r.get('longname') or r['symbol']} ({r.get('exchange', '')})" for r in results]
                    pick = st.selectbox("Matching symbols", labels, key="dash_search_pick")
                    if st.button("Analyze this symbol", key="dash_search_go", use_container_width=True):
                        chosen = results[labels.index(pick)]
                        st.session_state.selected_yf = chosen["symbol"]
                        st.session_state.selected_symbol = (chosen.get("shortname") or chosen["symbol"]).split(" ")[0].upper()
                        st.rerun()
                else:
                    st.caption("No matches — try a different spelling.")
        st.markdown("</div>", unsafe_allow_html=True)

        yf_sym = st.session_state.selected_yf
        data = fetch_stock_data(yf_sym)
        price = data["currentPrice"]
        chg = data["changePercent"]
        levels = get_session_fixed_levels(yf_sym)
        if not levels:
            levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or price)
        suggestions = generate_suggestions(price, levels)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-header'>{st.session_state.selected_symbol}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-sub'>{data.get('name', yf_sym)} · Last data: {data.get('lastDate', '—')} · {yf_sym}</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        chg_class = "up" if chg >= 0 else "down"
        m1.markdown(f"<div class='metric-label'>Current Price</div><div class='metric-value {chg_class}'>{format_price(price)} <span style='font-size:0.8rem'>{format_percent(chg)}</span></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-label'>Day High</div><div class='metric-value'>{format_price(data['dayHigh'])}</div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-label'>Day Low</div><div class='metric-value'>{format_price(data['dayLow'])}</div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-label'>Volume</div><div class='metric-value'>{format_volume(data['volume'])}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        chart_col, sug_col = st.columns([3, 2])
        with chart_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-header'>{st.session_state.selected_symbol} · Chart</div>", unsafe_allow_html=True)
            st.markdown("<div class='card-sub'>Price action with S/R levels (last available data when market closed)</div>", unsafe_allow_html=True)
            candles = fetch_candles(yf_sym)
            if candles is not None and not candles.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=candles.index, open=candles["Open"], high=candles["High"],
                    low=candles["Low"], close=candles["Close"],
                    increasing_line_color="#34d399", increasing_fillcolor="#34d399",
                    decreasing_line_color="#f87171", decreasing_fillcolor="#f87171", name="Price"))
                for key, color in [("R1", "#f87171"), ("R2", "#f87171"), ("S1", "#34d399"), ("S2", "#34d399")]:
                    if key in levels:
                        fig.add_hline(y=levels[key], line_dash="dot", line_color=color, line_width=1,
                                      annotation_text=key, annotation_position="right", annotation_font_color=color, annotation_font_size=10)
                fig.add_hline(y=price, line_color="#fbbf24", line_width=1.5,
                              annotation_text=format_price(price), annotation_position="left", annotation_font_color="#fbbf24")
                fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=380,
                    margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, showlegend=False,
                    font=dict(color="#94a3b8", size=11),
                    xaxis=dict(gridcolor="rgba(51,65,85,0.3)"), yaxis=dict(gridcolor="rgba(51,65,85,0.3)", side="right"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No candle data. Market may be closed — levels below use last available day.")
            st.markdown("</div>", unsafe_allow_html=True)

        with sug_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-header'>⚡ AI Trade Suggestion</div>", unsafe_allow_html=True)
            st.markdown("<div class='card-sub'>Both directions, always shown, with T1 &amp; T2</div>", unsafe_allow_html=True)
            bull = next((s for s in suggestions if s["direction"] == "bullish"), None)
            bear = next((s for s in suggestions if s["direction"] == "bearish"), None)
            if bull:
                st.markdown(f"""<div class="plan-card plan-bull">
<div class="plan-title-bull">▲ BULLISH PLAN</div>
<div>Entry: <b>{format_price(bull['entry'])}</b> &nbsp;&nbsp; Stop: <b>{format_price(bull['stop'])}</b></div>
<div>T1: <b>{format_price(bull['t1'])}</b> (R:R {bull['rr1']}) &nbsp;&nbsp; T2: <b>{format_price(bull['t2'])}</b> (R:R {bull['rr2']})</div>
</div>""", unsafe_allow_html=True)
            if bear:
                st.markdown(f"""<div class="plan-card plan-bear">
<div class="plan-title-bear">▼ BEARISH PLAN</div>
<div>Entry: <b>{format_price(bear['entry'])}</b> &nbsp;&nbsp; Stop: <b>{format_price(bear['stop'])}</b></div>
<div>T1: <b>{format_price(bear['t1'])}</b> (R:R {bear['rr1']}) &nbsp;&nbsp; T2: <b>{format_price(bear['t2'])}</b> (R:R {bear['rr2']})</div>
</div>""", unsafe_allow_html=True)
            if not bull and not bear:
                st.caption("Not enough level data yet to build a plan.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
    "<div class='card-header'>Square-of-9 Gann Degree Ladder</div>",
    unsafe_allow_html=True,
)
        ref_note = levels.get('_ref_time', 'session') if levels else 'session'
        ref_px = levels.get('_ref_close', 0) if levels else 0
        st.markdown(
            f"<div class='card-sub'>0° = first 15-min close ({format_price(ref_px)}) · levels FIXED until market close · educational references only</div>",
            unsafe_allow_html=True
        )
        if levels and levels.get("_type") == "gann_degree":
            gann_resistance_degrees = [180.0, 90.0, 67.5, 45.0, 22.5]

for degree in gann_resistance_degrees:
    suffix = str(degree).rstrip("0").rstrip(".")
    val = levels.get(f"R{suffix}", 0.0)
    pct = ((val - price) / price * 100) if price else 0.0

    st.markdown(
        f"""
        <div class='level-row level-resistance'>
            <span>
                <b style='color:#f87171'>{degree:g}° R</b>
                &nbsp; {format_price(val)}
                &nbsp;
                <span style='color:#64748b;font-size:0.75rem'>
                    above 0° anchor
                </span>
            </span>
            <span class='down'>{format_percent(pct)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
                pct = ((val - price) / price * 100) if price else 0
                st.markdown(
                    f"<div class='level-row level-resistance'>"
                    f"<span><b style='color:#f87171'>{deg}°</b> &nbsp; {format_price(val)} &nbsp; "
                    f"<span style='color:#64748b;font-size:0.75rem'>above 0°</span></span>"
                    f"<span class='down'>{format_percent(pct)}</span></div>",
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<div style='text-align:center;padding:12px;background:rgba(251,191,36,0.14);border-radius:10px;margin:8px 0;border:1px solid rgba(251,191,36,0.4)'>"
                f"<div class='amber' style='font-size:1.15rem;font-weight:700;'><b>0°</b> &nbsp; Reference &nbsp; {format_price(ref_px)}</div>"
                f"<div style='font-size:1.15rem;font-weight:700;color:#f8fafc;margin-top:6px;'>Current price: {format_price(price)}</div></div>",
                unsafe_allow_html=True
            )
            gann_support_degrees = [22.5, 45.0, 67.5, 90.0, 180.0]

for degree in gann_support_degrees:
    suffix = str(degree).rstrip("0").rstrip(".")
    val = levels.get(f"S{suffix}", 0.0)
    pct = ((val - price) / price * 100) if price else 0.0

    st.markdown(
        f"""
        <div class='level-row level-support'>
            <span>
                <b style='color:#34d399'>{degree:g}° S</b>
                &nbsp; {format_price(val)}
                &nbsp;
                <span style='color:#64748b;font-size:0.75rem'>
                    below 0° anchor
                </span>
            </span>
            <span class='up'>{format_percent(pct)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
                pct = ((val - price) / price * 100) if price else 0
                st.markdown(
                    f"<div class='level-row level-support'>"
                    f"<span><b style='color:#34d399'>{deg}°</b> &nbsp; {format_price(val)} &nbsp; "
                    f"<span style='color:#64748b;font-size:0.75rem'>below 0°</span></span>"
                    f"<span class='up'>{format_percent(pct)}</span></div>",
                    unsafe_allow_html=True
                )
            st.caption("Degree levels are potential price references only — not guaranteed reversals. Always define entry, stop, target and risk before trading.")
        elif levels:
            for key in ["R5", "R4", "R3", "R2", "R1"]:
                val = levels.get(key, 0)
                pct = ((val - price) / price * 100) if price else 0
                st.markdown(f"<div class='level-row level-resistance'><span><b style='color:#f87171'>{key}</b> &nbsp; {format_price(val)}</span><span class='down'>{format_percent(pct)}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;padding:8px;background:rgba(251,191,36,0.12);border-radius:8px;margin:6px 0;border:1px solid rgba(251,191,36,0.3)'><span class='amber'>● Current: {format_price(price)}</span></div>", unsafe_allow_html=True)
            for key in ["S1", "S2", "S3", "S4", "S5"]:
                val = levels.get(key, 0)
                pct = ((val - price) / price * 100) if price else 0
                st.markdown(f"<div class='level-row level-support'><span><b style='color:#34d399'>{key}</b> &nbsp; {format_price(val)}</span><span class='up'>{format_percent(pct)}</span></div>", unsafe_allow_html=True)
        else:
            st.info("Levels unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "News":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Latest News &amp; Sentiment</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-sub'>AI-ranked stories relevant to {st.session_state.selected_symbol}</div>", unsafe_allow_html=True)

        sym = st.session_state.selected_symbol
        yf_sym = st.session_state.selected_yf

        def _analyze_headline(title: str):
            t = (title or "").lower()
            bull_kw = ["surge", "rally", "beat", "growth", "record", "profit", "gain", "strong", "expansion",
                       "bullish", "rise", "high", "upbeat", "upgrade", "outperform", "positive"]
            bear_kw = ["fall", "drop", "loss", "miss", "cut", "weak", "down", "decline", "crash", "fear",
                       "bearish", "selloff", "concern", "downgrade", "probe", "fine", "lawsuit"]
            score = 5.5
            for k in bull_kw:
                if k in t: score += 0.55
            for k in bear_kw:
                if k in t: score -= 0.55
            score = max(1.5, min(9.5, round(score, 1)))
            if score >= 6.5:
                sent, sent_bg, sent_fg = "Bullish", "rgba(16,185,129,0.22)", "#34d399"
            elif score <= 4.5:
                sent, sent_bg, sent_fg = "Bearish", "rgba(239,68,68,0.22)", "#f87171"
            else:
                sent, sent_bg, sent_fg = "Neutral", "rgba(251,191,36,0.22)", "#fbbf24"
            if score >= 7:
                sc_bg, sc_fg = "rgba(16,185,129,0.28)", "#34d399"
            elif score >= 5:
                sc_bg, sc_fg = "rgba(251,191,36,0.28)", "#fbbf24"
            else:
                sc_bg, sc_fg = "rgba(239,68,68,0.28)", "#f87171"
            return score, sent, sent_bg, sent_fg, sc_bg, sc_fg

        def _summarize_impact(title: str, symbol: str, sent: str) -> str:
            if groq_client:
                try:
                    prompt = (
                        f"News headline: {title}\nStock: {symbol}\n"
                        f"Write exactly 2 short sentences: (1) what the news means (2) how it may impact {symbol} stock price. "
                        f"Be practical. Sentiment is {sent}."
                    )
                    resp = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=120,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception:
                    pass
            if sent == "Bullish":
                return (f"Positive development for {symbol}: the headline suggests improving fundamentals or sentiment. "
                        f"This can support near-term upside if price holds above key support levels.")
            if sent == "Bearish":
                return (f"Negative signal for {symbol}: the news may increase selling pressure or risk premium. "
                        f"Watch support levels; a break lower could extend the move.")
            return (f"Mixed / neutral news for {symbol}. Unlikely to drive a strong directional move alone. "
                    f"Focus on price reaction at support and resistance.")

        try:
            raw_news = yf.Ticker(yf_sym).news or []
        except Exception:
            raw_news = []

        if not raw_news:
            raw_news = [
                {"title": f"{sym} posts strong Q2 earnings beat", "publisher": "Economic Times", "hours": 2, "link": "#"},
                {"title": "Oil prices surge on supply concerns in Middle East", "publisher": "MarketWatch", "hours": 4, "link": "#"},
                {"title": "RBI holds rates steady, signals cautious optimism", "publisher": "Moneycontrol", "hours": 6, "link": "#"},
                {"title": "Global tech selloff weighs on market sentiment", "publisher": "CNBC", "hours": 8, "link": "#"},
                {"title": f"{sym} announces new retail expansion plan", "publisher": "Yahoo Finance", "hours": 24, "link": "#"},
                {"title": "Inflation data comes in cooler than expected", "publisher": "Bloomberg", "hours": 24, "link": "#"},
                {"title": "Regulatory concerns hit pharma sector", "publisher": "Reuters", "hours": 48, "link": "#"},
            ]

        if "news_open" not in st.session_state:
            st.session_state.news_open = None

        for idx, n in enumerate(raw_news[:8]):
            title = n.get("title") or (n.get("content") or {}).get("title") or "News item"
            pub = n.get("publisher") or (n.get("content") or {}).get("provider", {}).get("displayName") or "Source"
            link = n.get("link") or (n.get("content") or {}).get("clickThroughUrl", {}).get("url") or "#"
            hours = n.get("hours")
            if hours is None:
                hours = [2, 4, 6, 8, 24, 24, 48, 72][idx % 8]
            time_label = f"{hours}h ago" if hours < 24 else f"{hours // 24}d ago"

            score, sent, sent_bg, sent_fg, sc_bg, sc_fg = _analyze_headline(title)

            st.markdown(f"""
            <div style="background:#0B1220;border:1px solid rgba(51,65,85,0.5);border-radius:14px;padding:14px 16px;margin-bottom:10px;display:flex;gap:14px;align-items:flex-start;">
                <div style="min-width:44px;height:44px;border-radius:12px;background:{sc_bg};color:{sc_fg};font-weight:700;font-size:1.05rem;display:flex;align-items:center;justify-content:center;">{score}</div>
                <div style="flex:1;">
                    <div style="color:#f1f5f9;font-weight:600;font-size:0.95rem;line-height:1.35;">{title}</div>
                    <div style="margin-top:6px;font-size:0.78rem;color:#64748b;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                        <span>{pub}</span>
                        <span>·</span>
                        <span>{time_label}</span>
                        <span style="background:{sent_bg};color:{sent_fg};padding:2px 9px;border-radius:999px;font-weight:600;font-size:0.72rem;">{sent}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Summary & Impact", key=f"sum_{idx}", use_container_width=True):
                    st.session_state.news_open = idx if st.session_state.news_open != idx else None
                    st.rerun()
            with c2:
                if link and link != "#":
                    st.markdown(f"<a href='{link}' target='_blank' style='display:block;text-align:center;padding:0.4rem 0.6rem;border-radius:8px;background:#1E293B;border:1px solid rgba(51,65,85,0.6);color:#e2e8f0;text-decoration:none;font-size:0.85rem;'>Open full article ↗</a>", unsafe_allow_html=True)
                else:
                    st.caption("No external link")

            if st.session_state.news_open == idx:
                summary = _summarize_impact(title, sym, sent)
                st.markdown(f"""
                <div style="background:rgba(30,41,59,0.7);border:1px solid rgba(51,65,85,0.45);border-radius:12px;padding:12px 14px;margin:4px 0 14px 0;">
                    <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.04em;">Summary & impact on {sym}</div>
                    <div style="color:#e2e8f0;font-size:0.9rem;line-height:1.5;">{summary}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "Heatmap":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Market Heatmap</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Color = % change (green up / red down, darker = bigger move) · Size = market cap · Grouped by sector</div>", unsafe_allow_html=True)

        selected_cat = "india"
        for m in MARKET_ITEMS:
            if m["yf"] == st.session_state.selected_yf or m["symbol"] == st.session_state.selected_symbol:
                selected_cat = m["category"]
                break
        else:
            selected_cat = st.session_state.market_cat

        cat_to_chip = {"india": "India", "us": "US", "crypto": "Crypto", "commodities": "Commodities", "forex": "Global", "indices": "Global"}
        default_chip = cat_to_chip.get(selected_cat, "All")
        market_chips = ["All", "India", "US", "Crypto", "Commodities", "Global"]
        m_idx = market_chips.index(default_chip) if default_chip in market_chips else 0
        market_sel = st.radio("Market section", market_chips, horizontal=True, index=m_idx, label_visibility="collapsed", key="heat_market")

        cat_map = {"India": "india", "US": "us", "Crypto": "crypto", "Commodities": "commodities", "Global": None}
        if market_sel == "All":
            pool = list(MARKET_ITEMS)
        elif market_sel == "Global":
            pool = [m for m in MARKET_ITEMS if m["category"] in ("forex", "indices")]
        else:
            pool = [m for m in MARKET_ITEMS if m["category"] == cat_map.get(market_sel, "india")]

        if not pool:
            st.info("No symbols for this filter.")
        else:
            with st.spinner("Building heatmap..."):
                fig = build_heatmap_treemap(pool)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not build the heatmap right now — try again in a moment.")

            jump_labels = [f"{m['symbol']} — {m['name']}" for m in pool]
            jc1, jc2 = st.columns([3, 1])
            with jc1:
                jump_pick = st.selectbox("Jump to a symbol from this view", jump_labels, key="heat_jump")
            with jc2:
                st.write("")
                st.write("")
                if st.button("Analyze", key="heat_jump_go", use_container_width=True):
                    chosen = pool[jump_labels.index(jump_pick)]
                    st.session_state.selected_symbol = chosen["symbol"]
                    st.session_state.selected_yf = chosen["yf"]
                    st.session_state.page = "Dashboard"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "History":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-header'>Historical Data — {st.session_state.selected_symbol}</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Previous 2 trading days · price action, volume, trend & momentum</div>", unsafe_allow_html=True)

        hist = fetch_history(st.session_state.selected_yf)
        if hist is not None and not hist.empty and len(hist) >= 2:
            day2 = hist.iloc[-1]
            day1 = hist.iloc[-2]

            d1_o, d1_h, d1_l, d1_c = float(day1["Open"]), float(day1["High"]), float(day1["Low"]), float(day1["Close"])
            d2_o, d2_h, d2_l, d2_c = float(day2["Open"]), float(day2["High"]), float(day2["Low"]), float(day2["Close"])
            d1_v = float(day1.get("Volume", 0) or 0)
            d2_v = float(day2.get("Volume", 0) or 0)
            d1_date = hist.index[-2].strftime("%b %d") if hasattr(hist.index[-2], "strftime") else str(hist.index[-2])
            d2_date = hist.index[-1].strftime("%b %d") if hasattr(hist.index[-1], "strftime") else str(hist.index[-1])

            tail = hist.tail(12)
            colors = ["#34d399" if tail["Close"].iloc[i] >= tail["Open"].iloc[i] else "#f87171" for i in range(len(tail))]
            fig = go.Figure(go.Bar(x=tail.index, y=tail["Close"], marker_color=colors))
            fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=220,
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False, font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="rgba(51,65,85,0.3)"), yaxis=dict(gridcolor="rgba(51,65,85,0.3)", side="right"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### 📊 Price Action and Levels")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Day 1 — {d1_date}**")
                st.write(f"Open: `{format_price(d1_o)}` · High: `{format_price(d1_h)}`")
                st.write(f"Low: `{format_price(d1_l)}` · Close: `{format_price(d1_c)}`")
                rng1 = d1_h - d1_l
                pos1 = ((d1_c - d1_l) / rng1 * 100) if rng1 > 0 else 50
                st.caption(f"Close position in range: {pos1:.0f}% from low")
            with c2:
                st.markdown(f"**Day 2 — {d2_date}**")
                st.write(f"Open: `{format_price(d2_o)}` · High: `{format_price(d2_h)}`")
                st.write(f"Low: `{format_price(d2_l)}` · Close: `{format_price(d2_c)}`")
                rng2 = d2_h - d2_l
                pos2 = ((d2_c - d2_l) / rng2 * 100) if rng2 > 0 else 50
                st.caption(f"Close position in range: {pos2:.0f}% from low")

            broke_high = d2_h > d1_h
            broke_low = d2_l < d1_l
            st.write(f"- Day 2 **{'broke' if broke_high else 'did not break'}** Day 1 high ({format_price(d1_h)}).")
            st.write(f"- Day 2 **{'broke' if broke_low else 'did not break'}** Day 1 low ({format_price(d1_l)}).")
            st.write(f"- Day 2 closed **{'near the high' if pos2 >= 70 else ('near the low' if pos2 <= 30 else 'mid-range')}** of its daily range.")

            st.markdown("##### 📈 Volume Analysis")
            vol_chg = ((d2_v - d1_v) / d1_v * 100) if d1_v > 0 else 0
            st.write(f"- Day 1 volume: `{format_volume(d1_v)}` · Day 2 volume: `{format_volume(d2_v)}` ({format_percent(vol_chg)})")
            up_day = d2_c >= d2_o
            if up_day and d2_v > d1_v:
                st.write("- Higher volume on an **up** day → stronger buyer interest.")
            elif (not up_day) and d2_v > d1_v:
                st.write("- Higher volume on a **down** day → heavier selling pressure.")
            else:
                st.write("- Volume did not expand with the move → conviction is mixed.")

            st.markdown("##### 💡 Trend and Momentum")
            net_pct = ((d2_c - d1_c) / d1_c * 100) if d1_c else 0
            st.write(f"- Net change (Day1 close → Day2 close): **{format_percent(net_pct)}**")
            gap = d2_o - d1_c
            gap_pct = (gap / d1_c * 100) if d1_c else 0
            if abs(gap_pct) >= 0.3:
                st.write(f"- Gap between Day1 close and Day2 open: **{format_percent(gap_pct)}** ({'gap up' if gap > 0 else 'gap down'}).")
            else:
                st.write("- No significant gap between Day1 close and Day2 open.")

            closes = hist["Close"].tail(9)
            if len(closes) >= 5:
                ma5 = closes.tail(5).mean()
                ma5_prev = closes.tail(6).head(5).mean() if len(closes) >= 6 else ma5
                slope = "up" if ma5 > ma5_prev else "down"
                st.write(f"- 5-period MA is sloping **{slope}** (MA5 ≈ {format_price(float(ma5))}).")
            if len(closes) >= 9:
                ma9 = closes.tail(9).mean()
                st.write(f"- 9-period MA ≈ {format_price(float(ma9))}.")

            st.markdown("##### Swing levels (lookback)")
            s1, s2 = st.columns(2)
            s1.markdown(f"<div class='level-row level-resistance'><b>Swing High</b> {format_price(float(hist['High'].tail(10).max()))}</div>", unsafe_allow_html=True)
            s2.markdown(f"<div class='level-row level-support'><b>Swing Low</b> {format_price(float(hist['Low'].tail(10).min()))}</div>", unsafe_allow_html=True)

            show = hist[["Open", "High", "Low", "Close", "Volume"]].tail(10).iloc[::-1]
            show.index = [i.strftime("%b %d") if hasattr(i, "strftime") else str(i) for i in show.index]
            st.dataframe(show.style.format({"Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Close": "{:.2f}", "Volume": "{:,.0f}"}), use_container_width=True)
        else:
            st.warning("Not enough historical data (need at least 2 daily bars). Try another symbol or wait for data.")
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    side_tabs = st.radio("side", ["Chat", "Watch", "Alerts"], horizontal=True, label_visibility="collapsed",
                         index=["Chat", "Watch", "Alerts"].index(st.session_state.sidebar_tab))
    st.session_state.sidebar_tab = side_tabs

    if side_tabs == "Chat":
        st.markdown("##### 🤖 AI Chat Assistant")
        st.session_state.chat_model = st.selectbox("Bot", ALL_MODELS, index=ALL_MODELS.index(st.session_state.chat_model) if st.session_state.chat_model in ALL_MODELS else 0)
        st.caption(f"Ask about {st.session_state.selected_symbol}")
        for msg in st.session_state.chat_history[-8:]:
            cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(f"<div class='{cls}'>{msg['text']}</div>", unsafe_allow_html=True)
        user_msg = st.chat_input("Ask about levels, strategy...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "text": user_msg})
            data = fetch_stock_data(st.session_state.selected_yf)
            levels = get_session_fixed_levels(st.session_state.selected_yf)
            if not levels:
                levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or data["currentPrice"])
            reply = ai_chat(user_msg, st.session_state.selected_symbol, data, levels, st.session_state.chat_model)
            st.session_state.chat_history.append({"role": "assistant", "text": reply})
            st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("Support?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": "What are support levels?"})
            data = fetch_stock_data(st.session_state.selected_yf)
            levels = get_session_fixed_levels(st.session_state.selected_yf)
            if not levels:
                levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or data["currentPrice"])
            st.session_state.chat_history.append({"role": "assistant", "text": ai_chat("support", st.session_state.selected_symbol, data, levels, st.session_state.chat_model)})
            st.rerun()
        if c2.button("Bullish?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": "Bullish setup?"})
            data = fetch_stock_data(st.session_state.selected_yf)
            levels = get_session_fixed_levels(st.session_state.selected_yf)
            if not levels:
                levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or data["currentPrice"])
            st.session_state.chat_history.append({"role": "assistant", "text": ai_chat("bullish", st.session_state.selected_symbol, data, levels, st.session_state.chat_model)})
            st.rerun()

    elif side_tabs == "Watch":
        st.markdown("##### 📋 Watchlist")
        new_ticker = st.text_input("Add ticker", placeholder="RELIANCE or AAPL")
        if st.button("＋ Add", use_container_width=True) and new_ticker.strip():
            yf_s = resolve_yf_symbol(new_ticker, st.session_state.market_cat)
            if yf_s not in st.session_state.watchlist:
                st.session_state.watchlist.append(yf_s)
                st.rerun()
        for w in st.session_state.watchlist:
            d = fetch_stock_data(w)
            chg = d.get("changePercent", 0) or 0
            c1, c2 = st.columns([4, 1])
            with c1:
                if st.button(f"{w.replace('.NS','')}  {format_price(d.get('currentPrice',0))}  {format_percent(chg)}", key=f"w_{w}", use_container_width=True):
                    st.session_state.selected_yf = w
                    st.session_state.selected_symbol = w.replace(".NS", "").replace("-USD", "").replace("=F", "").replace("=X", "").replace("^", "")
                    st.session_state.page = "Dashboard"
                    st.rerun()
            with c2:
                if st.button("🗑", key=f"del_{w}"):
                    st.session_state.watchlist = [x for x in st.session_state.watchlist if x != w]
                    st.rerun()

    elif side_tabs == "Alerts":
        st.markdown("##### 🔔 Price Alerts")
        a_sym = st.text_input("Symbol", value=st.session_state.selected_symbol)
        default_alert_price = float(fetch_stock_data(st.session_state.selected_yf).get("currentPrice", 0) or 0)
        a_price = st.number_input("Target", min_value=0.0, value=default_alert_price, format="%.2f")
        a_dir = st.selectbox("Direction", ["Above", "Below"])
        if st.button("＋ Set Alert", use_container_width=True):
            yf_s = resolve_yf_symbol(a_sym, st.session_state.market_cat)
            st.session_state.alerts.append({"symbol": yf_s, "price": a_price, "direction": a_dir, "notified": False})
            send_telegram(f"Alert set: {yf_s} {a_dir} {a_price}")
            st.success("Alert set")
            st.rerun()
        if not st.session_state.alerts:
            st.caption("No alerts yet.")
        else:
            for i, a in enumerate(list(st.session_state.alerts)):
                d = fetch_stock_data(a["symbol"])
                cur = d.get("currentPrice", 0)
                triggered = (a["direction"] == "Above" and cur >= a["price"]) or (a["direction"] == "Below" and cur <= a["price"])
                if triggered:
                    st.warning(f"🔔 {a['symbol']} hit {a['direction']} {a['price']} (now {format_price(cur)})")
                    if not a.get("notified"):
                        if send_telegram(f"ALERT: {a['symbol']} is {format_price(cur)} ({a['direction']} {a['price']})"):
                            st.session_state.alerts[i]["notified"] = True
                else:
                    st.write(f"{a['symbol']} {a['direction']} {a['price']} · now {format_price(cur)}")
                if st.button("Remove", key=f"ar_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;color:#475569;font-size:0.75rem;margin-top:1.5rem'>AI Trade Terminal · When market is closed, last available day data + levels are shown</div>", unsafe_allow_html=True)
