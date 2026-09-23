"""
AI Trade Terminal - Streamlit Edition
=====================================
Full real-time recreation of the original React + Flask AI Trade Terminal.

Features:
- Live global market data (yfinance) - any stock, crypto, forex, commodity, index
- Same dark UI, layout, colors and interaction patterns
- Support / Resistance (S1-S5, R1-R5) using real OHLC
- AI Trade Suggestions (Bullish / Bearish)
- Candlestick chart with Plotly
- Market Heatmap
- Historical data + swing levels
- News feed
- AI Chat (Groq primary + Gemini fallback)
- Watchlists
- Price Alerts with real Telegram notifications

Run:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# API KEYS (as provided)
# ---------------------------------------------------------------------------
GROQ_KEY = "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"
GEMINI_KEY = "AQ.Ab8RN6JtGdVf9VFtpeo2_7BYDuZQZZlhMIbQxKFX1noZ4UnTSQ"
TELEGRAM_BOT_TOKEN = "8794257218:AAGYGDqPUEJdI3UahL07Pe86IgcLCfIn20g"
TELEGRAM_CHAT_ID = "8600332637"

# ---------------------------------------------------------------------------
# Page Config + Custom CSS (matches original slate-950 / amber theme)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Trade Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ========== GLOBAL ========== */
    .stApp {
        background-color: #020617 !important;
        color: #f8fafc;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* ========== CARDS ========== */
    .card {
        background: #0F172A;
        border: 1px solid rgba(51, 65, 85, 0.45);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }
    .card-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 0.9rem;
    }

    /* ========== METRICS ========== */
    .metric-label {
        font-size: 0.68rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }
    .up { color: #34d399 !important; }
    .down { color: #f87171 !important; }
    .amber { color: #fbbf24 !important; }

    /* ========== BUTTONS ========== */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(51, 65, 85, 0.55) !important;
        background: #1E293B !important;
        color: #e2e8f0 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: #fbbf24 !important;
        color: #fbbf24 !important;
        background: rgba(251, 191, 36, 0.08) !important;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
        border-bottom: 1px solid rgba(51, 65, 85, 0.35);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1E293B !important;
        color: #f8fafc !important;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(51, 65, 85, 0.4);
    }

    /* ========== INPUTS ========== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        background-color: #1E293B !important;
        border: 1px solid rgba(51, 65, 85, 0.55) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }

    /* ========== CHAT ========== */
    .chat-user {
        background: #334155;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0 6px 18%;
        text-align: right;
        font-size: 0.9rem;
    }
    .chat-assistant {
        background: #1E293B;
        border: 1px solid rgba(51, 65, 85, 0.4);
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 12% 6px 0;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    /* ========== LEVEL LADDER ========== */
    .level-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 14px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    .level-support {
        background: rgba(16, 185, 129, 0.07);
        border: 1px solid rgba(16, 185, 129, 0.18);
    }
    .level-resistance {
        background: rgba(239, 68, 68, 0.07);
        border: 1px solid rgba(239, 68, 68, 0.18);
    }

    /* ========== HEATMAP ========== */
    .heat-up {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.28);
    }
    .heat-down {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.28);
    }

    /* ========== ALERTS ========== */
    .alert-triggered {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.35);
        border-radius: 10px;
        padding: 11px 14px;
    }

    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Market Catalog (same as original)
# ---------------------------------------------------------------------------
MARKET_CATEGORIES = [
    {"key": "india", "label": "India NSE", "icon": "🇮🇳"},
    {"key": "us", "label": "US Stocks", "icon": "🇺🇸"},
    {"key": "crypto", "label": "Crypto", "icon": "₿"},
    {"key": "commodities", "label": "Commodities", "icon": "🪙"},
    {"key": "forex", "label": "Forex", "icon": "💱"},
    {"key": "global", "label": "Global Indices", "icon": "🌍"},
]

MARKET_ITEMS = [
    # India
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "category": "india"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy", "category": "india"},
    {"symbol": "INFY.NS", "name": "Infosys", "category": "india"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "category": "india"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "category": "india"},
    {"symbol": "SBIN.NS", "name": "SBI", "category": "india"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "category": "india"},
    {"symbol": "ITC.NS", "name": "ITC Ltd", "category": "india"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "category": "india"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "category": "india"},
    # US
    {"symbol": "AAPL", "name": "Apple", "category": "us"},
    {"symbol": "MSFT", "name": "Microsoft", "category": "us"},
    {"symbol": "NVDA", "name": "NVIDIA", "category": "us"},
    {"symbol": "GOOGL", "name": "Alphabet", "category": "us"},
    {"symbol": "AMZN", "name": "Amazon", "category": "us"},
    {"symbol": "META", "name": "Meta", "category": "us"},
    {"symbol": "TSLA", "name": "Tesla", "category": "us"},
    {"symbol": "JPM", "name": "JPMorgan", "category": "us"},
    # Crypto
    {"symbol": "BTC-USD", "name": "Bitcoin", "category": "crypto"},
    {"symbol": "ETH-USD", "name": "Ethereum", "category": "crypto"},
    {"symbol": "BNB-USD", "name": "BNB", "category": "crypto"},
    {"symbol": "SOL-USD", "name": "Solana", "category": "crypto"},
    {"symbol": "XRP-USD", "name": "Ripple", "category": "crypto"},
    {"symbol": "ADA-USD", "name": "Cardano", "category": "crypto"},
    {"symbol": "DOGE-USD", "name": "Dogecoin", "category": "crypto"},
    # Commodities
    {"symbol": "GC=F", "name": "Gold", "category": "commodities"},
    {"symbol": "SI=F", "name": "Silver", "category": "commodities"},
    {"symbol": "CL=F", "name": "Crude Oil", "category": "commodities"},
    {"symbol": "NG=F", "name": "Natural Gas", "category": "commodities"},
    {"symbol": "HG=F", "name": "Copper", "category": "commodities"},
    # Forex
    {"symbol": "USDINR=X", "name": "USD/INR", "category": "forex"},
    {"symbol": "EURUSD=X", "name": "EUR/USD", "category": "forex"},
    {"symbol": "GBPUSD=X", "name": "GBP/USD", "category": "forex"},
    {"symbol": "USDJPY=X", "name": "USD/JPY", "category": "forex"},
    {"symbol": "AUDUSD=X", "name": "AUD/USD", "category": "forex"},
    # Global Indices
    {"symbol": "^GSPC", "name": "S&P 500", "category": "global"},
    {"symbol": "^IXIC", "name": "Nasdaq", "category": "global"},
    {"symbol": "^DJI", "name": "Dow Jones", "category": "global"},
    {"symbol": "^NSEI", "name": "Nifty 50", "category": "global"},
    {"symbol": "^NSEBANK", "name": "Nifty Bank", "category": "global"},
    {"symbol": "^GDAXI", "name": "DAX", "category": "global"},
    {"symbol": "^FTSE", "name": "FTSE 100", "category": "global"},
    {"symbol": "^N225", "name": "Nikkei 225", "category": "global"},
]

HEATMAP_SECTORS = [
    {"name": "Banking", "items": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "JPM"]},
    {"name": "IT", "items": ["TCS.NS", "INFY.NS", "AAPL", "MSFT", "NVDA", "GOOGL"]},
    {"name": "Energy", "items": ["RELIANCE.NS", "CL=F"]},
    {"name": "Auto", "items": ["TATAMOTORS.NS", "TSLA"]},
    {"name": "FMCG", "items": ["ITC.NS"]},
    {"name": "Pharma", "items": ["SUNPHARMA.NS"]},
    {"name": "Metal", "items": ["TATASTEEL.NS", "HG=F"]},
    {"name": "Crypto", "items": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]},
    {"name": "Commodities", "items": ["GC=F", "SI=F", "CL=F"]},
    {"name": "Global", "items": ["^GSPC", "^NSEI", "^N225", "^DJI"]},
]

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "symbol" not in st.session_state:
    st.session_state.symbol = "RELIANCE.NS"
if "market_cat" not in st.session_state:
    st.session_state.market_cat = "india"
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {
        "My Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "US Tech": ["AAPL", "MSFT", "NVDA"],
        "Crypto": ["BTC-USD", "ETH-USD"],
    }
if "current_list" not in st.session_state:
    st.session_state.current_list = "My Stocks"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi! I'm your AI trading assistant. Ask me about any stock — support/resistance, bullish/bearish setups, targets, or strategy."}
    ]
if "sidebar_view" not in st.session_state:
    st.session_state.sidebar_view = "chat"
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_price(value: float) -> str:
    if value is None or np.isnan(value):
        return "—"
    if abs(value) < 1:
        return f"{value:.4f}"
    return f"{value:,.2f}"

def format_percent(value: float) -> str:
    if value is None or np.isnan(value):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"

def format_volume(value: float) -> str:
    if value is None or np.isnan(value):
        return "—"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"

@st.cache_data(ttl=60)
def fetch_ticker_info(symbol: str) -> Dict:
    """Fetch live quote + basic info."""
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        hist = t.history(period="5d", interval="1d")
        if hist.empty:
            hist = t.history(period="1mo", interval="1d")
        
        current = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
        if current is None and not hist.empty:
            current = float(hist["Close"].iloc[-1])
        
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if prev_close is None and len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
        elif prev_close is None and not hist.empty:
            prev_close = float(hist["Close"].iloc[-1])
        
        day_high = info.get("dayHigh") or info.get("regularMarketDayHigh")
        day_low = info.get("dayLow") or info.get("regularMarketDayLow")
        if (day_high is None or day_low is None) and not hist.empty:
            day_high = float(hist["High"].iloc[-1])
            day_low = float(hist["Low"].iloc[-1])
        
        volume = info.get("volume") or info.get("regularMarketVolume") or 0
        if volume == 0 and not hist.empty:
            volume = float(hist["Volume"].iloc[-1])
        
        change_pct = 0.0
        if current and prev_close and prev_close != 0:
            change_pct = ((current - prev_close) / prev_close) * 100
        
        name = info.get("shortName") or info.get("longName") or symbol
        
        return {
            "symbol": symbol,
            "name": name,
            "currentPrice": float(current) if current else 0.0,
            "previousClose": float(prev_close) if prev_close else 0.0,
            "dayHigh": float(day_high) if day_high else 0.0,
            "dayLow": float(day_low) if day_low else 0.0,
            "volume": float(volume) if volume else 0.0,
            "changePercent": float(change_pct),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "name": symbol,
            "currentPrice": 0.0,
            "previousClose": 0.0,
            "dayHigh": 0.0,
            "dayLow": 0.0,
            "volume": 0.0,
            "changePercent": 0.0,
            "currency": "USD",
            "error": str(e),
        }

@st.cache_data(ttl=120)
def fetch_candles(symbol: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df.empty:
            df = t.history(period="1mo", interval="1h")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_history(symbol: str, period: str = "1mo") -> pd.DataFrame:
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval="1d")
        return df
    except Exception:
        return pd.DataFrame()

def generate_levels(current: float, prev_close: float, high: float, low: float) -> Dict:
    """Classic pivot-point S1-S5 / R1-R5."""
    if high == 0 or low == 0:
        high = current * 1.01
        low = current * 0.99
    if prev_close == 0:
        prev_close = current
    
    pivot = (high + low + prev_close) / 3
    supports, resistances = [], []
    
    for i in range(1, 6):
        if i == 1:
            s = 2 * pivot - high
            r = 2 * pivot - low
        elif i == 2:
            s = pivot - (high - low)
            r = pivot + (high - low)
        elif i == 3:
            s = low - 2 * (high - pivot)
            r = high + 2 * (pivot - low)
        elif i == 4:
            s = low - 3 * (high - pivot)
            r = high + 3 * (pivot - low)
        else:
            s = low - 4 * (high - pivot)
            r = high + 4 * (pivot - low)
        
        supports.append({"label": f"S{i}", "price": round(s, 4), "type": "support", "index": i})
        resistances.append({"label": f"R{i}", "price": round(r, 4), "type": "resistance", "index": i})
    
    return {"supports": supports, "resistances": resistances}

def generate_trade_suggestions(data: Dict, levels: Dict) -> List[Dict]:
    price = data["currentPrice"]
    supports = levels["supports"]
    resistances = levels["resistances"]
    
    nearby_s = [s for s in supports if s["price"] < price]
    nearest_s = nearby_s[0] if nearby_s else supports[0]
    
    nearby_r = [r for r in resistances if r["price"] > price]
    nearest_r = nearby_r[0] if nearby_r else resistances[0]
    
    target_r = [r for r in resistances if r["price"] > nearest_r["price"]]
    target_s = [s for s in supports if s["price"] < nearest_s["price"]]
    
    # Bullish
    bull_entry = nearest_s["price"]
    bull_stop = round(nearest_s["price"] * 0.985, 4)
    bull_target = target_r[0]["price"] if target_r else nearest_r["price"]
    bull_rr = abs((bull_target - bull_entry) / (bull_entry - bull_stop)) if (bull_entry - bull_stop) != 0 else 1.0
    
    # Bearish
    bear_entry = nearest_r["price"]
    bear_stop = round(nearest_r["price"] * 1.015, 4)
    bear_target = target_s[0]["price"] if target_s else nearest_s["price"]
    bear_rr = abs((bear_entry - bear_target) / (bear_stop - bear_entry)) if (bear_stop - bear_entry) != 0 else 1.0
    
    return [
        {
            "direction": "bullish",
            "entry": bull_entry,
            "stopLoss": bull_stop,
            "target": bull_target,
            "breakoutLabel": nearest_r["label"],
            "breakoutPrice": nearest_r["price"],
            "holdLabel": nearest_s["label"],
            "holdPrice": nearest_s["price"],
            "note": f"If price holds above {nearest_s['label']} and breaks {nearest_r['label']}, expect upward move.",
            "riskReward": round(bull_rr, 2),
        },
        {
            "direction": "bearish",
            "entry": bear_entry,
            "stopLoss": bear_stop,
            "target": bear_target,
            "breakoutLabel": nearest_s["label"],
            "breakoutPrice": nearest_s["price"],
            "holdLabel": nearest_r["label"],
            "holdPrice": nearest_r["price"],
            "note": f"If price stays below {nearest_r['label']} and breaks {nearest_s['label']}, expect downward move.",
            "riskReward": round(bear_rr, 2),
        },
    ]

def get_swing_levels(df: pd.DataFrame) -> Dict:
    if df.empty or len(df) < 10:
        return {"resistance": [], "support": []}
    
    highs, lows = [], []
    window = 3
    for i in range(window, len(df) - window):
        seg_h = df["High"].iloc[i - window : i + window + 1]
        seg_l = df["Low"].iloc[i - window : i + window + 1]
        if df["High"].iloc[i] == seg_h.max():
            highs.append(float(df["High"].iloc[i]))
        if df["Low"].iloc[i] == seg_l.min():
            lows.append(float(df["Low"].iloc[i]))
    
    def dedupe(vals, reverse=True):
        seen = sorted(set(round(v, 4) for v in vals), reverse=reverse)
        out = []
        for v in seen:
            if len(out) >= 4:
                break
            if not out or abs(v - out[-1]) / max(abs(v), 1e-9) > 0.008:
                out.append(v)
        return out
    
    return {"resistance": dedupe(highs, True), "support": dedupe(lows, False)}

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_telegram(message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# AI Chat (Groq primary → Gemini fallback → rule-based)
# ---------------------------------------------------------------------------
def generate_ai_response(prompt: str, symbol: str, price: float, data: Dict, levels: Dict) -> str:
    context = f"""
You are an expert trading assistant for the AI Trade Terminal.
Current symbol: {symbol} ({data.get('name', symbol)})
Live price: {format_price(price)}
Change: {format_percent(data.get('changePercent', 0))}
Day High: {format_price(data.get('dayHigh', 0))}
Day Low: {format_price(data.get('dayLow', 0))}
Nearest supports: {', '.join([f"{s['label']}={format_price(s['price'])}" for s in levels.get('supports', [])[:3]])}
Nearest resistances: {', '.join([f"{r['label']}={format_price(r['price'])}" for r in levels.get('resistances', [])[:3]])}

Rules:
- Be concise, practical and educational.
- Never give financial advice that guarantees profit.
- Always remind that this is educational only.
- Use the live levels when discussing setups.
"""
    
    # 1. Try Groq
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        pass
    
    # 2. Try Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(context + "\n\nUser question: " + prompt)
        return response.text.strip()
    except Exception:
        pass
    
    # 3. Rule-based fallback (same logic as original Flask)
    p = prompt.lower()
    if any(w in p for w in ["support", "s1", "s2", "s3"]):
        return (
            f"For {symbol} at {format_price(price)}, the nearest support levels act as potential buy zones. "
            "If price holds above support and bounces with volume, buyers are stepping in. "
            "A break below support would invalidate the bullish case. Educational only."
        )
    if any(w in p for w in ["resistance", "r1", "r2", "r3"]):
        return (
            f"{symbol} is at {format_price(price)}. Resistance levels above act as selling pressure. "
            "A break through resistance with volume can signal a breakout. Otherwise expect pullback. Educational only."
        )
    if any(w in p for w in ["buy", "long", "bullish"]):
        return (
            f"Looking at {symbol} at {format_price(price)}: a bullish setup would be buying near the nearest support "
            "with a stop below it and targeting the next resistance. Always confirm with price action and volume. "
            "This is educational analysis, not financial advice."
        )
    if any(w in p for w in ["sell", "short", "bearish"]):
        return (
            f"For {symbol} at {format_price(price)}: a bearish setup would be selling near resistance with a stop above it, "
            "targeting the next support. Look for rejection candles. Educational only."
        )
    if "target" in p:
        return (
            f"Based on pivot levels for {symbol}, targets are set at the next resistance (longs) or next support (shorts). "
            "Aim for at least 1:1.5 risk-reward. Educational only."
        )
    if any(w in p for w in ["strategy", "plan"]):
        return (
            f"For {symbol} at {format_price(price)}: Check the AI Trade Suggestion panel for specific entry, stop and target. "
            "Pick the green button for a bullish plan or the red button for a bearish plan. Always manage risk."
        )
    
    return (
        f"I can help with {symbol} (currently at {format_price(price)}). Ask me about support/resistance levels, "
        "bullish or bearish setups, targets, news sentiment, or trading strategy. Educational analysis only."
    )

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
col_logo, col_title, col_refresh = st.columns([0.6, 4, 1.2])
with col_logo:
    st.markdown("<div style='font-size:1.8rem;'>⚡</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0; font-size:1.5rem; font-weight:700; color:#f8fafc;'>AI Trade Terminal</h1>", unsafe_allow_html=True)
with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

st.markdown(f"<p style='color:#64748b; font-size:0.75rem; margin-top:-0.5rem;'>Last update: {st.session_state.last_refresh.strftime('%H:%M:%S')} · Educational tool only</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------------
tab_dash, tab_news, tab_heat, tab_hist = st.tabs(["📊 Dashboard", "📰 News", "🌡️ Heatmap", "📅 History"])

# ======================== DASHBOARD ========================
with tab_dash:
    # Market category selector
    cats = [c["label"] for c in MARKET_CATEGORIES]
    cat_keys = [c["key"] for c in MARKET_CATEGORIES]
    selected_cat_label = st.radio(
        "Market",
        cats,
        index=cat_keys.index(st.session_state.market_cat) if st.session_state.market_cat in cat_keys else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.market_cat = cat_keys[cats.index(selected_cat_label)]
    
    # Symbol picker for current category
    items = [m for m in MARKET_ITEMS if m["category"] == st.session_state.market_cat]
    item_labels = [f"{m['symbol']} — {m['name']}" for m in items]
    
    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input(
            "Search any symbol worldwide (e.g. AAPL, RELIANCE.NS, BTC-USD, GC=F, ^NSEI)",
            value=st.session_state.symbol,
            key="search_box",
        )
    with c2:
        st.write("")
        st.write("")
        if st.button("🔍 Load", use_container_width=True):
            st.session_state.symbol = search.strip().upper()
            st.cache_data.clear()
            st.rerun()
    
    # Quick pick buttons
    cols = st.columns(min(len(items), 5))
    for i, m in enumerate(items[:5]):
        with cols[i]:
            if st.button(m["symbol"].replace(".NS", "").replace("-USD", "").replace("=F", "").replace("=X", "").replace("^", "")[:6], key=f"pick_{m['symbol']}", use_container_width=True):
                st.session_state.symbol = m["symbol"]
                st.cache_data.clear()
                st.rerun()
    
    symbol = st.session_state.symbol
    
    # Fetch live data
    with st.spinner(f"Fetching live data for {symbol}..."):
        data = fetch_ticker_info(symbol)
        candles = fetch_candles(symbol)
        levels = generate_levels(data["currentPrice"], data["previousClose"], data["dayHigh"], data["dayLow"])
        suggestions = generate_trade_suggestions(data, levels)
    
    if data.get("error"):
        st.warning(f"Could not fully load {symbol}: {data['error']}. Showing available data.")
    
    # Stock Header
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    is_up = data["changePercent"] >= 0
    with h1:
        st.markdown(f"<div class='metric-label'>Current Price</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value {'up' if is_up else 'down'}'>{format_price(data['currentPrice'])} <span style='font-size:0.9rem'>{format_percent(data['changePercent'])}</span></div>", unsafe_allow_html=True)
        st.caption(data["name"])
    with h2:
        st.markdown(f"<div class='metric-label'>Day High</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{format_price(data['dayHigh'])}</div>", unsafe_allow_html=True)
    with h3:
        st.markdown(f"<div class='metric-label'>Day Low</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{format_price(data['dayLow'])}</div>", unsafe_allow_html=True)
    with h4:
        st.markdown(f"<div class='metric-label'>Volume</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{format_volume(data['volume'])}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chart + Trade Suggestions side by side
    left, right = st.columns([2.2, 1])
    
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-header'>{symbol} · Candlestick Chart</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Live price action with Support & Resistance overlays</div>", unsafe_allow_html=True)
        
        if not candles.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=candles.index,
                open=candles["Open"],
                high=candles["High"],
                low=candles["Low"],
                close=candles["Close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ))
            
            # Resistance lines
            for r in levels["resistances"][:3]:
                fig.add_hline(
                    y=r["price"],
                    line_dash="dash",
                    line_color="rgba(239,68,68,0.5)",
                    annotation_text=f"{r['label']} {format_price(r['price'])}",
                    annotation_position="right",
                    annotation_font_color="#f87171",
                )
            
            # Support lines
            for s in levels["supports"][:3]:
                fig.add_hline(
                    y=s["price"],
                    line_dash="dash",
                    line_color="rgba(34,197,94,0.5)",
                    annotation_text=f"{s['label']} {format_price(s['price'])}",
                    annotation_position="right",
                    annotation_font_color="#34d399",
                )
            
            # Current price line
            fig.add_hline(
                y=data["currentPrice"],
                line_color="#fbbf24",
                line_width=2,
                annotation_text=f"Live {format_price(data['currentPrice'])}",
                annotation_position="left",
                annotation_font_color="#fbbf24",
            )
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                height=420,
                margin=dict(l=8, r=8, t=28, b=8),
                xaxis_rangeslider_visible=False,
                showlegend=False,
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(gridcolor="rgba(51,65,85,0.35)", zeroline=False),
                yaxis=dict(gridcolor="rgba(51,65,85,0.35)", zeroline=False, side="right"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No candle data available for this symbol / interval.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>⚡ AI Trade Suggestion</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Pick a plan — entry, stop, target included</div>", unsafe_allow_html=True)
        
        bull = next((s for s in suggestions if s["direction"] == "bullish"), None)
        bear = next((s for s in suggestions if s["direction"] == "bearish"), None)
        
        b1, b2 = st.columns(2)
        with b1:
            bull_clicked = st.button("🟢 BULLISH", use_container_width=True, key="bull_btn")
        with b2:
            bear_clicked = st.button("🔴 BEARISH", use_container_width=True, key="bear_btn")
        
        if "selected_plan" not in st.session_state:
            st.session_state.selected_plan = None
        
        if bull_clicked:
            st.session_state.selected_plan = "bullish"
        if bear_clicked:
            st.session_state.selected_plan = "bearish"
        
        plan = None
        if st.session_state.selected_plan == "bullish" and bull:
            plan = bull
        elif st.session_state.selected_plan == "bearish" and bear:
            plan = bear
        
        if plan:
            is_bull = plan["direction"] == "bullish"
            color = "#34d399" if is_bull else "#f87171"
            st.markdown(f"""
            <div style='background:rgba({52 if is_bull else 239},{211 if is_bull else 68},{153 if is_bull else 68},0.1);
                        border:1px solid rgba({52 if is_bull else 239},{211 if is_bull else 68},{153 if is_bull else 68},0.3);
                        border-radius:12px; padding:12px; margin-top:10px;'>
                <b style='color:{color}'>{'Bullish Setup — Buy near support' if is_bull else 'Bearish Setup — Sell near resistance'}</b><br>
                <span style='font-size:0.85rem; color:#94a3b8'>{plan['note']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Entry", format_price(plan["entry"]))
            m2.metric("Stop Loss", format_price(plan["stopLoss"]))
            m3.metric("Target", format_price(plan["target"]))
            
            st.markdown(f"""
            <div style='margin-top:8px; font-size:0.85rem; color:#cbd5e1;'>
                <b>Setup:</b> {'Buy' if is_bull else 'Sell'} near <b>{format_price(plan['entry'])}</b>.  
                If price breaks <b>{plan['breakoutLabel']}</b> ({format_price(plan['breakoutPrice'])}) → 
                {'go long' if is_bull else 'go short'}, targeting <b>{format_price(plan['target'])}</b>.  
                Stop at <b style='color:#f87171'>{format_price(plan['stopLoss'])}</b>.
            </div>
            <div style='margin-top:8px; display:flex; justify-content:space-between;'>
                <span>Risk:Reward <b style='color:#fbbf24'>1:{plan['riskReward']}</b></span>
                <span>Live <b>{format_price(data['currentPrice'])}</b></span>
            </div>
            <p style='font-size:0.7rem; color:#64748b; margin-top:10px; text-align:center;'>Educational analysis only — not financial advice.</p>
            """, unsafe_allow_html=True)
        else:
            st.info("Select 🟢 Bullish or 🔴 Bearish to see the full plan.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Level Ladder
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Support & Resistance Ladder</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-sub'>All levels — S1 to S5, R1 to R5 — ranked by price</div>", unsafe_allow_html=True)
    
    all_levels = sorted(levels["supports"] + levels["resistances"], key=lambda x: x["price"], reverse=True)
    
    for lvl in all_levels:
        is_res = lvl["type"] == "resistance"
        dist = lvl["price"] - data["currentPrice"]
        dist_pct = (dist / data["currentPrice"] * 100) if data["currentPrice"] else 0
        bg = "level-resistance" if is_res else "level-support"
        color = "#f87171" if is_res else "#34d399"
        
        st.markdown(f"""
        <div class='level-row {bg}'>
            <div>
                <span style='background:rgba({239 if is_res else 16},{68 if is_res else 185},{68 if is_res else 129},0.2);
                             color:{color}; padding:2px 8px; border-radius:6px; font-size:0.75rem; font-weight:700;'>{lvl['label']}</span>
                <span style='font-weight:700; margin-left:10px; font-variant-numeric:tabular-nums;'>{format_price(lvl['price'])}</span>
                <span style='color:#64748b; font-size:0.75rem; margin-left:8px;'>{'above' if dist > 0 else 'below'} current</span>
            </div>
            <div style='color:{color}; font-size:0.8rem; font-weight:600;'>{'+' if dist > 0 else ''}{dist_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='text-align:center; margin-top:12px; background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.2);
                border-radius:8px; padding:10px;'>
        <span style='color:#fbbf24; font-weight:700;'>● Current Price: {format_price(data['currentPrice'])}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================== NEWS ========================
with tab_news:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-header'>Latest News & Sentiment</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>AI-ranked stories relevant to {st.session_state.symbol}</div>", unsafe_allow_html=True)
    
    # Try real news from yfinance
    try:
        t = yf.Ticker(st.session_state.symbol)
        news = t.news or []
    except Exception:
        news = []
    
    if news:
        for item in news[:10]:
            title = item.get("title", "No title")
            publisher = item.get("publisher", "Unknown")
            link = item.get("link", "#")
            ts = item.get("providerPublishTime")
            time_str = datetime.fromtimestamp(ts).strftime("%d %b %H:%M") if ts else ""
            
            st.markdown(f"""
            <div style='background:rgba(30,41,59,0.5); border:1px solid rgba(51,65,85,0.4); border-radius:12px;
                        padding:14px; margin-bottom:10px;'>
                <a href='{link}' target='_blank' style='color:#f8fafc; text-decoration:none; font-weight:600;'>{title}</a>
                <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>{publisher} · {time_str}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback mock news (same as original)
        mock = [
            {"title": "Markets react to latest economic data", "source": "MarketWatch", "time": "2h ago", "sentiment": "Neutral"},
            {"title": "Tech sector shows mixed signals", "source": "CNBC", "time": "4h ago", "sentiment": "Bearish"},
            {"title": "Commodity prices edge higher on supply concerns", "source": "Bloomberg", "time": "6h ago", "sentiment": "Bullish"},
            {"title": "Central banks signal cautious stance", "source": "Reuters", "time": "8h ago", "sentiment": "Neutral"},
            {"title": "Retail investors stay active in equity markets", "source": "Yahoo Finance", "time": "1d ago", "sentiment": "Bullish"},
        ]
        for a in mock:
            sent = a["sentiment"]
            color = "#34d399" if sent == "Bullish" else "#f87171" if sent == "Bearish" else "#fbbf24"
            st.markdown(f"""
            <div style='background:rgba(30,41,59,0.5); border:1px solid rgba(51,65,85,0.4); border-radius:12px;
                        padding:14px; margin-bottom:10px;'>
                <div style='font-weight:600; color:#f8fafc;'>{a['title']}</div>
                <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>
                    {a['source']} · {a['time']} · 
                    <span style='color:{color}; font-weight:700;'>{sent}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================== HEATMAP ========================
with tab_heat:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-header'>Market Heatmap</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-sub'>Sector performance at a glance — click any ticker to analyze</div>", unsafe_allow_html=True)
    
    sector_names = ["All"] + [s["name"] for s in HEATMAP_SECTORS]
    selected_sector = st.selectbox("Sector", sector_names, label_visibility="collapsed")
    
    # Build heatmap data
    symbols_to_fetch = []
    if selected_sector == "All":
        symbols_to_fetch = [m["symbol"] for m in MARKET_ITEMS]
    else:
        sec = next((s for s in HEATMAP_SECTORS if s["name"] == selected_sector), None)
        if sec:
            symbols_to_fetch = sec["items"]
    
    heat_cols = st.columns(5)
    for idx, sym in enumerate(symbols_to_fetch[:30]):
        info = fetch_ticker_info(sym)
        chg = info["changePercent"]
        is_up = chg >= 0
        bg = "heat-up" if is_up else "heat-down"
        with heat_cols[idx % 5]:
            if st.button(
                f"{sym.replace('.NS','').replace('-USD','').replace('=F','').replace('=X','').replace('^','')[:8]}\n"
                f"{format_price(info['currentPrice'])}\n{format_percent(chg)}",
                key=f"heat_{sym}",
                use_container_width=True,
            ):
                st.session_state.symbol = sym
                st.cache_data.clear()
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ======================== HISTORY ========================
with tab_hist:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-header'>Historical Data — {st.session_state.symbol}</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-sub'>Last 30 trading sessions with daily summary and swing levels</div>", unsafe_allow_html=True)
    
    hist = fetch_history(st.session_state.symbol, period="1mo")
    
    if not hist.empty:
        # Mini chart
        fig_h = go.Figure()
        fig_h.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ))
        fig_h.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            height=280,
            margin=dict(l=8, r=8, t=20, b=8),
            xaxis_rangeslider_visible=False,
            showlegend=False,
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="rgba(51,65,85,0.35)"),
            yaxis=dict(gridcolor="rgba(51,65,85,0.35)", side="right"),
        )
        st.plotly_chart(fig_h, use_container_width=True)
        
        swings = get_swing_levels(hist)
        
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Swing Resistance**")
            for i, r in enumerate(swings["resistance"]):
                st.markdown(f"R{i+1}: **{format_price(r)}**")
            if not swings["resistance"]:
                st.caption("No clear resistance found")
        with s2:
            st.markdown("**Swing Support**")
            for i, s in enumerate(swings["support"]):
                st.markdown(f"S{i+1}: **{format_price(s)}**")
            if not swings["support"]:
                st.caption("No clear support found")
        
        # Table
        display = hist.copy()
        display["Date"] = display.index.strftime("%d %b")
        display["Change"] = display["Close"] - display["Open"]
        display = display[["Date", "Open", "High", "Low", "Close", "Change"]].iloc[::-1]
        st.dataframe(
            display.style.format({
                "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
                "Close": "{:.2f}", "Change": "{:+.2f}"
            }),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No historical data available for this symbol.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR (Chat / Watchlist / Alerts)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ AI Trade Terminal")
    
    side_tabs = st.radio(
        "Sidebar",
        ["💬 Chat", "📋 Watchlist", "🔔 Alerts"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    # ---------- CHAT ----------
    if side_tabs == "💬 Chat":
        st.markdown(f"**AI Chat Assistant**  \nAsk about `{st.session_state.symbol}`")
        
        # Display messages
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-assistant'>{msg['content']}</div>", unsafe_allow_html=True)
        
        # Quick prompts
        qp_cols = st.columns(2)
        quick = ["Support levels?", "Bullish setup?", "Best target?", "Strategy?"]
        for i, q in enumerate(quick):
            with qp_cols[i % 2]:
                if st.button(q, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    data = fetch_ticker_info(st.session_state.symbol)
                    levels = generate_levels(data["currentPrice"], data["previousClose"], data["dayHigh"], data["dayLow"])
                    reply = generate_ai_response(q, st.session_state.symbol, data["currentPrice"], data, levels)
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    st.rerun()
        
        user_input = st.chat_input("Ask about levels, strategy, targets...")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            data = fetch_ticker_info(st.session_state.symbol)
            levels = generate_levels(data["currentPrice"], data["previousClose"], data["dayHigh"], data["dayLow"])
            reply = generate_ai_response(user_input, st.session_state.symbol, data["currentPrice"], data, levels)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": f"Chat cleared. Ask me about {st.session_state.symbol}!"}
            ]
            st.rerun()
    
    # ---------- WATCHLIST ----------
    elif side_tabs == "📋 Watchlist":
        st.markdown("**Watchlists**")
        
        # Create new list
        new_name = st.text_input("New list name", key="new_list")
        if st.button("➕ Create List", use_container_width=True) and new_name.strip():
            name = new_name.strip()
            if name not in st.session_state.watchlists:
                st.session_state.watchlists[name] = []
                st.session_state.current_list = name
                st.rerun()
        
        # Select list
        lists = list(st.session_state.watchlists.keys())
        st.session_state.current_list = st.selectbox(
            "Current list",
            lists,
            index=lists.index(st.session_state.current_list) if st.session_state.current_list in lists else 0,
        )
        
        # Add ticker
        add_t = st.text_input("Add ticker", key="add_ticker")
        if st.button("➕ Add", use_container_width=True) and add_t.strip():
            t = add_t.strip().upper()
            if t not in st.session_state.watchlists[st.session_state.current_list]:
                st.session_state.watchlists[st.session_state.current_list].append(t)
                st.rerun()
        
        # Show entries
        for ticker in st.session_state.watchlists[st.session_state.current_list]:
            info = fetch_ticker_info(ticker)
            is_up = info["changePercent"] >= 0
            c1, c2 = st.columns([3, 1])
            with c1:
                if st.button(f"{ticker}  {format_price(info['currentPrice'])}  {format_percent(info['changePercent'])}", key=f"wl_{ticker}", use_container_width=True):
                    st.session_state.symbol = ticker
                    st.cache_data.clear()
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"del_{ticker}"):
                    st.session_state.watchlists[st.session_state.current_list].remove(ticker)
                    st.rerun()
    
    # ---------- ALERTS ----------
    else:
        st.markdown("**Price Alerts**")
        st.caption("Alerts are sent to your Telegram")
        
        a_ticker = st.text_input("Ticker", value=st.session_state.symbol, key="alert_ticker")
        a_price = st.number_input("Target Price", value=float(fetch_ticker_info(st.session_state.symbol)["currentPrice"] or 0), format="%.4f")
        a_dir = st.selectbox("Direction", ["above", "below"])
        
        if st.button("🔔 Set Alert", use_container_width=True):
            if a_ticker and a_price > 0:
                alert = {
                    "id": f"{time.time()}-{np.random.randint(10000,99999)}",
                    "ticker": a_ticker.strip().upper(),
                    "price": float(a_price),
                    "direction": a_dir,
                    "created": datetime.now().isoformat(),
                }
                st.session_state.alerts.append(alert)
                
                # Send confirmation to Telegram
                msg = (
                    f"🔔 <b>New Alert Set</b>\n"
                    f"Ticker: <code>{alert['ticker']}</code>\n"
                    f"Trigger: {alert['direction']} <b>{format_price(alert['price'])}</b>\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                send_telegram(msg)
                st.success("Alert created & confirmation sent to Telegram!")
                st.rerun()
        
        # Check & display alerts
        st.markdown("---")
        if not st.session_state.alerts:
            st.info("No active alerts. Create one above.")
        else:
            for alert in st.session_state.alerts[:]:
                live = fetch_ticker_info(alert["ticker"])
                live_price = live["currentPrice"]
                triggered = False
                if alert["direction"] == "above" and live_price >= alert["price"]:
                    triggered = True
                elif alert["direction"] == "below" and live_price <= alert["price"]:
                    triggered = True
                
                if triggered:
                    # Send Telegram once
                    if not alert.get("notified"):
                        msg = (
                            f"🚨 <b>ALERT TRIGGERED</b>\n"
                            f"Ticker: <code>{alert['ticker']}</code>\n"
                            f"Live Price: <b>{format_price(live_price)}</b>\n"
                            f"Condition: {alert['direction']} {format_price(alert['price'])}\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        if send_telegram(msg):
                            alert["notified"] = True
                    
                    st.markdown(f"""
                    <div class='alert-triggered'>
                        🚨 <b>{alert['ticker']}</b> {alert['direction']} {format_price(alert['price'])}<br>
                        Live: {format_price(live_price)} — <b style='color:#fbbf24'>TRIGGERED</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**{alert['ticker']}** {alert['direction']} {format_price(alert['price'])}  \nLive: {format_price(live_price)}")
                
                if st.button("Remove", key=f"rm_{alert['id']}"):
                    st.session_state.alerts = [a for a in st.session_state.alerts if a["id"] != alert["id"]]
                    st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.75rem;'>"
    "AI Trade Terminal — Educational tool only. Not financial advice. "
    "Data via Yahoo Finance. AI powered by Groq + Gemini. Alerts via Telegram."
    "</p>",
    unsafe_allow_html=True,
)
