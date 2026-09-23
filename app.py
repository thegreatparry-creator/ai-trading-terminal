"""
AI Trade Terminal - Streamlit Edition
=====================================
Recreated to closely match the original React UI screenshots.

- Layout: Main content left + right sidebar (Chat / Watch / Alerts)
- India NSE auto-appends .NS
- Real data via yfinance (falls back to last available day when market closed)
- Proper heatmap colors (green = up, red = down)
- Support/Resistance S1-S5 / R1-R5 from last day OHLC
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------
GROQ_KEY = "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"
GEMINI_KEY = "AQ.Ab8RN6JtGdVf9VFtpeo2_7BYDuZQZZlhMIbQxKFX1noZ4UnTSQ"
TELEGRAM_BOT_TOKEN = "8794257218:AAGYGDqPUEJdI3UahL07Pe86IgcLCfIn20g"
TELEGRAM_CHAT_ID = "8600332637"

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Trade Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

    .heat-tile {
        border-radius: 10px; padding: 10px 8px; text-align: center;
        margin-bottom: 6px; min-height: 68px;
    }
    .heat-up { background: rgba(16,185,129,0.18); border: 1px solid rgba(16,185,129,0.35); }
    .heat-down { background: rgba(239,68,68,0.18); border: 1px solid rgba(239,68,68,0.35); }
    .heat-flat { background: rgba(100,116,139,0.15); border: 1px solid rgba(100,116,139,0.3); }

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
    {"symbol": "RELIANCE", "name": "Reliance Industries", "category": "india", "yf": "RELIANCE.NS"},
    {"symbol": "TCS", "name": "Tata Consultancy", "category": "india", "yf": "TCS.NS"},
    {"symbol": "INFY", "name": "Infosys", "category": "india", "yf": "INFY.NS"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "category": "india", "yf": "HDFCBANK.NS"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "category": "india", "yf": "ICICIBANK.NS"},
    {"symbol": "SBIN", "name": "State Bank of India", "category": "india", "yf": "SBIN.NS"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors", "category": "india", "yf": "TATAMOTORS.NS"},
    {"symbol": "ITC", "name": "ITC Ltd", "category": "india", "yf": "ITC.NS"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharma", "category": "india", "yf": "SUNPHARMA.NS"},
    {"symbol": "TATASTEEL", "name": "Tata Steel", "category": "india", "yf": "TATASTEEL.NS"},
    {"symbol": "AAPL", "name": "Apple", "category": "us", "yf": "AAPL"},
    {"symbol": "MSFT", "name": "Microsoft", "category": "us", "yf": "MSFT"},
    {"symbol": "NVDA", "name": "NVIDIA", "category": "us", "yf": "NVDA"},
    {"symbol": "GOOGL", "name": "Alphabet", "category": "us", "yf": "GOOGL"},
    {"symbol": "AMZN", "name": "Amazon", "category": "us", "yf": "AMZN"},
    {"symbol": "META", "name": "Meta", "category": "us", "yf": "META"},
    {"symbol": "TSLA", "name": "Tesla", "category": "us", "yf": "TSLA"},
    {"symbol": "JPM", "name": "JPMorgan", "category": "us", "yf": "JPM"},
    {"symbol": "BTC", "name": "Bitcoin", "category": "crypto", "yf": "BTC-USD"},
    {"symbol": "ETH", "name": "Ethereum", "category": "crypto", "yf": "ETH-USD"},
    {"symbol": "BNB", "name": "BNB", "category": "crypto", "yf": "BNB-USD"},
    {"symbol": "SOL", "name": "Solana", "category": "crypto", "yf": "SOL-USD"},
    {"symbol": "XRP", "name": "XRP", "category": "crypto", "yf": "XRP-USD"},
    {"symbol": "ADA", "name": "Cardano", "category": "crypto", "yf": "ADA-USD"},
    {"symbol": "DOGE", "name": "Dogecoin", "category": "crypto", "yf": "DOGE-USD"},
    {"symbol": "GC", "name": "Gold", "category": "commodities", "yf": "GC=F"},
    {"symbol": "SI", "name": "Silver", "category": "commodities", "yf": "SI=F"},
    {"symbol": "CL", "name": "Crude Oil", "category": "commodities", "yf": "CL=F"},
    {"symbol": "NG", "name": "Natural Gas", "category": "commodities", "yf": "NG=F"},
    {"symbol": "HG", "name": "Copper", "category": "commodities", "yf": "HG=F"},
    {"symbol": "USDINR", "name": "USD/INR", "category": "forex", "yf": "USDINR=X"},
    {"symbol": "EURUSD", "name": "EUR/USD", "category": "forex", "yf": "EURUSD=X"},
    {"symbol": "GBPUSD", "name": "GBP/USD", "category": "forex", "yf": "GBPUSD=X"},
    {"symbol": "USDJPY", "name": "USD/JPY", "category": "forex", "yf": "USDJPY=X"},
    {"symbol": "AUDUSD", "name": "AUD/USD", "category": "forex", "yf": "AUDUSD=X"},
    {"symbol": "NIFTY", "name": "Nifty 50", "category": "indices", "yf": "^NSEI"},
    {"symbol": "SENSEX", "name": "Sensex", "category": "indices", "yf": "^BSESN"},
    {"symbol": "SPX", "name": "S&P 500", "category": "indices", "yf": "^GSPC"},
    {"symbol": "DJI", "name": "Dow Jones", "category": "indices", "yf": "^DJI"},
    {"symbol": "IXIC", "name": "Nasdaq", "category": "indices", "yf": "^IXIC"},
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

def generate_suggestions(price: float, levels: Dict) -> List[Dict]:
    if not levels or price <= 0:
        return []
    out = []
    s1, s2, r1 = levels.get("S1"), levels.get("S2"), levels.get("R1")
    if s1 and s2 and r1:
        risk = abs(s1 - s2)
        reward = abs(r1 - s1)
        out.append({"direction": "bullish", "entry": round(s1, 2), "stop": round(s2, 2),
                    "target": round(r1, 2), "rr": round(reward / risk, 2) if risk > 0 else 0})
    r1, r2, s1 = levels.get("R1"), levels.get("R2"), levels.get("S1")
    if r1 and r2 and s1:
        risk = abs(r2 - r1)
        reward = abs(r1 - s1)
        out.append({"direction": "bearish", "entry": round(r1, 2), "stop": round(r2, 2),
                    "target": round(s1, 2), "rr": round(reward / risk, 2) if risk > 0 else 0})
    return out

def send_telegram(msg: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def ai_chat(prompt: str, symbol: str, data: Dict, levels: Dict) -> str:
    context = f"Symbol: {symbol}, Price: {data.get('currentPrice')}, Change: {data.get('changePercent'):.2f}%, Levels: {levels}"
    full = f"You are a trading assistant. Context: {context}\nUser: {prompt}\nAnswer briefly."
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        resp = client.chat.completions.create(model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": full}], max_tokens=400)
        return resp.choices[0].message.content
    except Exception:
        pass
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(full).text
    except Exception:
        pass
    p = prompt.lower()
    if "support" in p or "s1" in p:
        return f"Supports for {symbol}: S1={format_price(levels.get('S1',0))}, S2={format_price(levels.get('S2',0))}, S3={format_price(levels.get('S3',0))}"
    if "resist" in p or "r1" in p:
        return f"Resistances for {symbol}: R1={format_price(levels.get('R1',0))}, R2={format_price(levels.get('R2',0))}, R3={format_price(levels.get('R3',0))}"
    if "bull" in p:
        return f"Bullish: Watch bounce near S1 ({format_price(levels.get('S1',0))}). Target R1 ({format_price(levels.get('R1',0))})."
    if "bear" in p:
        return f"Bearish: Watch rejection near R1 ({format_price(levels.get('R1',0))}). Target S1 ({format_price(levels.get('S1',0))})."
    return f"Price of {symbol} is {format_price(data.get('currentPrice',0))}. Ask about support, resistance, bullish or bearish setup."

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
if "show_plan" not in st.session_state:
    st.session_state.show_plan = None

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
        cat_labels = [c["label"] for c in MARKET_CATEGORIES]
        cat_keys = [c["key"] for c in MARKET_CATEGORIES]
        selected_cat_label = st.radio("Market", cat_labels, horizontal=True, label_visibility="collapsed",
            index=cat_keys.index(st.session_state.market_cat) if st.session_state.market_cat in cat_keys else 0)
        st.session_state.market_cat = cat_keys[cat_labels.index(selected_cat_label)]

        items = [m for m in MARKET_ITEMS if m["category"] == st.session_state.market_cat]
        tile_cols = st.columns(5)
        for i, item in enumerate(items):
            with tile_cols[i % 5]:
                if st.button(f"{item['symbol']}", key=f"tile_{item['symbol']}", use_container_width=True):
                    st.session_state.selected_symbol = item["symbol"]
                    st.session_state.selected_yf = item["yf"]
                    st.rerun()

        search_col1, search_col2 = st.columns([4, 1])
        with search_col1:
            search_input = st.text_input("Search", placeholder="Type RELIANCE or AAPL or BTC — no .NS needed", label_visibility="collapsed")
        with search_col2:
            if st.button("Search", use_container_width=True) and search_input.strip():
                yf_sym = resolve_yf_symbol(search_input, st.session_state.market_cat)
                st.session_state.selected_yf = yf_sym
                st.session_state.selected_symbol = search_input.strip().upper().replace(".NS", "")
                st.rerun()

        yf_sym = st.session_state.selected_yf
        data = fetch_stock_data(yf_sym)
        price = data["currentPrice"]
        chg = data["changePercent"]
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
            st.markdown("<div class='card-sub'>Pick a plan — entry, stop, target included</div>", unsafe_allow_html=True)
            bull = next((s for s in suggestions if s["direction"] == "bullish"), None)
            bear = next((s for s in suggestions if s["direction"] == "bearish"), None)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("▲ BULLISH\nBuy plan", key="bull_btn", use_container_width=True):
                    st.session_state.show_plan = "bull"
            with b2:
                if st.button("▼ BEARISH\nSell plan", key="bear_btn", use_container_width=True):
                    st.session_state.show_plan = "bear"
            plan = st.session_state.show_plan
            if plan == "bull" and bull:
                st.success(f"**Entry:** {format_price(bull['entry'])}")
                st.write(f"**Stop:** {format_price(bull['stop'])}")
                st.write(f"**Target:** {format_price(bull['target'])}")
                st.write(f"**R:R:** {bull['rr']}")
            elif plan == "bear" and bear:
                st.error(f"**Entry:** {format_price(bear['entry'])}")
                st.write(f"**Stop:** {format_price(bear['stop'])}")
                st.write(f"**Target:** {format_price(bear['target'])}")
                st.write(f"**R:R:** {bear['rr']}")
            else:
                st.caption("Select Bullish or Bearish to see the full plan.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Support & Resistance Ladder</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>S1–S5 / R1–R5 from last available day OHLC</div>", unsafe_allow_html=True)
        if levels:
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
        st.markdown("<div class='card-header'>Latest News & Sentiment</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-sub'>Stories for {st.session_state.selected_symbol}</div>", unsafe_allow_html=True)
        try:
            news = yf.Ticker(st.session_state.selected_yf).news or []
        except Exception:
            news = []
        if news:
            for n in news[:10]:
                title = n.get("title") or (n.get("content") or {}).get("title") or "News"
                pub = n.get("publisher") or "Source"
                link = n.get("link") or "#"
                st.markdown(f"**[{title}]({link})**")
                st.caption(pub)
                st.markdown("---")
        else:
            st.info("No recent news. Try another ticker.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "Heatmap":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Market Heatmap</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Green = up · Red = down · Click Open to analyze</div>", unsafe_allow_html=True)
        sectors = ["All", "india", "us", "crypto", "commodities", "forex", "indices"]
        sec = st.radio("Filter", sectors, horizontal=True, label_visibility="collapsed")
        pool = MARKET_ITEMS if sec == "All" else [m for m in MARKET_ITEMS if m["category"] == sec]
        cols = st.columns(5)
        for i, item in enumerate(pool):
            d = fetch_stock_data(item["yf"])
            chg = d.get("changePercent", 0) or 0
            cls = "heat-up" if chg > 0.05 else ("heat-down" if chg < -0.05 else "heat-flat")
            color = "#34d399" if chg > 0 else ("#f87171" if chg < 0 else "#94a3b8")
            with cols[i % 5]:
                st.markdown(f"<div class='heat-tile {cls}'><div style='font-weight:700'>{item['symbol']}</div><div>{format_price(d.get('currentPrice',0))}</div><div style='color:{color};font-weight:600'>{format_percent(chg)}</div></div>", unsafe_allow_html=True)
                if st.button("Open", key=f"hm_{item['symbol']}", use_container_width=True):
                    st.session_state.selected_symbol = item["symbol"]
                    st.session_state.selected_yf = item["yf"]
                    st.session_state.page = "Dashboard"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.page == "History":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-header'>Historical Data — {st.session_state.selected_symbol}</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Last 30 sessions + swing levels</div>", unsafe_allow_html=True)
        hist = fetch_history(st.session_state.selected_yf)
        if hist is not None and not hist.empty:
            colors = ["#34d399" if hist["Close"].iloc[i] >= hist["Open"].iloc[i] else "#f87171" for i in range(len(hist))]
            fig = go.Figure(go.Bar(x=hist.index, y=hist["Close"], marker_color=colors))
            fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", height=260,
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False, font=dict(color="#94a3b8"),
                xaxis=dict(gridcolor="rgba(51,65,85,0.3)"), yaxis=dict(gridcolor="rgba(51,65,85,0.3)", side="right"))
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='level-row level-resistance'><b>Swing Resistance</b> {format_price(float(hist['High'].max()))}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='level-row level-support'><b>Swing Support</b> {format_price(float(hist['Low'].min()))}</div>", unsafe_allow_html=True)
            show = hist[["Open", "High", "Low", "Close"]].tail(15).iloc[::-1]
            show.index = show.index.strftime("%b %d")
            st.dataframe(show.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("No historical data.")
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    side_tabs = st.radio("side", ["Chat", "Watch", "Alerts"], horizontal=True, label_visibility="collapsed",
                         index=["Chat", "Watch", "Alerts"].index(st.session_state.sidebar_tab))
    st.session_state.sidebar_tab = side_tabs

    if side_tabs == "Chat":
        st.markdown("##### 🤖 AI Chat Assistant")
        st.caption(f"Ask about {st.session_state.selected_symbol}")
        for msg in st.session_state.chat_history[-8:]:
            cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
            st.markdown(f"<div class='{cls}'>{msg['text']}</div>", unsafe_allow_html=True)
        user_msg = st.chat_input("Ask about levels, strategy...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "text": user_msg})
            data = fetch_stock_data(st.session_state.selected_yf)
            levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or data["currentPrice"])
            reply = ai_chat(user_msg, st.session_state.selected_symbol, data, levels)
            st.session_state.chat_history.append({"role": "assistant", "text": reply})
            st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("Support?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": "What are support levels?"})
            data = fetch_stock_data(st.session_state.selected_yf)
            levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or data["currentPrice"])
            st.session_state.chat_history.append({"role": "assistant", "text": ai_chat("support", st.session_state.selected_symbol, data, levels)})
            st.rerun()
        if c2.button("Bullish?", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "text": "Bullish setup?"})
            data = fetch_stock_data(st.session_state.selected_yf)
            levels = generate_levels(data["dayHigh"], data["dayLow"], data["previousClose"] or data["currentPrice"])
            st.session_state.chat_history.append({"role": "assistant", "text": ai_chat("bullish", st.session_state.selected_symbol, data, levels)})
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
        a_price = st.number_input("Target", min_value=0.0, value=float(fetch_stock_data(st.session_state.selected_yf).get("currentPrice", 0) or 0), format="%.2f")
        a_dir = st.selectbox("Direction", ["Above", "Below"])
        if st.button("＋ Set Alert", use_container_width=True):
            yf_s = resolve_yf_symbol(a_sym, st.session_state.market_cat)
            st.session_state.alerts.append({"symbol": yf_s, "price": a_price, "direction": a_dir})
            send_telegram(f"Alert set: {yf_s} {a_dir} {a_price}")
            st.success("Alert set")
            st.rerun()
        if not st.session_state.alerts:
            st.caption("No alerts yet.")
        else:
            for i, a in enumerate(st.session_state.alerts):
                d = fetch_stock_data(a["symbol"])
                cur = d.get("currentPrice", 0)
                triggered = (a["direction"] == "Above" and cur >= a["price"]) or (a["direction"] == "Below" and cur <= a["price"])
                if triggered:
                    st.warning(f"🔔 {a['symbol']} hit {a['direction']} {a['price']} (now {format_price(cur)})")
                    send_telegram(f"ALERT: {a['symbol']} is {format_price(cur)}")
                else:
                    st.write(f"{a['symbol']} {a['direction']} {a['price']} · now {format_price(cur)}")
                if st.button("Remove", key=f"ar_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;color:#475569;font-size:0.75rem;margin-top:1.5rem'>AI Trade Terminal · When market is closed, last available day data + levels are shown</div>", unsafe_allow_html=True)
