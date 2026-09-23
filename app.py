import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import re
import html as html_module
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import email.utils
from datetime import datetime, timezone, timedelta
from groq import Groq
from google import genai as google_genai
import requests


# ================== API KEYS ==================
# ⚠️ SECURITY NOTE: these were hardcoded in the original file. Anything pasted into a
# chat/shared file should be treated as compromised — please rotate all 4 of these
# (Gemini, Groq, Telegram bot token) in their respective dashboards, then keep the new
# ones ONLY in .streamlit/secrets.toml (never in code you share or commit). The helper
# below reads from st.secrets first and only falls back to the old hardcoded value so
# the app keeps working until you migrate.
def _get_secret(key, fallback):
    try:
        return st.secrets[key]
    except Exception:
        return fallback

GEMINI_KEY = _get_secret("GEMINI_KEY", "AQ.Ab8RN6JtGdVf9VFtpeo2_7BYDuZQZZlhMIbQxKFX1noZ4UnTSQ")
GROQ_KEY = _get_secret("GROQ_KEY", "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")


# ================== TELEGRAM BOT CONFIG ==================
TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN", "8794257218:AAGYGDqPUEJdI3UahL07Pe86IgcLCfIn20g")
TELEGRAM_CHAT_ID = _get_secret("TELEGRAM_CHAT_ID", "8600332637")


def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=data, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


groq_client = Groq(api_key=GROQ_KEY)
gemini_client = genai.Client(api_key=GEMINI_KEY)


# ================== AI MODEL MAPS ==================
# Groq retired llama-3.1-8b-instant / llama-3.3-70b-versatile for free & dev-tier keys
# (Aug 16, 2026), and mixtral-8x7b-32768 / gemma2-9b-it were removed even earlier.
# These are the current production/preview models on a normal Groq API key.
GROQ_MODEL_MAP = {
    "Groq (GPT-OSS 20B - Fast)": "openai/gpt-oss-20b",
    "Groq (GPT-OSS 120B - Smartest)": "openai/gpt-oss-120b",
    "Groq (Qwen3 32B)": "qwen/qwen3-32b",
    "Groq (Kimi K2)": "moonshotai/kimi-k2-instruct",
}

# google-generativeai (the old "import google.generativeai as genai" SDK) is deprecated
# and gemini-1.5-* models are long retired. Using the current google-genai SDK instead.
GEMINI_MODEL_MAP = {
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.1 Pro (Preview)": "gemini-3.1-pro-preview",
}


# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Institutional Terminal", layout="wide")

# Mobile-friendly typography & news-card styling. Streamlit already stacks columns on
# narrow screens; this just makes text sizes and tap targets comfortable for someone
# scanning charts/news on a phone during market hours.
st.markdown("""
<style>
    html, body, [class*="css"]  { font-size: 16px; }
    h1 { font-size: 1.65rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.12rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.82rem !important; }
    .stButton>button { padding: 0.5rem 1rem; font-size: 1rem; border-radius: 8px; }
    .news-card {
        border: 1px solid rgba(150,150,150,0.28);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .news-title { font-size: 1.05rem; font-weight: 700; line-height: 1.35; margin-bottom: 5px; }
    .news-summary { font-size: 0.95rem; line-height: 1.5; opacity: 0.85; margin-bottom: 7px; }
    .news-meta { font-size: 0.78rem; opacity: 0.6; }
    .news-meta a { text-decoration: none; }
    @media (max-width: 640px) {
        h1 { font-size: 1.35rem !important; }
        h2 { font-size: 1.15rem !important; }
        .news-title { font-size: 1.08rem; }
        .news-summary { font-size: 0.98rem; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Trading Terminal - PRO")


def yahoo_symbol_search(query, max_results=8):
    """Looks up ANY stock/index/crypto/forex symbol worldwide via Yahoo Finance's
    search endpoint, so the user can type a company/index name instead of guessing
    the exact ticker + suffix (e.g. 'Nifty 50' correctly resolves to ^NSEI)."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": max_results, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        resp.raise_for_status()
        results = []
        for q in resp.json().get("quotes", []):
            symbol = q.get("symbol")
            if not symbol:
                continue
            results.append({
                "symbol": symbol,
                "name": q.get("shortname") or q.get("longname") or symbol,
                "exchange": q.get("exchange", ""),
                "type": q.get("quoteType", ""),
            })
        return results
    except Exception as e:
        print(f"Symbol search error: {e}")
        return []


# ================== SIDEBAR ==================
st.sidebar.header("🔍 Search Any Stock / Index / Crypto Worldwide")
global_query = st.sidebar.text_input("Type a name or ticker — e.g. 'Nifty 50', 'Apple', 'Reliance', 'Bitcoin', 'EURUSD'", key="global_search_query")

symbol_override = None
if global_query.strip():
    search_results = yahoo_symbol_search(global_query)
    if search_results:
        options = [f"{r['symbol']} — {r['name']} ({r['exchange']} · {r['type']})" for r in search_results]
        picked = st.sidebar.radio("Matching symbols", options, key="global_search_pick")
        symbol_override = search_results[options.index(picked)]["symbol"]
    else:
        st.sidebar.caption("No matches found. Try a different spelling, or use the manual selector below.")

with st.sidebar.expander("⚙️ Manual selector (if you already know the exact suffix)", expanded=not bool(global_query.strip())):
    select_market = st.selectbox("Asset Class / Country",
        ["India - NSE (.NS)", "India Index Benchmark", "United States (No Suffix)", "Cryptocurrency (-USD)", "Forex Currency (=X)", "Commodities", "Global Indices"])

    default_symbol = "RELIANCE"
    if select_market == "India Index Benchmark": default_symbol = "^NSEI"
    elif select_market == "United States (No Suffix)": default_symbol = "AAPL"
    elif select_market == "Cryptocurrency (-USD)": default_symbol = "BTC-USD"
    elif select_market == "Forex Currency (=X)": default_symbol = "EURUSD=X"
    elif select_market == "Commodities": default_symbol = "GC=F"
    elif select_market == "Global Indices": default_symbol = "^GSPC"
    else: default_symbol = "RELIANCE"

    search_ticker = st.text_input("Enter Ticker / Symbol", default_symbol).strip().upper()

    if select_market == "India - NSE (.NS)" and not search_ticker.endswith(".NS"): manual_symbol = f"{search_ticker}.NS"
    elif select_market == "Cryptocurrency (-USD)" and not search_ticker.endswith("-USD"): manual_symbol = f"{search_ticker}-USD"
    elif select_market == "Forex Currency (=X)" and not search_ticker.endswith("=X"): manual_symbol = f"{search_ticker}=X"
    elif select_market == "Commodities" and not search_ticker.endswith("=F"): manual_symbol = f"{search_ticker}=F"
    else: manual_symbol = search_ticker

# The universal search result (if the person picked one) always wins over the manual
# selector — that's what actually fixes "NIFTY.NS has no data": searching "Nifty 50"
# correctly resolves to ^NSEI instead of guessing a wrong suffix.
if symbol_override:
    full_symbol = symbol_override
    search_ticker = symbol_override
else:
    full_symbol = manual_symbol


st.sidebar.markdown("---")
st.sidebar.header("📡 Global News Sources")
# Global news sources - scans all, shows relevant
global_news_sources = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://alphaideas.in/feed",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1998028306.cms",
    "https://feeds.marketwatch.com/marketwatch/topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://cointelegraph.com/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss",
    "https://www.forexlive.com/servicexml/xml.aspx?xml=1"
]


sentiment_filter = st.sidebar.multiselect("Filter News by Sentiment",
    ["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"],
    default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"])


st.sidebar.markdown("---")
st.sidebar.header("🚀 Optional Features")
enable_fundamentals = st.sidebar.checkbox("📊 Enhanced Fundamentals", value=False)


# ================== WATCHLIST ==================
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {"My Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS"], "Watchlist 2": ["AAPL", "NVDA"], "Crypto": ["BTC-USD", "ETH-USD"]}
if "current_watchlist" not in st.session_state: st.session_state.current_watchlist = "My Stocks"


st.sidebar.markdown("---")
st.sidebar.header("📋 Watchlists")
new_list_name = st.sidebar.text_input("New Watchlist Name")
if st.sidebar.button("➕ Create Watchlist"):
    if new_list_name and new_list_name not in st.session_state.watchlists:
        st.session_state.watchlists[new_list_name] = []
        st.rerun()


selected_watchlist = st.sidebar.selectbox("Select Watchlist", list(st.session_state.watchlists.keys()),
    index=list(st.session_state.watchlists.keys()).index(st.session_state.current_watchlist))
st.session_state.current_watchlist = selected_watchlist


new_stock = st.sidebar.text_input("Add Stock Ticker")
if st.sidebar.button("➕ Add to Watchlist"):
    if new_stock and new_stock not in st.session_state.watchlists[selected_watchlist]:
        st.session_state.watchlists[selected_watchlist].append(new_stock)
        st.rerun()


if st.session_state.watchlists[selected_watchlist]:
    st.sidebar.markdown("### Current Watchlist:")
    for idx, stock in enumerate(st.session_state.watchlists[selected_watchlist]):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"{idx+1}. {stock}")
        if col2.button("🗑️", key=f"remove_{idx}"):
            st.session_state.watchlists[selected_watchlist].remove(stock)
            st.rerun()


# ================== TELEGRAM TEST ==================
st.sidebar.markdown("---")
st.sidebar.header("📱 Telegram Alerts")
if st.sidebar.button("🧪 Test Telegram Alert"):
    test_msg = f"🧪 <b>TEST ALERT</b>\n\n✅ Telegram working!\n🕐 {datetime.now().strftime('%H:%M:%S')}"
    if send_telegram_alert(test_msg): st.sidebar.success("✅ Test sent!")
    else: st.sidebar.error("❌ Failed")


# ================== HELPER FUNCTIONS ==================
def clean_news_summary(raw_html, max_sentences=3, max_chars=240):
    """Turns a raw RSS <description> (often HTML) into a clean 1-3 line summary."""
    if not raw_html:
        return ""
    text = re.sub(r'<[^<]+?>', ' ', raw_html)          # strip HTML tags
    text = html_module.unescape(text)                   # &amp; -> &, etc.
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    text = ' '.join(sentences[:max_sentences]).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(' ', 1)[0].rstrip(',.;:') + "…"
    return text


def harvest_global_news():
    """Fetch news from ALL global sources, keeping a short summary for each headline."""
    news_items = []
    for url in global_news_sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                root = ET.fromstring(resp.read())
                for item in root.findall(".//item")[:8]:  # wider pool per source so relevance filtering has enough to work with
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub = item.findtext("pubDate", "").strip()
                    raw_desc = item.findtext("description", "") or ""
                    summary = clean_news_summary(raw_desc) or title
                    source = item.find("source").text.strip() if item.find("source") is not None else "Global News"
                    is_breaking = False
                    try:
                        dt = email.utils.parsedate_to_datetime(pub)
                        if (datetime.now(timezone.utc) - dt).total_seconds() <= 86400:
                            is_breaking = True
                    except:
                        is_breaking = "hours ago" in pub.lower() or "min ago" in pub.lower()
                    score = 0
                    title_lower = title.lower()
                    for w in ["soars", "jumps", "beats", "rally", "buy", "gain", "rise"]:
                        if w in title_lower: score += 1
                    for w in ["slumps", "falls", "drops", "loss", "sell", "crash", "decline"]:
                        if w in title_lower: score -= 1
                    score = max(-3, min(3, score))
                    news_items.append({"title": title, "summary": summary, "link": link, "source": source, "time": pub, "impact": score, "is_breaking": is_breaking})
        except Exception as e:
            print(f"News source error: {e}")
    return news_items[:60]  # a wider pool; the stock-relevance filter picks the best 5-8 later


def get_relevance_keywords(search_ticker, stock_info):
    """Keywords used to decide whether a headline is actually about the selected stock."""
    raw = re.sub(r'(\.NS|-USD|=X|=F)$', '', search_ticker or "").strip().lower()
    keywords = {raw} if raw else set()
    name = (stock_info or {}).get("longName") or (stock_info or {}).get("shortName") or ""
    if name:
        name_lower = name.lower()
        keywords.add(name_lower)
        first_word = name_lower.split()[0] if name_lower.split() else ""
        if len(first_word) > 2:
            keywords.add(first_word)
    return [k for k in keywords if k and len(k) > 1]


def filter_relevant_news(news_list, keywords, max_items=8):
    """Keeps only headlines that actually mention the stock/company, best matches first."""
    if not keywords:
        return []
    scored = []
    for n in news_list:
        haystack = f"{n['title']} {n.get('summary', '')}".lower()
        hits = sum(1 for k in keywords if k in haystack)
        if hits > 0:
            scored.append((hits, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored][:max_items]


def detect_patterns(df):
    if len(df) < 3: return "Insufficient"
    c, o = df["Close"].values, df["Open"].values
    if c[-1] > o[-1] and c[-2] < o[-2] and c[-1] >= o[-2]: return "🔥 Bullish Engulfing"
    elif c[-1] < o[-1] and c[-2] > o[-2] and c[-1] <= o[-2]: return "⚠️ Bearish Engulfing"
    return "Standard"


def calculate_levels(df, first_15m_open=None):
    if first_15m_open is None or isinstance(first_15m_open, str):
        first_15m_open = float(df["Close"].iloc[-1])
    else:
        first_15m_open = float(first_15m_open)
    
    sqrt_price = math.sqrt(abs(first_15m_open))
    
    gann_levels = {}
    for deg, delta in [(22.5, 22.5/180), (45, 45/180), (67.5, 67.5/180), (90, 90/180), (180, 180/180)]:
        gann_levels[f'R_{deg}'] = math.pow(sqrt_price + delta, 2)
        gann_levels[f'S_{deg}'] = math.pow(sqrt_price - delta, 2)
    
    high, low, close = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), float(df["Close"].iloc[-1])
    pivot = (high + low + close) / 3
    
    return {
        "price": close,
        "pivot": round(pivot, 2),
        "r1": round((2*pivot)-low, 2),
        "r2": round(pivot+(high-low), 2),
        "r3": round(high+2*(pivot-low), 2),
        "r4": round(high+3*(pivot-low), 2),
        "s1": round((2*pivot)-high, 2),
        "s2": round(pivot-(high-low), 2),
        "s3": round(low-2*(high-pivot), 2),
        "s4": round(low-3*(high-pivot), 2),
        "pattern": detect_patterns(df),
        **gann_levels
    }


def build_gann_ladder(quant, open_price):
    """
    Builds one sorted ladder of every Gann-angle level (all resistances + all supports)
    PLUS the 0-degree opening price used as the anchor for the whole square-of-9 grid.
    This is what lets us say 'price is currently between level X and level Y' for
    whatever degree those two levels happen to be — exactly like the ladder shown in
    the reference clip (22.5°, 45°, 67.5°, 90°, 180° stacked above/below the open).
    """
    degs = [22.5, 45, 67.5, 90, 180]
    ladder = [{"label": "0° (Open)", "deg": 0.0, "side": "OPEN", "value": float(open_price)}]
    for d in degs:
        r_val, s_val = quant.get(f"R_{d}"), quant.get(f"S_{d}")
        if r_val is not None:
            ladder.append({"label": f"{d}° R", "deg": d, "side": "R", "value": float(r_val)})
        if s_val is not None:
            ladder.append({"label": f"{d}° S", "deg": d, "side": "S", "value": float(s_val)})
    ladder.sort(key=lambda x: x["value"])
    return ladder


def locate_price_band(ladder, price):
    """
    Returns (lower_level, upper_level) — the two consecutive ladder levels the current
    price is sitting between right now — or the string 'below'/'above' if price has
    broken outside every calculated level (breakdown/breakout zone).
    """
    if not ladder:
        return "below"
    if price <= ladder[0]["value"]:
        return "below"
    if price >= ladder[-1]["value"]:
        return "above"
    for i in range(len(ladder) - 1):
        if ladder[i]["value"] <= price <= ladder[i + 1]["value"]:
            return ladder[i], ladder[i + 1]
    return "below"


def get_chart_zoom_window(ladder, band, price, context=1):
    """
    Picks a sane y-axis window: the current band plus `context` extra levels on each
    side, instead of the full ladder — which is what made the chart unreadable for
    instruments like EURUSD=X where far-out levels dwarf the tight real trading range.
    """
    if not ladder:
        return None
    if isinstance(band, tuple):
        lo_idx, hi_idx = ladder.index(band[0]), ladder.index(band[1])
    else:
        lo_idx, hi_idx = 0, len(ladder) - 1
    start = max(0, lo_idx - context)
    end = min(len(ladder) - 1, hi_idx + context)
    window = ladder[start:end + 1]
    window_vals = [lvl["value"] for lvl in window] + [price]
    y_min, y_max = min(window_vals), max(window_vals)
    pad = (y_max - y_min) * 0.18 if y_max > y_min else max(abs(price) * 0.01, 0.0001)
    return y_min - pad, y_max + pad, window


def detect_order_block(price_df, lookback=60, impulse_atr_mult=1.8, search_back=6):
    """
    Simplified Smart-Money-Concepts order block detector: scans recent candles for a
    strong impulsive move (body much bigger than the recent average range), then marks
    the last opposite-colored candle right before that move as the order block zone —
    a bullish OB (demand) before an up-impulse, a bearish OB (supply) before a down-impulse.
    This is a simplified heuristic, not a certified SMC/ICT tool — treat it as a zone to
    watch, not a signal on its own.
    """
    if price_df is None or len(price_df) < 15:
        return None
    d = price_df.tail(lookback).copy()
    d["body"] = (d["Close"] - d["Open"]).abs()
    d["range"] = d["High"] - d["Low"]
    atr = d["range"].rolling(14, min_periods=5).mean()
    bullish_ob, bearish_ob = None, None
    for i in range(len(d) - 2, 0, -1):
        a = atr.iloc[i]
        if pd.isna(a) or a == 0:
            continue
        if d["body"].iloc[i] <= impulse_atr_mult * a:
            continue
        moved_up = d["Close"].iloc[i] > d["Open"].iloc[i]
        moved_down = d["Close"].iloc[i] < d["Open"].iloc[i]
        if moved_up and bullish_ob is None:
            for j in range(i - 1, max(i - search_back, -1), -1):
                if d["Close"].iloc[j] < d["Open"].iloc[j]:
                    bullish_ob = {"low": float(d["Low"].iloc[j]), "high": float(d["High"].iloc[j]), "time": d.index[j]}
                    break
        if moved_down and bearish_ob is None:
            for j in range(i - 1, max(i - search_back, -1), -1):
                if d["Close"].iloc[j] > d["Open"].iloc[j]:
                    bearish_ob = {"low": float(d["Low"].iloc[j]), "high": float(d["High"].iloc[j]), "time": d.index[j]}
                    break
        if bullish_ob and bearish_ob:
            break
    if not bullish_ob and not bearish_ob:
        return None
    return {"bullish": bullish_ob, "bearish": bearish_ob}


def detect_golden_pocket(price_df, lookback=80):
    """
    Finds the most recent significant swing high/low in the lookback window and returns
    the 61.8%-65% Fibonacci retracement zone ('golden pocket'), plus whether the current
    price is sitting inside it right now.
    """
    if price_df is None or len(price_df) < 15:
        return None
    d = price_df.tail(lookback)
    swing_high, swing_low = float(d["High"].max()), float(d["Low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return None
    uptrend = d["Low"].idxmin() < d["High"].idxmax()  # low printed first -> this is a pullback in an up-move
    if uptrend:
        gp_low, gp_high = swing_high - diff * 0.65, swing_high - diff * 0.618
    else:
        gp_low, gp_high = swing_low + diff * 0.618, swing_low + diff * 0.65
    return {"gp_low": gp_low, "gp_high": gp_high, "swing_high": swing_high, "swing_low": swing_low, "uptrend": uptrend}


def get_15min_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df_15m = stock.history(period="5d", interval="15m")
        if df_15m.empty: return None, None
        today = datetime.now().date()
        df_15m = df_15m[df_15m.index.date >= today - timedelta(days=2)]
        return df_15m, float(df_15m['Open'].iloc[0]) if len(df_15m) > 0 else None
    except: return None, None


def get_historical_15min_days(symbol, days=2):
    try:
        stock = yf.Ticker(symbol)
        df_15m = stock.history(period=f"{days}d", interval="15m")
        if df_15m.empty: return []
        df_15m['Date'] = df_15m.index.date
        daily_data = []
        for date in sorted(df_15m['Date'].unique(), reverse=True)[:2]:
            day_df = df_15m[df_15m['Date'] == date]
            if len(day_df) < 5: continue
            opening = float(day_df['Open'].iloc[0])
            high, low, close = float(day_df['High'].max()), float(day_df['Low'].min()), float(day_df['Close'].iloc[-1])
            pivot = (high + low + close) / 3
            sqrt_open = math.sqrt(abs(opening))
            levels = {}
            for deg, delta in [(22.5, 22.5/180), (45, 45/180), (67.5, 67.5/180), (90, 90/180), (180, 180/180)]:
                levels[f'R_{deg}'] = math.pow(sqrt_open + delta, 2)
                levels[f'S_{deg}'] = math.pow(sqrt_open - delta, 2)
            daily_data.append({
                'Date': date, 'Open': opening, 'High': high, 'Low': low, 'Close': close,
                'Pivot': round(pivot, 2), 'R1': round((2*pivot)-low, 2), 'R2': round(pivot+(high-low), 2),
                'R3': round(high+2*(pivot-low), 2), 'R4': round(high+3*(pivot-low), 2),
                'S1': round((2*pivot)-high, 2), 'S2': round(pivot-(high-low), 2),
                'S3': round(low-2*(high-pivot), 2), 'S4': round(low-3*(high-pivot), 2),
                **levels
            })
        return daily_data
    except: return []


def get_groq_suggestions(symbol, quant, news_dossier):
    prompt = f"Analyze {symbol}: Price={quant['price']}, Pattern={quant['pattern']}, R1={quant['r1']}, S1={quant['s1']}\nNews: {news_dossier}\n\nGive INTRADAY and SWING suggestions with ENTRY, SL, TP."
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error (Groq): {type(e).__name__}: {e}"


def get_chat_response(user_message, context_text, model_choice):
    system_prompt = f"Trading assistant. Context: {context_text}. Answer questions. Educational only."
    try:
        if model_choice in GROQ_MODEL_MAP:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL_MAP[model_choice],
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                temperature=0.3, max_tokens=800
            )
            return response.choices[0].message.content
        elif model_choice in GEMINI_MODEL_MAP:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL_MAP[model_choice],
                contents=f"{system_prompt}\n\nQ: {user_message}"
            )
            return response.text
        else:
            return "⚠️ Invalid model selected"
    except Exception as e:
        # Surface the REAL reason (bad key, rate limit, retired model, etc.) instead of
        # a generic "Error" so it's actually possible to debug from the chat UI.
        return f"⚠️ AI Error ({model_choice}): {type(e).__name__}: {e}"


def run_backtest(symbol, start_date, end_date, strategy):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, interval="1d")
        if df.empty: return None
        trades = []
        for i in range(20, len(df)):
            pivot = (df['High'].iloc[i-1] + df['Low'].iloc[i-1] + df['Close'].iloc[i-1]) / 3
            s1, r1 = (2*pivot) - df['High'].iloc[i-1], (2*pivot) - df['Low'].iloc[i-1]
            if strategy == "Buy at S1" and df['Low'].iloc[i] <= s1: trades.append((s1, pivot))
            elif strategy == "Sell at R1" and df['High'].iloc[i] >= r1: trades.append((r1, pivot))
        if not trades: return {"win_rate": 0, "avg_profit": 0, "total_trades": 0}
        wins = sum(1 for e, x in trades if x > e)
        return {"win_rate": (wins/len(trades))*100, "avg_profit": sum(x-e for e,x in trades)/len(trades), "total_trades": len(trades), "total_profit": sum(x-e for e,x in trades)}
    except: return None


# ================== CACHING ==================
@st.cache_data(ttl=120)
def get_cached_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="3mo", interval="1d")
        info = stock.info
        return df, info
    except: return pd.DataFrame(), {}


@st.cache_data(ttl=300)
def get_cached_global_news():
    return harvest_global_news()


@st.cache_data(ttl=60)
def get_cached_15min_data(symbol):
    return get_15min_data(symbol)


@st.cache_data(ttl=600)
def get_cached_historical(symbol, days):
    return get_historical_15min_days(symbol, days)


# ================== INITIALIZE STATE ==================
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
if "quant_data" not in st.session_state: st.session_state.quant_data = None
if "portfolio" not in st.session_state: st.session_state.portfolio = {}
if "alerts" not in st.session_state: st.session_state.alerts = []


# ================== FETCH DATA ==================
with st.spinner("🔄 Loading market data..."):
    df, stock_info = get_cached_stock_data(full_symbol)


# ================== CREATE TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Live Analysis", "💬 AI Chat", "📊 Market Heatmap",
    "📋 Watchlist & Portfolio", "🔔 Alerts & Backtest", "📅 Historical Data"])


# ================== TAB 1: LIVE ANALYSIS ==================
with tab1:
    st.header(f"📊 Live Analysis: {full_symbol}")
    if df.empty or len(df) < 5:
        st.error(f"❌ No data for {full_symbol}. Use the universal search box in the sidebar to find the right symbol (e.g. search 'Nifty 50' instead of typing NIFTY.NS).")
    else:
        df_15m, opening_15m = get_cached_15min_data(full_symbol)
        quant = calculate_levels(df, opening_15m)
        st.session_state.quant_data = quant
        news_list = get_cached_global_news()
        chart_df = df_15m if (df_15m is not None and not df_15m.empty) else df

        col_main, col_ai = st.columns([2, 1], gap="large")

        # ---------- MAIN LIVE ANALYSIS (left) ----------
        with col_main:
            c1, c2, c3 = st.columns(3)
            c1.metric("💰 Current Price", f"{quant['price']:.2f}")
            c2.metric("📊 Pattern", quant['pattern'])
            c3.metric("📈 Pivot", f"{quant['pivot']:.2f}")

            st.markdown("---")
            st.subheader("🎯 Live Price Position on Gann Ladder")
            st.caption("0° = the opening price of the first 15-min candle. Every other degree is calculated from that anchor.")

            open_ref = opening_15m if opening_15m else quant['price']
            ladder = build_gann_ladder(quant, open_ref)
            band = locate_price_band(ladder, quant['price'])

            if band == "below":
                st.warning(f"⬇️ Price **{quant['price']:.2f}** is BELOW every calculated level (lowest = {ladder[0]['label']} @ {ladder[0]['value']:.2f}) — possible breakdown zone.")
            elif band == "above":
                st.warning(f"⬆️ Price **{quant['price']:.2f}** is ABOVE every calculated level (highest = {ladder[-1]['label']} @ {ladder[-1]['value']:.2f}) — possible breakout zone.")
            else:
                lo, hi = band
                st.success(f"💰 Price **{quant['price']:.2f}** is between **{lo['label']}** ({lo['value']:.2f}) and **{hi['label']}** ({hi['value']:.2f})")
                span = hi['value'] - lo['value']
                pct_into_band = ((quant['price'] - lo['value']) / span) if span else 0.0
                st.progress(min(max(pct_into_band, 0.0), 1.0),
                            text=f"{pct_into_band*100:.1f}% of the way from {lo['label']} to {hi['label']}")

            # Zoomed, decluttered chart: only the current band + one level either side
            if chart_df is not None and not chart_df.empty:
                zoom = get_chart_zoom_window(ladder, band, quant['price'], context=1)
                fig = go.Figure(data=[go.Candlestick(
                    x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
                    low=chart_df['Low'], close=chart_df['Close'], name=full_symbol)])
                if zoom:
                    y_min, y_max, visible_levels = zoom
                    for lvl in visible_levels:
                        color = "#26a69a" if lvl["side"] == "S" else ("#1e88e5" if lvl["side"] == "OPEN" else "#ef5350")
                        fig.add_hline(y=lvl["value"], line_dash="dash", line_width=1.6, line_color=color, opacity=0.9,
                                      annotation_text=f"{lvl['label']}  {lvl['value']:.2f}", annotation_position="left",
                                      annotation_font_size=12, annotation_bgcolor="rgba(255,255,255,0.75)")
                    if isinstance(band, tuple):
                        fig.add_hrect(y0=band[0]["value"], y1=band[1]["value"], fillcolor="#ffe08a", opacity=0.18, line_width=0)
                    fig.update_yaxes(range=[y_min, y_max])
                fig.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=100, t=30, b=10),
                                   title=f"{full_symbol} — zoomed to the current band (0° anchor = {open_ref:.2f})")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No candle data available to draw the chart right now.")

            # Smart-money zones: live order block + golden pocket, as requested
            st.markdown("---")
            st.subheader("🧠 Smart Money Zones")
            zc1, zc2 = st.columns(2)
            with zc1:
                st.markdown("**📦 Live Order Block**")
                ob = detect_order_block(chart_df)
                price = quant['price']
                if ob:
                    if ob.get("bullish"):
                        lo_ob, hi_ob = ob["bullish"]["low"], ob["bullish"]["high"]
                        inside = lo_ob <= price <= hi_ob
                        st.success(f"🟩 Bullish OB: {lo_ob:.2f} – {hi_ob:.2f}" + (" — price is inside it now" if inside else ""))
                    if ob.get("bearish"):
                        lo_ob, hi_ob = ob["bearish"]["low"], ob["bearish"]["high"]
                        inside = lo_ob <= price <= hi_ob
                        st.error(f"🟥 Bearish OB: {lo_ob:.2f} – {hi_ob:.2f}" + (" — price is inside it now" if inside else ""))
                else:
                    st.caption("No clear recent order block detected in this window.")
            with zc2:
                st.markdown("**🌀 Golden Pocket (61.8%–65% Fib)**")
                gp = detect_golden_pocket(chart_df)
                if gp:
                    price = quant['price']
                    inside = gp["gp_low"] <= price <= gp["gp_high"]
                    direction = "pullback in an uptrend" if gp["uptrend"] else "bounce in a downtrend"
                    msg = f"Zone: {gp['gp_low']:.2f} – {gp['gp_high']:.2f} ({direction})"
                    if inside:
                        st.success(f"✅ Price IS inside the golden pocket. {msg}")
                    else:
                        st.info(f"Price is outside it right now. {msg}")
                else:
                    st.caption("Not enough recent data to compute a golden pocket.")

            st.markdown("---")
            st.markdown("### 🎯 Traditional Pivot Levels")
            pivot_rows = []
            for level in ["4", "3", "2", "1"]:
                r_raw = quant.get(f'r{level}')
                s_raw = quant.get(f's{level}')
                try:
                    r_val = float(r_raw) if r_raw is not None else 0.0
                except (ValueError, TypeError):
                    r_val = 0.0
                try:
                    s_val = float(s_raw) if s_raw is not None else 0.0
                except (ValueError, TypeError):
                    s_val = 0.0
                pivot_rows.append({"Resistance": r_val, "Support": s_val})
            levels_df = pd.DataFrame(pivot_rows, index=["Level 4", "Level 3", "Level 2", "Level 1"])
            st.dataframe(levels_df, use_container_width=True)

        # ---------- AI SUGGESTIONS (right, beside live analysis) ----------
        with col_ai:
            st.subheader("💡 AI Trading Suggestions")
            news_text = "\n".join([n['title'] for n in news_list[:5]]) if news_list else "No news"
            if st.button("🚀 Generate AI Suggestions", use_container_width=True):
                with st.spinner("AI analyzing..."):
                    st.session_state["last_ai_suggestion"] = get_groq_suggestions(full_symbol, quant, news_text)
            last = st.session_state.get("last_ai_suggestion")
            if last:
                if last.startswith("⚠️"):
                    st.error(last)
                else:
                    st.markdown(last)
            else:
                st.caption("Click the button to get an INTRADAY + SWING read on this ticker.")

        # ---------- NEWS (full width, stock-specific) ----------
        st.markdown("---")
        st.subheader(f"📰 News on {full_symbol}")

        sentiment_ok = lambda n: (n["impact"] > 0 and "🟢 Bullish" in sentiment_filter) or (n["impact"] < 0 and "🔴 Bearish" in sentiment_filter) or (n["impact"] == 0 and "🟡 Neutral" in sentiment_filter)

        keywords = get_relevance_keywords(search_ticker, stock_info)
        stock_news = [n for n in filter_relevant_news(news_list, keywords, max_items=8) if sentiment_ok(n)]

        used_fallback = False
        if not stock_news:
            used_fallback = True
            stock_news = [n for n in news_list if sentiment_ok(n)][:5]

        if used_fallback:
            st.info(f"No headlines in the current feeds specifically mention {full_symbol} right now — showing top general market news instead.")

        if stock_news:
            for n in stock_news[:8]:
                badge = "🟢" if n['impact'] > 0 else ("🔴" if n['impact'] < 0 else "🟡")
                title_safe = html_module.escape(n['title'])
                summary_safe = html_module.escape(n.get('summary', ''))
                st.markdown(f"""
<div class="news-card">
  <div class="news-title">{badge} {title_safe}</div>
  <div class="news-summary">{summary_safe}</div>
  <div class="news-meta">📌 {html_module.escape(n['source'])} · 📅 {html_module.escape(n['time'])} · <a href="{n['link']}" target="_blank">Read full story →</a></div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No news matching your current sentiment filters.")


# ================== TAB 2: AI CHAT ==================
with tab2:
    st.header("💬 AI Chat Assistant")
    st.markdown("**Ask about:** Stock analysis, levels, news, strategy, market outlook")
    
    model = st.selectbox("Choose Model", list(GROQ_MODEL_MAP.keys()) + list(GEMINI_MODEL_MAP.keys()))
    
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = ""
    
    if st.session_state.quant_data:
        q = st.session_state.quant_data
        st.session_state.chat_context = f"Ticker: {full_symbol}, Price: {q['price']}, R1: {q['r1']}, S1: {q['s1']}, Pattern: {q['pattern']}"
    
    if not st.session_state.chat_context:
        st.warning("⚠️ Analysis data not ready. Please wait for Tab 1 to load.")
    else:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about stock, levels, news, strategy..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(f"AI thinking... ({model})"):
                    response = get_chat_response(prompt, st.session_state.chat_context, model)
                    if response and not response.startswith("⚠️"):
                        st.markdown(response)
                    else:
                        st.error(response or "❌ AI Error - Try a different model or check your API keys")
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()


# ================== TAB 3: MARKET HEATMAP ==================
with tab3:
    subtab1, subtab2 = st.tabs(["📊 Sector Performance", "🔥 Stock Heatmap"])
    with subtab1:
        st.header("📊 Sector Performance Heatmap")
        sector_indices = {
            "India - NSE (.NS)": {"NIFTY Bank": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY Auto": "^CNXAUTO", "NIFTY FMCG": "^CNXFMCG", "NIFTY Pharma": "^CNXPHARMA", "NIFTY Metal": "^CNXMETAL"},
            "United States (No Suffix)": {"S&P 500 Tech": "XLK", "S&P 500 Bank": "XLF", "S&P 500 Health": "XLV", "S&P 500 Consumer": "XLP", "S&P 500 Energy": "XLE"},
            "Global Indices": {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "DAX": "^GDAXI", "FTSE": "^FTSE", "Nikkei": "^N225"},
            "Commodities": {"Gold": "GC=F", "Silver": "SI=F", "Oil": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F"},
            "Cryptocurrency (-USD)": {"BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD", "SOL": "SOL-USD", "XRP": "XRP-USD"}
        }
        sectors = sector_indices.get(select_market, {"General": "^GSPC"})
        sector_data = []
        for sector_name, symbol in sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                change = info.get('regularMarketChangePercent', 0)
                intensity = min(10, abs(change) / 0.5)
                color_bars = "🟢" * int(intensity) if change > 0 else "🔴" * int(intensity)
                sector_data.append({"Sector": sector_name, "Price": price, "Change %": change, "Intensity": f"{intensity:.1f}/10", "Visual": color_bars if color_bars else "⚪"})
            except: pass
        if sector_data:
            df_sectors = pd.DataFrame(sector_data).sort_values("Change %", ascending=False)
            for idx, row in df_sectors.iterrows():
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1: st.markdown(f"### {row['Visual']} {row['Sector']}")
                with col2: st.metric("Price", f"${row['Price']:.2f}")
                with col3: st.metric("Change", f"{row['Change %']:.2f}%", f"{row['Change %']:.2f}%")
                st.markdown("---")
        else:
            st.warning("Could not fetch sector data")
    with subtab2:
        st.header("🔥 Stock Heatmap - Intensity View")
        sector_stocks = {
            "India - NSE (.NS)": {
                "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
                "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
                "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
                "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
                "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS"],
                "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
                "Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "ADANIENT.NS"]
            },
            "United States (No Suffix)": {
                "Tech Giants": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN"],
                "Banking": ["JPM", "BAC", "WFC", "GS", "MS"],
                "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
                "Consumer": ["WMT", "PG", "KO", "PEP", "COST"],
                "Auto": ["TSLA", "F", "GM", "RIVN"]
            },
            "Cryptocurrency (-USD)": {
                "Top 10": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "TRX-USD", "DOT-USD"],
                "DeFi": ["UNI-USD", "LINK-USD", "AAVE-USD", "MKR-USD", "COMP-USD"],
                "Layer 2": ["MATIC-USD", "ARB-USD", "OP-USD", "LRC-USD"]
            },
            "Commodities": {
                "Metals": ["GC=F", "SI=F", "PL=F", "PA=F"],
                "Energy": ["CL=F", "NG=F", "HO=F", "RB=F"],
                "Agriculture": ["ZC=F", "ZW=F", "ZS=F", "KC=F"]
            },
            "Forex Currency (=X)": {
                "Major Pairs": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"],
                "Commodity Pairs": ["AUDUSD=X", "USDCAD=X", "NZDUSD=X"],
                "Emerging Markets": ["USDINR=X", "USDBRL=X", "USDMXN=X"]
            }
        }
        market_sectors = sector_stocks.get(select_market, sector_stocks.get("India - NSE (.NS)", {}))
        selected_sector = st.selectbox("Select Sector", list(market_sectors.keys()))
        stocks_list = market_sectors[selected_sector]
        heatmap_data = []
        for stock in stocks_list:
            try:
                ticker = yf.Ticker(stock)
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                change = info.get('regularMarketChangePercent', 0)
                intensity = min(10, abs(change) / 0.3)
                if change > 0:
                    color_bars = "🟢" * int(intensity) + "⚫" * (10 - int(intensity))
                elif change < 0:
                    color_bars = "🔴" * int(intensity) + "⚫" * (10 - int(intensity))
                else:
                    color_bars = "⚪" * 10
                heatmap_data.append({"Stock": stock.replace(".NS", "").replace("-USD", "").replace("=F", "").replace("=X", ""), "Price": price, "Change %": change, "Intensity": f"{intensity:.1f}/10", "Color Bars": color_bars})
            except: pass
        if heatmap_data:
            df_heatmap = pd.DataFrame(heatmap_data).sort_values("Change %", ascending=False)
            for idx, row in df_heatmap.iterrows():
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1: st.markdown(f"### {row['Color Bars']} {row['Stock']}")
                with col2: st.metric("Price", f"${row['Price']:.2f}")
                with col3: st.metric("Change", f"{row['Change %']:.2f}%", f"{row['Change %']:.2f}%")
                st.markdown("---")
        else:
            st.warning("Could not fetch stock data")


# ================== TAB 4: WATCHLIST & PORTFOLIO ==================
with tab4:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📋 Watchlist Overview")
        if not st.session_state.watchlists:
            st.warning("No watchlists")
        else:
            for list_name, stocks in st.session_state.watchlists.items():
                st.markdown(f"### 📋 {list_name}")
                if stocks:
                    cols = st.columns(min(len(stocks), 3))
                    for idx, stock in enumerate(stocks):
                        try:
                            ticker = yf.Ticker(stock)
                            info = ticker.info
                            price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                            change = info.get('regularMarketChangePercent', 0)
                            with cols[idx % len(cols)]:
                                st.metric(stock, f"{price:.2f}" if price else "N/A", f"{change:.2f}%" if change else "0.00%")
                        except:
                            with cols[idx % len(cols)]: st.metric(stock, "N/A", "0.00%")
                st.markdown("---")
    with col2:
        st.header("💼 Portfolio Tracker")
        st.subheader("➕ Add Holding")
        col_a, col_b, col_c = st.columns(3)
        with col_a: portfolio_stock = st.text_input("Ticker", key="pf_stock")
        with col_b: portfolio_qty = st.number_input("Qty", min_value=1, value=10, key="pf_qty")
        with col_c: portfolio_buy_price = st.number_input("Price", min_value=0.01, value=100.0, key="pf_price")
        if st.button("Add to Portfolio"):
            if portfolio_stock:
                st.session_state.portfolio[portfolio_stock] = {"qty": portfolio_qty, "buy_price": portfolio_buy_price}
                st.rerun()
        if st.session_state.portfolio:
            st.subheader("📊 Your Holdings")
            total_invested, total_current = 0, 0
            for stock, data in st.session_state.portfolio.items():
                try:
                    ticker = yf.Ticker(stock)
                    current_price = ticker.info.get('currentPrice') or 0
                    invested = data['qty'] * data['buy_price']
                    current = data['qty'] * current_price
                    pnl = current - invested
                    pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
                    total_invested += invested
                    total_current += current
                    st.markdown(f"### {stock}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Qty", data['qty'])
                    c2.metric("Buy", f"${data['buy_price']:.2f}")
                    c3.metric("Current", f"${current_price:.2f}")
                    c4.metric("P&L", f"${pnl:.2f}", f"{pnl_pct:.2f}%")
                    if st.button(f"🗑️", key=f"remove_pf_{stock}"):
                        del st.session_state.portfolio[stock]
                        st.rerun()
                except: st.error(f"Could not fetch {stock}")
            total_pnl = total_current - total_invested
            total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
            st.markdown("### 💰 Total Portfolio")
            c1, c2, c3 = st.columns(3)
            c1.metric("Invested", f"${total_invested:.2f}")
            c2.metric("Current", f"${total_current:.2f}")
            c3.metric("P&L", f"${total_pnl:.2f}", f"{total_pnl_pct:.2f}%")


# ================== TAB 5: ALERTS & BACKTEST ==================
with tab5:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("🔔 Price Alerts")
        col_a, col_b, col_c = st.columns(3)
        with col_a: alert_stock = st.text_input("Ticker", key="alert_stock")
        with col_b: alert_price = st.number_input("Price", min_value=0.01, key="alert_price")
        with col_c: alert_type = st.selectbox("Type", ["Above", "Below"], key="alert_type")
        if st.button("Create Alert"):
            if alert_stock:
                st.session_state.alerts.append({"stock": alert_stock, "price": alert_price, "type": alert_type, "active": True})
                st.rerun()
        if st.session_state.alerts:
            st.subheader("📋 Active Alerts")
            for idx, alert in enumerate(st.session_state.alerts):
                if alert["active"]:
                    try:
                        ticker = yf.Ticker(alert["stock"])
                        current = ticker.info.get('currentPrice') or 0
                        triggered = (alert["type"] == "Above" and current >= alert["price"]) or (alert["type"] == "Below" and current <= alert["price"])
                        if triggered:
                            st.error(f"🚨 {alert['stock']} {alert['type']} ${alert['price']}! Current: ${current:.2f}")
                            if st.button(f"✅ Ack", key=f"ack_{idx}"):
                                st.session_state.alerts[idx]["active"] = False
                        else:
                            st.info(f"⏰ {alert['stock']} {alert['type']} ${alert['price']} | Current: ${current:.2f}")
                            if st.button(f"🗑️", key=f"del_alert_{idx}"):
                                st.session_state.alerts.pop(idx)
                                st.rerun()
                    except: st.warning(f"Could not fetch {alert['stock']}")
    with col2:
        st.header("🧪 Backtesting")
        col_a, col_b = st.columns(2)
        with col_a: bt_symbol = st.text_input("Ticker", value="RELIANCE.NS")
        with col_b: strategy = st.selectbox("Strategy", ["Buy at S1", "Sell at R1"])
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=90))
        end_date = st.date_input("End", value=datetime.now())
        if st.button("🚀 Run Backtest"):
            with st.spinner(f"Testing {strategy}..."):
                result = run_backtest(bt_symbol, start_date, end_date, strategy)
                if result:
                    st.success("✅ Complete!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Trades", result['total_trades'])
                    c2.metric("Win %", f"{result['win_rate']:.1f}%")
                    c3.metric("Avg $", f"${result['avg_profit']:.2f}")
                    c4.metric("Total $", f"${result['total_profit']:.2f}")
                else: st.error("Failed")


# ================== TAB 6: HISTORICAL DATA ==================
with tab6:
    st.header("📅 Historical 15m Data (Last 2 Days)")
    historical = get_cached_historical(full_symbol, 2)
    if not historical:
        st.warning("No historical data")
    else:
        for day in historical:
            st.markdown(f"### 📅 {day['Date']}")
            st.metric("Opening Price", f"${day['Open']:.2f}")
            st.markdown("#### 🎯 Gann Angle Levels")
            gann_rows = []
            for deg in ["22.5°", "45°", "67.5°", "90°", "180°"]:
                r_key = f'R_{deg.replace("°", "")}'
                s_key = f'S_{deg.replace("°", "")}'
                r_raw = day.get(r_key)
                s_raw = day.get(s_key)
                try:
                    r_val = float(r_raw) if r_raw is not None else 0.0
                except (ValueError, TypeError):
                    r_val = 0.0
                try:
                    s_val = float(s_raw) if s_raw is not None else 0.0
                except (ValueError, TypeError):
                    s_val = 0.0
                gann_rows.append({"Angle": deg, "Resistance": r_val, "Support": s_val})
            gann_hist_df = pd.DataFrame(gann_rows)
            st.dataframe(gann_hist_df, use_container_width=True)
            st.markdown("#### 🎯 Traditional Pivot Levels")
            pivot_rows = []
            for level in ["4", "3", "2", "1"]:
                r_raw = day.get(f'R{level}')
                s_raw = day.get(f'S{level}')
                try:
                    r_val = float(r_raw) if r_raw is not None else 0.0
                except (ValueError, TypeError):
                    r_val = 0.0
                try:
                    s_val = float(s_raw) if s_raw is not None else 0.0
                except (ValueError, TypeError):
                    s_val = 0.0
                pivot_rows.append({"Resistance": r_val, "Support": s_val})
            pivot_hist_df = pd.DataFrame(pivot_rows, index=["Level 4", "Level 3", "Level 2", "Level 1"])
            st.dataframe(pivot_hist_df, use_container_width=True)
            st.markdown("#### 📊 Day Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Open", f"${day['Open']:.2f}")
            c2.metric("High", f"${day['High']:.2f}")
            c3.metric("Low", f"${day['Low']:.2f}")
            c4.metric("Close", f"${day['Close']:.2f}")
            st.markdown("---")
