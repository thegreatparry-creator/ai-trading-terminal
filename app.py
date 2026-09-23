import email.utils
import html as html_module
import math
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from groq import Groq
from google import genai as google_genai
st.set_page_config(page_title="AI Institutional Terminal", layout="wide")


# ================== SECRETS ==================
def secret(name):
    try:
        configured = st.secrets.get(name, "")
        return configured or os.getenv(name, "")
    except Exception:
        return os.getenv(name, "")


GROQ_KEY = secret("# 1. Install Required Libraries
!pip install -q -U google-genai groq yfinance pandas numpy

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import email.utils
import math
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from google import genai
from google.genai.errors import ServerError
from groq import Groq
import numpy as np
import pandas as pd
import yfinance as yf

# 2. Configure Dual Free AI Keys
GEMINI_KEY = "AQ.Ab8RN6JtGdVf9VFtpeo2_7BYDuZQZZlhMIbQxKFX1noZ4UnTSQ"
GROQ_KEY = "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"

gemini_client = genai.Client(api_key=GEMINI_KEY)
groq_client = Groq(api_key=GROQ_KEY)


def call_gemini(prompt):
    for model in ["gemini-3.6-flash", "gemini-3.5-flash-lite"]:
        try:
            chat = gemini_client.chats.create(model=model)
            return chat.send_message(prompt).text
        except Exception:
            time.sleep(0.5)
    return None


def call_groq(prompt):
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return res.choices[0].message.content
    except Exception:
        return None


def run_hybrid_agent(preferred_engine, role, task):
    prompt = f"ROLE: {role}\nTASK: {task}\nProvide a concise, ultra-sharp bulleted response."
    if preferred_engine == "groq":
        out = call_groq(prompt) or call_gemini(prompt)
    else:
        out = call_gemini(prompt) or call_groq(prompt)
    return out if out else "Analysis unavailable."


def shorten_link(long_url):
    try:
        if len(long_url) < 45:
            return long_url
        api_url = (
            "http://tinyurl.com/api-create.php?"
            + urllib.parse.urlencode({"url": long_url})
        )
        req = urllib.request.Request(
            api_url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return long_url[:40] + "..."


# ======================================================================
# 1. PRE-MARKET & REAL-TIME CANDLESTICK PATTERN ENGINE
# ======================================================================
def detect_candle_patterns_and_structure(df):
    """Scans OHLCV price action for classic institutional candlestick patterns."""
    if len(df) < 3:
        return "Insufficient candle history."

    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values

    curr_body = abs(c[-1] - o[-1])
    curr_range = h[-1] - l[-1] if h[-1] != l[-1] else 0.001
    prev_body = abs(c[-2] - o[-2])

    patterns = []

    # 1. Bullish & Bearish Engulfing
    if (
        c[-1] > o[-1]
        and c[-2] < o[-2]
        and c[-1] >= o[-2]
        and o[-1] <= c[-2]
        and curr_body > prev_body
    ):
        patterns.append("🔥 Bullish Engulfing (Reversal Signal)")
    elif (
        c[-1] < o[-1]
        and c[-2] > o[-2]
        and o[-1] >= c[-2]
        and c[-1] <= o[-2]
        and curr_body > prev_body
    ):
        patterns.append("⚠️ Bearish Engulfing (Reversal Signal)")

    # 2. Hammer & Shooting Star
    lower_wick = min(o[-1], c[-1]) - l[-1]
    upper_wick = h[-1] - max(o[-1], c[-1])
    if lower_wick >= 2 * curr_body and upper_wick <= 0.2 * curr_body:
        patterns.append("🔨 Bullish Hammer / Pinbar (Buyer Defense at Lows)")
    elif upper_wick >= 2 * curr_body and lower_wick <= 0.2 * curr_body:
        patterns.append("🌠 Bearish Shooting Star (Seller Defense at Highs)")

    # 3. Inside Bar Compression
    if h[-1] < h[-2] and l[-1] > l[-2]:
        patterns.append("⏳ Inside Bar Compression (Volatility Squeeze Imminent)")

    return " | ".join(patterns) if patterns else "Standard Price Action Candle"


def calculate_pinescript_quant_signals(ticker_symbol):
    """Calculates Pre-Market Gap, Gann Square of 9, Dynamic EMAs, Golden Pocket & Patterns."""
    try:
        stock = yf.Ticker(ticker_symbol)

        # Download Intraday Data INCLUDING Pre-Market / After-Hours
        df_15m = stock.history(period="5d", interval="15m", prepost=True)
        df_daily = stock.history(period="3mo", interval="1d")

        if df_15m.empty:
            df_15m = df_daily

        last_close = float(df_15m["Close"].iloc[-1])

        # --- Pre-Market Opening Structure ---
        prev_day_close = (
            float(df_daily["Close"].iloc[-2])
            if len(df_daily) >= 2
            else last_close
        )
        pre_market_high = float(df_15m["High"].iloc[-10:].max())
        pre_market_low = float(df_15m["Low"].iloc[-10:].min())
        gap_pct = round(
            ((last_close - prev_day_close) / prev_day_close) * 100, 2
        )
        gap_status = (
            f"Gap Up (+{gap_pct}%)"
            if gap_pct > 0.3
            else (f"Gap Down ({gap_pct}%)" if gap_pct < -0.3 else "Flat Open")
        )

        # --- Candlestick Pattern Detection ---
        candle_pattern = detect_candle_patterns_and_structure(df_15m)

        # --- Gann Square of 9 ---
        anchor_price = float(df_15m["Close"].iloc[-min(25, len(df_15m))])
        anchor_sqrt = math.sqrt(anchor_price)

        def calc_sq9(root, deg, is_r):
            delta = deg / 180.0
            return (
                math.pow(root + delta, 2)
                if is_r
                else math.pow(max(0.1, root - delta), 2)
            )

        r225, r450, r675, r900, r180 = (
            calc_sq9(anchor_sqrt, d, True)
            for d in [22.5, 45.0, 67.5, 90.0, 180.0]
        )
        s225, s450, s675, s900, s180 = (
            calc_sq9(anchor_sqrt, d, False)
            for d in [22.5, 45.0, 67.5, 90.0, 180.0]
        )

        # --- Dynamic EMAs ---
        ema20 = df_15m["Close"].ewm(span=20).mean().iloc[-1]
        ema50 = df_15m["Close"].ewm(span=50).mean().iloc[-1]
        trend_up = ema20 >= ema50
        fast_src = df_15m["Low"] if trend_up else df_15m["High"]
        slow_src = df_15m["Low"] if trend_up else df_15m["High"]
        fast_ema = fast_src.ewm(span=9).mean().iloc[-1]
        slow_ema = slow_src.ewm(span=50).mean().iloc[-1]
        ema_bias = (
            "BULLISH (Fast > Slow EMA)"
            if fast_ema > slow_ema
            else "BEARISH (Fast < Slow EMA)"
        )

        # --- Wave Target & Golden Pocket ---
        highs = df_15m["High"].values
        lows = df_15m["Low"].values
        pA = np.min(lows[-15:])
        pB = np.max(highs[-15:])
        pC = lows[-1]
        wave_target = (pB * pC) / pA if pA > 0 else last_close * 1.03
        fib_500 = pB - (0.500 * (pB - pA))
        fib_618 = pB - (0.618 * (pB - pA))

        # --- Candle 3-5-9 Streak ---
        is_green = df_15m["Close"] > df_15m["Open"]
        green_streak = (
            int(is_green.iloc[::-1].cumprod().sum())
            if is_green.iloc[-1]
            else 0
        )
        red_streak = (
            int((~is_green).iloc[::-1].cumprod().sum())
            if not is_green.iloc[-1]
            else 0
        )
        streak_signal = (
            f"Exhaustion Warning ({green_streak} green bars)"
            if green_streak in [3, 5, 9]
            else (
                f"Exhaustion Warning ({red_streak} red bars)"
                if red_streak in [3, 5, 9]
                else "Normal Progression"
            )
        )

        quant_summary = f"""
        === 📊 REAL-TIME CANDLES & PRE-MARKET STRUCTURE: {ticker_symbol} ===
        - Live Close: {round(last_close, 2)}
        - Pre-Market Gap Structure: {gap_status} (Prev Close: {round(prev_day_close, 2)})
        - Pre-Market High (PMH): {round(pre_market_high, 2)} | Pre-Market Low (PML): {round(pre_market_low, 2)}
        - Live Candle Pattern Detected: {candle_pattern}
        - Gann Anchor (1st 15m): {round(anchor_price, 2)} (sqrt: {round(anchor_sqrt, 4)})
        - Gann Resistances (R): 22.5°: {round(r225, 2)} | 45°: {round(r450, 2)} | 90°: {round(r900, 2)} | 180°: {round(r180, 2)}
        - Gann Supports (S): 22.5°: {round(s225, 2)} | 45°: {round(s450, 2)} | 90°: {round(s900, 2)} | 180°: {round(s180, 2)}
        - Dynamic EMAs: Fast(9)={round(fast_ema, 2)} vs Slow(50)={round(slow_ema, 2)} ({ema_bias})
        - Golden Pocket: {round(fib_618, 2)} - {round(fib_500, 2)} | Wave Target ((B*C)/A): {round(wave_target, 2)}
        - Candle Exhaustion: {streak_signal}
        """
        return quant_summary, last_close
    except Exception as e:
        return f"Quant computation fallback: {str(e)}", 0.0


# ======================================================================
# 2. MULTI-SOURCE GLOBAL NEWS WIRE HARVESTER
# ======================================================================
def fetch_multi_source_global_news(ticker_symbol):
    clean_keyword = (
        ticker_symbol.replace(".NS", "")
        .replace(".BO", "")
        .replace(".L", "")
        .replace(".DE", "")
        .replace("-USD", "")
        .replace("=F", "")
    )

    queries = [
        f"{clean_keyword} stock site:moneycontrol.com OR site:economictimes.indiatimes.com",
        f"{clean_keyword} site:investing.com OR site:tradingview.com",
        f"{clean_keyword} stock breaking news financial",
    ]

    collected_articles = []
    seen_titles = set()

    for q in queries:
        try:
            encoded_query = urllib.parse.quote(q)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            req = urllib.request.Request(
                rss_url, headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(req, timeout=4) as response:
                root = ET.fromstring(response.read())
                for item in root.findall(".//item")[:4]:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub_date = item.findtext("pubDate", "").strip()
                    source = (
                        item.find("source").text.strip()
                        if item.find("source") is not None
                        else "Financial Wire"
                    )

                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        short_url = shorten_link(link)
                        collected_articles.append(
                            {
                                "title": title,
                                "link": short_url,
                                "source": source,
                                "time": pub_date,
                            }
                        )
        except Exception:
            pass

    try:
        stock = yf.Ticker(ticker_symbol)
        for n in (stock.news or [])[:2]:
            title = n.get("title") or (
                n.get("content", {}).get("title")
                if isinstance(n.get("content"), dict)
                else None
            )
            link = (
                n.get("link")
                or n.get("clickThroughUrl", {}).get("url")
                or (
                    n.get("content", {}).get("canonicalUrl", {})
                    if isinstance(n.get("content"), dict)
                    else None
                )
            )
            publisher = n.get("publisher") or (
                n.get("content", {}).get("provider", {}).get("displayName")
                if isinstance(n.get("content"), dict)
                else "Yahoo Wire"
            )

            if title and title not in seen_titles:
                seen_titles.add(title)
                short_url = shorten_link(link or "https://finance.yahoo.com")
                collected_articles.append(
                    {
                        "title": title,
                        "link": short_url,
                        "source": publisher,
                        "time": "Recent",
                    }
                )
    except Exception:
        pass

    breaking_news, recent_news = [], []
    for art in collected_articles:
        formatted_entry = f"• [{art['source']}] {art['title']} ({art['time']})\n  Link: {art['link']}"
        is_breaking = False
        try:
            parsed_time = email.utils.parsedate_to_datetime(art["time"])
            if (
                datetime.now(parsed_time.tzinfo) - parsed_time
            ).total_seconds() <= 86400:
                is_breaking = True
        except Exception:
            if (
                "hours ago" in art["time"].lower()
                or "mins ago" in art["time"].lower()
            ):
                is_breaking = True

        if is_breaking:
            breaking_news.append(formatted_entry)
        else:
            recent_news.append(formatted_entry)

    breaking_text = (
        "\n".join(breaking_news)
        if breaking_news
        else "• No urgent breaking headlines in the past 24 hours."
    )
    recent_text = (
        "\n".join(recent_news[:3])
        if recent_news
        else "• No recent news context available."
    )

    return f"""
    --- ⚡ BREAKING NEWS (LAST 24H) ---
    {breaking_text}

    --- 📅 RECENT NEWS (PAST 7-30D) ---
    {recent_text}
    """


def fetch_deep_market_context(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info or {}
        pe = info.get("trailingPE", "N/A")
        fwd_pe = info.get("forwardPE", "N/A")
        margin = info.get("profitMargins", "N/A")
        growth = info.get("revenueGrowth", "N/A")
        short_float = info.get("shortPercentOfFloat", "N/A")

        multi_wire_news = fetch_multi_source_global_news(ticker_symbol)

        dataset = f"""
        VALUATION: P/E: {pe} | Forward P/E: {fwd_pe} | Margin: {margin} | Growth: {growth} | Short Float: {short_float}
        {multi_wire_news}
        """
        return dataset, multi_wire_news
    except Exception as e:
        return f"Context error: {str(e)}", "No news feeds available."


# ======================================================================
# 3. DUAL-AI QUANT & NEWS DESK EXECUTION
# ======================================================================
def execute_trading_desk_master(ticker_symbol):
    start_time = time.time()
    ticker = ticker_symbol.split(" ")[0].strip().upper()

    print("=" * 65)
    print(f"🚀 RUNNING REAL-TIME CANDLES & DUAL-AI DESK FOR: {ticker}")
    print("=" * 65)

    quant_hud, last_price = calculate_pinescript_quant_signals(ticker)
    market_context, news_dossier = fetch_deep_market_context(ticker)

    tasks = {
        "technicals_and_pinescript": (
            "groq",
            "PineScript & Candle Pattern Quant Specialist",
            f"Analyze candle patterns, Pre-Market Gap, Gann Square of 9 levels, Dynamic EMAs, and Wave Target for {ticker}:\n{quant_hud}",
        ),
        "fundamentals": (
            "gemini",
            "Equity Fundamental Analyst",
            f"Audit P/E ratios and profit margins for {ticker}:\n{market_context}",
        ),
        "sentiment": (
            "gemini",
            "Market Sentiment & Breaking News Analyst",
            f"Compare 24h breaking news vs 30-day sentiment momentum for {ticker}:\n{news_dossier}",
        ),
        "geopolitical": (
            "gemini",
            "Geopolitical Risk Analyst",
            f"Audit trade barriers and regulatory risks for {ticker}.",
        ),
        "forensics": (
            "gemini",
            "Financial Forensics Analyst",
            f"Check accounting health and debt maturity for {ticker}.",
        ),
        "shorts": (
            "groq",
            "Short-Seller & Insider Analyst",
            f"Examine short float squeeze dynamics for {ticker}.",
        ),
        "earnings": (
            "groq",
            "Pre-Earnings Analyst",
            f"Check earnings estimate revisions for {ticker}.",
        ),
        "scenarios": (
            "groq",
            "Scenario Modeler",
            f"Model breakout targets vs breakdown support levels for {ticker}.",
        ),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_agent = {
            executor.submit(run_hybrid_agent, engine, role, task): key
            for key, (engine, role, task) in tasks.items()
        }
        for future in future_to_agent:
            key = future_to_agent[future]
            results[key] = future.result()

    chief_trader_prompt = f"""
    You are a 20+ Year Veteran Institutional Trader.
    Synthesize the Real-Time Candle Patterns, Pre-Market Structure, PineScript Quant Calculations, and 8 Specialist Reports from Gemini and Groq for {ticker}:

    --- CANDLE PATTERNS & QUANT HUD ---
    {quant_hud}

    --- VALUATION & NEWS ---
    {market_context}

    --- SPECIALIST INTELLIGENCE ---
    - PINESCRIPT & CANDLE PATTERNS (Groq): {results['technicals_and_pinescript']}
    - FUNDAMENTALS (Gemini): {results['fundamentals']}
    - SENTIMENT (Gemini): {results['sentiment']}
    - GEOPOLITICS (Gemini): {results['geopolitical']}
    - FORENSICS (Gemini): {results['forensics']}
    - SHORTS (Groq): {results['shorts']}
    - EARNINGS (Groq): {results['earnings']}
    - SCENARIOS (Groq): {results['scenarios']}

    Deliver a sharp, clean, institutional briefing with concise bullet points:

    ======================================================================
    🎯 1. INTRADAY TRADING ACTION (15m / Scalp Horizon)
    ======================================================================
    • ACTION BUTTON: [🟢 BUY NOW / 🔴 SELL / SHORT / 🟡 WAIT FOR LEVEL]
    • CANDLE & PRE-MARKET SETUP: [Gap structure & detected candlestick pattern status]
    • ENTRY ZONE: [Exact price range based on PMH/PML and Gann Support]
    • STOP LOSS: [Exact invalidation price based on Gann 22.5°/45°]
    • TARGET 1 (Conservative - 45°/90°): [Price]
    • TARGET 2 (Aggressive - Wave Target): [Price]
    • RISK / REWARD: [e.g., 1:2.5]
    • WHY (1-2 lines): [Why enter here based on pattern and Gann support]

    ======================================================================
    📈 2. SWING TRADE ACTION (Multi-Day to Multi-Week)
    ======================================================================
    • ACTION BUTTON: [🟢 ACCUMULATE / 🔴 EXIT / 🟡 HOLD]
    • BUY ZONE: [Golden Pocket price range]
    • SWING STOP LOSS: [180° Major Gann floor]
    • SWING TARGETS: [Target 1 / Target 2 / Target 3]
    • TIMEFRAME: [e.g., 5-15 Days]
    • WHY (1-2 lines): [Macro and fundamental thesis]

    ======================================================================
    📰 3. NEWS MOMENTUM & SHORT LINKS
    ======================================================================
    • News Verdict (1 line): [Is sentiment accelerating or decelerating?]
{news_dossier}

    ======================================================================
    🔍 4. TRADE CONVICTION SCORE
    ======================================================================
    • Conviction Rating: [Score: 1 - 10]
    • Model Consensus: [Brief confirmation: Gemini + Groq agreement]
    """

    final_action_plan = run_hybrid_agent(
        "groq",
        "Chief Trade Execution Officer (20-Year Veteran)",
        chief_trader_prompt,
    )
    elapsed = round(time.time() - start_time, 2)

    print("\n" + "=" * 65)
    print(f"INSTITUTIONAL TRADE BRIEF: {ticker} (Generated in {elapsed}s)")
    print("=" * 65)
    print(final_action_plan)


print(" Real-Time Candle & Pre-Market Engine Configured!")")
GEMINI_KEY = secret("GEMINI_KEY")
TELEGRAM_BOT_TOKEN = secret("8794257218:AAGYGDqPUEJdI3UahL07Pe86IgcLCfIn20g")
TELEGRAM_CHAT_ID = secret("8600332637")

groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
# google-generativeai (old SDK) is deprecated; using the current google-genai SDK.
gemini_client = google_genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

GROQ_MODELS = {
    "Groq GPT-OSS 20B (fast)": "openai/gpt-oss-20b",
    "Groq GPT-OSS 120B (smartest)": "openai/gpt-oss-120b",
    "Groq Qwen3 32B": "qwen/qwen3-32b",
    "Groq Kimi K2": "moonshotai/kimi-k2-instruct",
}
GEMINI_MODELS = {"Gemini 3.5 Flash": "gemini-3.5-flash", "Gemini 3.1 Pro (Preview)": "gemini-3.1-pro-preview"}

NEWS_SOURCES = [
    ("Moneycontrol", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1998028306.cms"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline"),
    ("Alpha Ideas", "https://alphaideas.in/feed"),
    ("Cointelegraph", "https://cointelegraph.com/feed"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss"),
    ("ForexLive", "https://www.forexlive.com/servicexml/xml.aspx?xml=1"),
]

SECTOR_INDICES = {
    "India - NSE": {"Nifty Bank": "^NSEBANK", "Nifty IT": "^CNXIT", "Nifty Auto": "^CNXAUTO", "Nifty FMCG": "^CNXFMCG", "Nifty Pharma": "^CNXPHARMA", "Nifty Metal": "^CNXMETAL"},
    "United States": {"Tech (XLK)": "XLK", "Financials (XLF)": "XLF", "Healthcare (XLV)": "XLV", "Consumer (XLP)": "XLP", "Energy (XLE)": "XLE"},
    "Global Indices": {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "DAX": "^GDAXI", "FTSE": "^FTSE", "Nikkei": "^N225"},
    "Commodities": {"Gold": "GC=F", "Silver": "SI=F", "Oil": "CL=F", "Natural Gas": "NG=F", "Copper": "HG=F"},
    "Cryptocurrency": {"BTC": "BTC-USD", "ETH": "ETH-USD", "BNB": "BNB-USD", "SOL": "SOL-USD", "XRP": "XRP-USD"},
}
SECTOR_STOCKS = {
    "India - NSE": {
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
        "FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
        "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
        "Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "ADANIENT.NS"],
    },
    "United States": {
        "Tech Giants": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN"],
        "Banking": ["JPM", "BAC", "WFC", "GS", "MS"],
        "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
        "Consumer": ["WMT", "PG", "KO", "PEP", "COST"],
        "Auto": ["TSLA", "F", "GM", "RIVN"],
    },
    "Cryptocurrency": {
        "Top 10": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "TRX-USD", "DOT-USD"],
        "DeFi": ["UNI-USD", "LINK-USD", "AAVE-USD", "MKR-USD", "COMP-USD"],
        "Layer 2": ["MATIC-USD", "ARB-USD", "OP-USD", "LRC-USD"],
    },
    "Commodities": {
        "Metals": ["GC=F", "SI=F", "PL=F", "PA=F"],
        "Energy": ["CL=F", "NG=F", "HO=F", "RB=F"],
        "Agriculture": ["ZC=F", "ZW=F", "ZS=F", "KC=F"],
    },
    "Global Indices": {"Major": ["^GSPC", "^IXIC", "^DJI", "^GDAXI", "^FTSE", "^N225"]},
}

# Curated aliases for tickers whose Yahoo symbol never appears in a headline (indices,
# commodities, forex, crypto) — the main reason "news wasn't changing per stock".
SYMBOL_ALIASES = {
    "^NSEI": ["nifty", "nifty 50", "nifty50"], "^NSEBANK": ["bank nifty", "nifty bank"], "^BSESN": ["sensex"],
    "^GSPC": ["s&p 500", "s&p500", "sp500"], "^DJI": ["dow jones", "dow"], "^IXIC": ["nasdaq"],
    "^N225": ["nikkei"], "^FTSE": ["ftse"], "^GDAXI": ["dax"], "GC=F": ["gold"], "SI=F": ["silver"],
    "CL=F": ["crude oil", "oil prices", "wti"], "NG=F": ["natural gas"], "BTC-USD": ["bitcoin", "btc"],
    "ETH-USD": ["ethereum", "eth"], "EURUSD=X": ["euro", "eur/usd"], "GBPUSD=X": ["pound", "sterling", "gbp/usd"],
    "USDJPY=X": ["yen", "usd/jpy"], "USDINR=X": ["rupee", "usd/inr"],
}

st.markdown("""
<style>
html,body,[class*="css"]{font-size:16px}
h1{font-size:1.6rem!important}h2{font-size:1.25rem!important}h3{font-size:1.1rem!important}
div[data-testid="stMetricValue"]{font-size:1.25rem!important}
.stButton>button{border-radius:8px;padding:.5rem 1rem;font-size:1rem}
.news-card{border:1px solid rgba(150,150,150,.28);border-radius:14px;padding:15px 16px;margin:0 0 13px;background:rgba(255,255,255,.025)}
.news-card.direct{border-left:5px solid #26a269}.news-card.indirect{border-left:5px solid #e8a317}
.news-head{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}
.news-title{font-size:1.05rem;font-weight:700;line-height:1.4;margin:7px 0}
.news-summary,.news-reason{line-height:1.55;opacity:.88}
.news-meta{font-size:.78rem;opacity:.68;margin-top:9px}
.badge{padding:5px 10px;border-radius:99px;font-size:.75rem;font-weight:700;white-space:nowrap}
.green{background:#164d35;color:#6ee7a0}.yellow{background:#604811;color:#ffd166}.red{background:#5b2020;color:#ff8b8b}
.zone{border:1px solid rgba(140,140,140,.3);border-left:5px solid #999;border-radius:15px;padding:15px;margin:10px 0}
.zone.bullish{border-left-color:#26a269}.zone.bearish{border-left-color:#e05252}.zone.neutral{border-left-color:#e8a317}
.zone-title{font-weight:700;font-size:.95rem;margin-bottom:8px}
.zone-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
.stat{padding:9px;border-radius:9px;background:rgba(0,0,0,.08)}
.small{font-size:.75rem;opacity:.7}.value{font-weight:700;margin-top:3px}
.ladder-row{display:flex;justify-content:space-between;padding:7px 12px;border-radius:7px;margin-bottom:4px;font-size:.92rem}
.ladder-row.resistance{background:rgba(224,82,82,.08)}.ladder-row.support{background:rgba(38,162,105,.08)}
.ladder-row.anchor{background:rgba(59,130,246,.14);font-weight:700}
.ladder-row.current{outline:2px solid #f4c542;background:rgba(244,197,66,.2);font-weight:700}
.ladder-distance{display:inline-block;margin-left:8px;font-size:.72rem;opacity:.7;font-weight:400}
@media(max-width:640px){h1{font-size:1.4rem!important}.zone-grid{grid-template-columns:1fr}.news-title{font-size:1rem}.ladder-row{font-size:.85rem;padding:8px 9px}.ladder-distance{display:block;margin-left:0;margin-top:2px}}
</style>
""", unsafe_allow_html=True)
st.title("🎯 AI Trading Terminal - PRO")


# ================== SMALL UTILITIES ==================
def clean(value):
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def yahoo_search(query, limit=8):
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                          params={"q": query, "quotesCount": limit, "newsCount": 0},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        r.raise_for_status()
        return [x for x in r.json().get("quotes", []) if x.get("symbol")]
    except Exception:
        return []


@st.cache_data(ttl=10)
def fast_quote(ticker):
    """Fetch a current-ish quote, falling back to recent daily closes when needed."""
    try:
        fi = yf.Ticker(ticker).fast_info
        last = float(fi.get("last_price") or 0)
        prev = float(fi.get("previous_close") or 0)
        if last > 0 and prev > 0 and math.isfinite(last) and math.isfinite(prev):
            return last, (last - prev) / prev * 100
    except Exception:
        pass

    try:
        history = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
        closes = pd.to_numeric(history.get("Close"), errors="coerce").dropna()
        if closes.empty:
            return None, None
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else last
        if last <= 0 or not math.isfinite(last) or prev <= 0 or not math.isfinite(prev):
            return None, None
        return last, (last - prev) / prev * 100 if prev else 0.0
    except Exception:
        return None, None


# ================== SIDEBAR: UNIVERSAL SYMBOL SEARCH ==================
st.sidebar.header("🔍 Search Any Stock / Index / Crypto")
query = st.sidebar.text_input("Company name or ticker", "RELIANCE", key="global_search_query").strip().upper()
searches = yahoo_search(query) if query else []
if searches:
    labels = [f"{x['symbol']} — {x.get('shortname') or x.get('longname') or x['symbol']} ({x.get('exchange', '')} · {x.get('quoteType', '')})" for x in searches]
    picked = st.sidebar.radio("Matching symbols", labels, key="global_search_pick")
    symbol = searches[labels.index(picked)]["symbol"]
else:
    st.sidebar.caption("No matches found — using the manual selector below.")
    manual_market = st.sidebar.selectbox("Market", ["India - NSE (.NS)", "United States", "Crypto", "Forex", "Commodities"])
    raw = st.sidebar.text_input("Exact ticker", query).strip().upper()
    suffix = {"India - NSE (.NS)": ".NS", "Crypto": "-USD", "Forex": "=X", "Commodities": "=F"}.get(manual_market, "")
    symbol = raw if not suffix or raw.endswith(suffix) else raw + suffix

sentiment_filter = st.sidebar.multiselect("Sentiment filter", ["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"],
                                            default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"])

st.sidebar.markdown("---")
st.sidebar.header("📱 Telegram Alerts")
if st.sidebar.button("🧪 Test Telegram Alert"):
    ok = send_telegram_alert(f"🧪 <b>TEST ALERT</b>\n✅ Telegram working!\n🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.success("✅ Test sent!") if ok else st.sidebar.error("❌ Failed — check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in secrets.")


# ================== STOCK DATA ==================
@st.cache_data(ttl=120)
def get_stock(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.history(period="3mo", interval="1d"), (t.info or {})
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=15)
def get_intraday(ticker):
    """Last ~2 trading sessions of 15-min candles + that day's opening price (the 0° Gann anchor)."""
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="15m")
        if data.empty:
            return pd.DataFrame(), None
        today = datetime.now().date()
        data = data[data.index.date >= today - timedelta(days=2)]
        return data, (float(data["Open"].iloc[0]) if len(data) else None)
    except Exception:
        return pd.DataFrame(), None


@st.cache_data(ttl=120)
def get_two_day_intraday(ticker):
    """Per-day summary + Gann ladder for the last 2 available trading sessions (auto-skips
    weekends/holidays since yfinance only returns actual trading candles)."""
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="15m")
        if data.empty:
            return []
        data = data.copy()
        data["Date"] = data.index.date
        out = []
        for d in sorted(data["Date"].unique(), reverse=True)[:2]:
            day_df = data[data["Date"] == d]
            if len(day_df) < 3:
                continue
            o = float(day_df["Open"].iloc[0])
            out.append({
                "date": d, "open": o, "high": float(day_df["High"].max()),
                "low": float(day_df["Low"].min()), "close": float(day_df["Close"].iloc[-1]),
                "ladder": gann_ladder_levels(o),
            })
        return out
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_swing_data(ticker):
    try:
        return yf.Ticker(ticker).history(period="6mo", interval="1d")
    except Exception:
        return pd.DataFrame()


# ================== GANN LADDER / SMART MONEY ZONES ==================
def gann_ladder_levels(open_price):
    """Sorted ladder of every Gann-angle level (R + S) plus the 0° opening-price anchor."""
    sqrt_p = math.sqrt(abs(open_price))
    ladder = [{"label": "0° (Open)", "side": "OPEN", "value": float(open_price)}]
    for deg, delta in [(22.5, 22.5 / 180), (45, 45 / 180), (67.5, 67.5 / 180), (90, 90 / 180), (180, 180 / 180)]:
        ladder.append({"label": f"{deg}° R", "side": "R", "value": (sqrt_p + delta) ** 2})
        ladder.append({"label": f"{deg}° S", "side": "S", "value": (sqrt_p - delta) ** 2})
    ladder.sort(key=lambda x: x["value"])
    return ladder


def locate_band(ladder, price):
    if price <= ladder[0]["value"]:
        return "below"
    if price >= ladder[-1]["value"]:
        return "above"
    for i in range(len(ladder) - 1):
        if ladder[i]["value"] <= price <= ladder[i + 1]["value"]:
            return ladder[i], ladder[i + 1]
    return "below"


def zoom_window(ladder, band, price, context=1):
    if isinstance(band, tuple):
        lo_idx, hi_idx = ladder.index(band[0]), ladder.index(band[1])
    else:
        lo_idx, hi_idx = 0, len(ladder) - 1
    start, end = max(0, lo_idx - context), min(len(ladder) - 1, hi_idx + context)
    window = ladder[start:end + 1]
    vals = [lvl["value"] for lvl in window] + [price]
    y_min, y_max = min(vals), max(vals)
    pad = (y_max - y_min) * 0.18 if y_max > y_min else max(abs(price) * 0.01, 0.0001)
    return y_min - pad, y_max + pad, window


def detect_golden_pocket(price_df, lookback=80):
    if price_df is None or len(price_df) < 15:
        return None
    d = price_df.tail(lookback)
    swing_high, swing_low = float(d["High"].max()), float(d["Low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return None
    uptrend = d["Low"].idxmin() < d["High"].idxmax()
    if uptrend:
        gp_low, gp_high = swing_high - diff * 0.65, swing_high - diff * 0.618
    else:
        gp_low, gp_high = swing_low + diff * 0.618, swing_low + diff * 0.65
    return {"gp_low": gp_low, "gp_high": gp_high, "uptrend": uptrend}


def swing_levels(daily_df, window=4, lookback=130, max_levels=4, tol_pct=0.008):
    """Long-term swing support/resistance from local pivot highs/lows over ~6 months."""
    d = daily_df.tail(lookback)
    if len(d) < window * 2 + 5:
        return [], []
    h_vals, l_vals = d["High"].values, d["Low"].values
    highs, lows = [], []
    for i in range(window, len(d) - window):
        seg_h, seg_l = h_vals[i - window:i + window + 1], l_vals[i - window:i + window + 1]
        if h_vals[i] == seg_h.max():
            highs.append(float(h_vals[i]))
        if l_vals[i] == seg_l.min():
            lows.append(float(l_vals[i]))

    def dedupe(values):
        out = []
        for v in sorted(set(round(v, 2) for v in values), reverse=True):
            if not out or all(abs(v - o) / v > tol_pct for o in out):
                out.append(v)
            if len(out) >= max_levels:
                break
        return out

    return dedupe(highs), dedupe(lows)


def render_zone_card(css_class, title, price, zone_low, zone_high, note):
    st.markdown(f"""<div class="zone {css_class}"><div class="zone-title">{title}</div>
<div class="zone-grid">
  <div class="stat"><div class="small">Current</div><div class="value">{price:.2f}</div></div>
  <div class="stat"><div class="small">Zone Low</div><div class="value">{zone_low:.2f}</div></div>
  <div class="stat"><div class="small">Zone High</div><div class="value">{zone_high:.2f}</div></div>
</div><p>{note}</p></div>""", unsafe_allow_html=True)


def render_ladder(ladder, band, current_price):
    for lvl in reversed(ladder):
        css = "anchor" if lvl["side"] == "OPEN" else ("resistance" if lvl["side"] == "R" else "support")
        distance = current_price - lvl["value"]
        if abs(distance) <= 0.005:
            css = "current"
            distance_text = "CURRENT PRICE"
        else:
            distance_text = f"{distance:+.2f} from current"
        st.markdown(
            f"""
            <div class="ladder-row {css}">
                <span>
                    <b>{html_module.escape(lvl["label"])}</b>
                    <small class="ladder-distance">
                        {html_module.escape(distance_text)}
                    </small>
                </span>
                <span>{lvl["value"]:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_price_position(price, ladder, band):
    st.markdown("#### 🎯 Live Price Position")
    st.caption("The current price is shown relative to the Gann levels below.")
    if band == "below":
        st.warning(
            f"⬇️ Price **{price:.2f}** is BELOW every calculated level "
            f"(lowest: {ladder[0]['label']} @ {ladder[0]['value']:.2f}) — possible breakdown zone."
        )
    elif band == "above":
        st.warning(
            f"⬆️ Price **{price:.2f}** is ABOVE every calculated level "
            f"(highest: {ladder[-1]['label']} @ {ladder[-1]['value']:.2f}) — possible breakout zone."
        )
    else:
        lo, hi = band
        st.success(
            f"💰 Price **{price:.2f}** is between **{lo['label']}** ({lo['value']:.2f}) "
            f"and **{hi['label']}** ({hi['value']:.2f})"
        )
        span = hi["value"] - lo["value"]
        pct = (price - lo["value"]) / span if span else 0
        st.progress(
            min(max(pct, 0.0), 1.0),
            text=f"{pct * 100:.1f}% of the way from {lo['label']} to {hi['label']}",
        )


# ================== NEWS PIPELINE ==================
@st.cache_data(ttl=180, show_spinner=False)
def get_news():
    unique = {}
    for source, url in NEWS_SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as response:
                root = ET.fromstring(response.read())
            for item in root.findall(".//item")[:20]:
                title, link = clean(item.findtext("title", "")), (item.findtext("link", "") or "").strip()
                if not title:
                    continue
                published = (item.findtext("pubDate", "") or "").strip()
                try:
                    dt = email.utils.parsedate_to_datetime(published)
                    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc) - timedelta(days=7)
                key = link or re.sub(r"[^a-z0-9]", "", title.lower())
                unique.setdefault(key, {"title": title, "summary": clean(item.findtext("description", "")) or title,
                                         "link": link, "source": source, "published": published, "dt": dt})
        except Exception as exc:
            print(f"News source error ({source}): {exc}")
    return list(unique.values())


def keywords(ticker, info):
    raw = re.sub(r"(\.NS|-USD|=X|=F)$", "", ticker.lower()).lstrip("^")
    name = (info.get("longName") or info.get("shortName") or "").lower()
    kw = {raw, name, *re.findall(r"[a-z0-9]+", name)}
    kw.update(a.lower() for a in SYMBOL_ALIASES.get(ticker, []))
    return [x for x in kw if len(x) >= 3]


def sector_words(ticker, info):
    text = f"{ticker} {info.get('longName', '')} {info.get('sector', '')} {info.get('industry', '')}".lower()
    groups = ["technology", "software", "cloud", "semiconductor", "chip", "bank", "finance", "interest rate", "credit",
              "oil", "gas", "energy", "crude", "auto", "vehicle", "ev", "pharma", "drug", "healthcare", "fda",
              "consumer", "retail", "fmcg", "metal", "steel", "copper", "mining", "crypto", "bitcoin", "blockchain",
              "inflation", "tariff", "currency", "regulation"]
    return [w for w in groups if w in text]


def rank_article(article, direct_keys, indirect_keys):
    text = f"{article['title']} {article['summary']}".lower()
    d_hits, i_hits = sum(k in text for k in direct_keys), sum(k in text for k in indirect_keys)
    if not d_hits and not i_hits:
        return None
    catalysts = ["earnings", "revenue", "profit", "guidance", "regulator", "lawsuit", "acquisition", "merger",
                 "tariff", "rate", "inflation", "sanction", "upgrade", "downgrade", "contract", "warning",
                 "forecast", "results"]
    age = max(0, (datetime.now(timezone.utc) - article["dt"]).total_seconds() / 3600)
    freshness = max(0, 10 - min(age / 3, 10))
    source_weight = {"CNBC": 1.7, "Economic Times": 1.5, "Moneycontrol": 1.4, "Yahoo Finance": 1.4, "MarketWatch": 1.3}.get(article["source"], 1)
    result = dict(article)
    result["direct"] = bool(d_hits)
    result["importance"] = round(min(100, 42 * min(d_hits, 2) + 14 * min(i_hits, 3) + 7 * min(sum(w in text for w in catalysts), 3) + freshness + source_weight * 4), 1)
    return result


def impact(article, ticker, info):
    text = f"{article['title']} {article['summary']}".lower()
    positive = ["beat", "growth", "profit", "upgrade", "strong", "rises", "surge", "record", "contract", "expansion", "demand", "guidance raised"]
    negative = ["miss", "loss", "downgrade", "weak", "falls", "drop", "lawsuit", "warning", "cuts", "default", "pressure", "guidance cut"]
    score = max(0, min(10, 5 + sum(w in text for w in positive) * 1.4 - sum(w in text for w in negative) * 1.6))
    tone, label = ("green", "Bullish") if score >= 6.5 else (("red", "Bearish") if score <= 3.5 else ("yellow", "Neutral"))
    if article["direct"]:
        reason = f"Direct catalyst: this story mentions {ticker} or its company name and may affect earnings, valuation, demand, or risk sentiment directly."
    else:
        reason = f"Indirect catalyst: this concerns {info.get('sector') or info.get('industry') or 'a related market factor'} and may affect {ticker} through demand, costs, rates, regulation, currency, or investor risk appetite."
    return {"score": round(score, 1), "tone": tone, "label": label, "reason": reason}


def explain_indirect(items, ticker, info):
    if not groq_client or not items:
        return {}
    body = "\n".join(f"{i}|{x['title']} — {x['summary']}" for i, x in enumerate(items))
    prompt = f"Explain the indirect connection of each news item to {ticker} ({info.get('longName', '')}). Return one concise line as INDEX|EXPLANATION. Do not invent facts.\n{body}"
    try:
        answer = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": prompt}], temperature=.1, max_tokens=900).choices[0].message.content
        return {int(m.group(1)): m.group(2).strip() for line in answer.splitlines() if (m := re.match(r"\s*(\d+)\s*[|:-]\s*(.+)", line))}
    except Exception:
        return {}


def render_news(article):
    x = article["impact"]
    kind = "direct" if article["direct"] else "indirect"
    st.markdown(f"<div class='news-card {kind}'><div class='news-head'><span class='badge {x['tone']}'>{'DIRECT' if article['direct'] else 'INDIRECT'} · {x['label']}</span><b>Importance {article['importance']:.0f}/100 · Impact {x['score']:.1f}/10</b></div><div class='news-title'>{html_module.escape(article['title'])}</div><div class='news-summary'>{html_module.escape(article['summary'])}</div><div class='news-reason'>🧠 <b>Why it matters:</b> {html_module.escape(x['reason'])}</div><div class='news-meta'>📌 {html_module.escape(article['source'])} · 🕒 {html_module.escape(article['published'])} · <a href='{html_module.escape(article['link'])}' target='_blank'>Read source →</a></div></div>", unsafe_allow_html=True)


# ================== LEVELS / TRADE SUGGESTION / BACKTEST ==================
def levels(data):
    high, low, close = map(float, [data.High.iloc[-1], data.Low.iloc[-1], data.Close.iloc[-1]])
    p = (high + low + close) / 3
    return {"price": close, "pivot": round(p, 2), "r1": round(2 * p - low, 2), "r2": round(p + high - low, 2),
            "r3": round(high + 2 * (p - low), 2), "r4": round(high + 3 * (p - low), 2), "s1": round(2 * p - high, 2),
            "s2": round(p - high + low, 2), "s3": round(low - 2 * (high - p), 2), "s4": round(low - 3 * (high - p), 2)}


def get_trade_suggestion(ticker, price, q, ladder, band, articles):
    if not groq_client:
        return (
            "⚠️ AI is not connected. Add `GROQ_KEY` to Replit Secrets, or add "
            "`GROQ_KEY = \"your-key\"` to `.streamlit/secrets.toml` when running "
            "this script outside Replit, then restart the app."
        )
    band_text = (f"between {band[0]['label']} ({band[0]['value']:.2f}) and {band[1]['label']} ({band[1]['value']:.2f})"
                 if isinstance(band, tuple) else f"{band} every calculated level")
    news_text = "\n".join(f"- {a['title']} (impact {a['impact']['label']} {a['impact']['score']}/10)" for a in articles[:5]) or "No relevant news."
    prompt = (f"You are a trading analyst. Ticker: {ticker}. Current price: {price:.2f}.\n"
              f"Pivot={q['pivot']}, R1={q['r1']}, S1={q['s1']}.\n"
              f"Price is currently {band_text} on the Gann angle ladder.\n"
              f"Recent relevant news:\n{news_text}\n\n"
              f"Give a concise INTRADAY plan (entry, stop-loss, target) and a SWING plan (entry, stop-loss, target). "
              f"Be specific with numeric levels. This is educational, not financial advice.")
    try:
        res = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=700)
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Error: {type(e).__name__}: {e}"


def backtest(ticker, start, end, strategy):
    try:
        data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        trades = []
        for i in range(1, len(data)):
            p = (data.High.iloc[i - 1] + data.Low.iloc[i - 1] + data.Close.iloc[i - 1]) / 3
            s1, r1 = 2 * p - data.High.iloc[i - 1], 2 * p - data.Low.iloc[i - 1]
            if strategy == "Buy at S1" and data.Low.iloc[i] <= s1:
                trades.append((s1, p))
            if strategy == "Sell at R1" and data.High.iloc[i] >= r1:
                trades.append((r1, p))
        profits = [x - e for e, x in trades]
        return {"Trades": len(profits), "Win %": round(sum(x > 0 for x in profits) / len(profits) * 100, 1) if profits else 0,
                "Average profit": round(sum(profits) / len(profits), 2) if profits else 0, "Total profit": round(sum(profits), 2)}
    except Exception:
        return None


# ================== LOAD DATA FOR SELECTED SYMBOL ==================
df, info = get_stock(symbol)
if df.empty:
    st.error(f"No market data found for {symbol}. Try the search box in the sidebar with a different spelling.")
    st.stop()

live_price, _live_change = fast_quote(symbol)
price = live_price if live_price is not None else float(df.Close.iloc[-1])

articles = []
for raw in get_news():
    item = rank_article(raw, keywords(symbol, info), sector_words(symbol, info))
    if item:
        item["impact"] = impact(item, symbol, info)
        articles.append(item)
articles.sort(key=lambda x: (x["importance"], x["dt"]), reverse=True)
indirect = [x for x in articles if not x["direct"]][:10]
for idx, text in explain_indirect(indirect, symbol, info).items():
    if 0 <= idx < len(indirect):
        indirect[idx]["impact"]["reason"] = text

intraday_df, open_ref = get_intraday(symbol)
chart_df = intraday_df if not intraday_df.empty else df
gann_anchor = open_ref if open_ref else price
ladder = gann_ladder_levels(gann_anchor)
band = locate_band(ladder, price)

# ================== TABS ==================
tab_live, tab_news, tab_chat, tab_heat, tab_watch, tab_alerts, tab_history = st.tabs(
    ["📊 Live Analysis", "📰 Ranked Stock News", "💬 AI Chat", "📊 Market Heatmap",
     "📋 Watchlists", "🔔 Alerts & Backtest", "📅 Historical Data"])

# ---------------- TAB: LIVE ANALYSIS ----------------
with tab_live:
    st.header(f"📊 Live Analysis: {symbol}")
    col_main, col_ai = st.columns([2, 1], gap="large")

    with col_main:
        a, b, c = st.columns(3)
        a.metric("Current Price", f"{price:.2f}")
        b.metric("Company", info.get("longName", symbol))
        c.metric("Related stories", len(articles))

        if not chart_df.empty:
            zoom = zoom_window(ladder, band, price, context=1)
            fig = go.Figure(go.Candlestick(x=list(range(len(chart_df))), open=chart_df.Open, high=chart_df.High,
                                            low=chart_df.Low, close=chart_df.Close, name=symbol))
            if zoom:
                y_min, y_max, visible = zoom
                for lvl in visible:
                    color = "#26a269" if lvl["side"] == "S" else ("#3b82f6" if lvl["side"] == "OPEN" else "#e05252")
                    fig.add_hline(y=lvl["value"], line_dash="dash", line_width=1.6, line_color=color, opacity=0.9,
                                  annotation_text=f"{lvl['label']}  {lvl['value']:.2f}", annotation_position="left",
                                  annotation_font_size=12, annotation_bgcolor="rgba(255,255,255,0.75)")
                if isinstance(band, tuple):
                    fig.add_hrect(y0=band[0]["value"], y1=band[1]["value"], fillcolor="#f4c542", opacity=0.15, line_width=0)
                fig.update_yaxes(range=[y_min, y_max])
            # Categorical x-axis avoids ugly weekend/overnight gaps for any market worldwide
            step = max(1, len(chart_df) // 6)
            ticks = list(range(0, len(chart_df), step))
            fig.update_xaxes(tickvals=ticks, ticktext=[chart_df.index[i].strftime("%d %b %H:%M") for i in ticks], tickangle=-30)
            fig.update_layout(
                height=500,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,20,32,0.92)",
                hovermode="x unified",
                xaxis_rangeslider_visible=False,
                margin=dict(l=12, r=110, t=45, b=55),
                font=dict(family="Inter, Arial, sans-serif", size=12),
                title=dict(
                    text=f"{symbol} · Live 15-minute price action",
                    x=0.02,
                    xanchor="left",
                    font=dict(size=16),
                ),
                xaxis=dict(
                    showgrid=False,
                    showline=True,
                    linecolor="rgba(255,255,255,.18)",
                    zeroline=False,
                    fixedrange=False,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,.08)",
                    showline=False,
                    zeroline=False,
                    side="right",
                    fixedrange=False,
                ),
            )
            fig.update_traces(
                increasing_line_color="#22c55e",
                increasing_fillcolor="#22c55e",
                decreasing_line_color="#ef4444",
                decreasing_fillcolor="#ef4444",
                selector=dict(type="candlestick"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No intraday candle data available for this symbol right now.")

        st.subheader("📐 Full Gann Ladder")
        render_price_position(price, ladder, band)
        render_ladder(ladder, band, price)

        st.subheader("🧠 Smart Money Zones")
        gp = detect_golden_pocket(chart_df)
        if gp:
            inside = gp["gp_low"] <= price <= gp["gp_high"]
            trend = "Bullish pullback" if gp["uptrend"] else "Bearish bounce"
            render_zone_card("bullish" if gp["uptrend"] else "bearish", f"🌀 Golden Pocket (61.8%–65% Fib) · {trend}",
                              price, gp["gp_low"], gp["gp_high"],
                              "Price is inside the golden pocket." if inside else "Price is outside the golden pocket.")
        else:
            st.caption("Not enough recent data to compute a golden pocket.")

        st.subheader("🎯 Traditional Pivot Levels")
        q = levels(df)
        st.dataframe(pd.DataFrame([{"Level": k.upper(), "Value": q[k]} for k in ["r4", "r3", "r2", "r1", "pivot", "s1", "s2", "s3", "s4"]]),
                     use_container_width=True, hide_index=True)

    with col_ai:
        st.subheader("💡 AI Trade Suggestion")
        st.caption("Intraday & swing read: entry, stop-loss, target.")
        if st.button("🚀 Generate Suggestion", use_container_width=True):
            with st.spinner("Analyzing..."):
                st.session_state["last_suggestion"] = get_trade_suggestion(symbol, price, q, ladder, band, articles)
        last = st.session_state.get("last_suggestion")
        if last:
            st.error(last) if last.startswith("⚠️") else st.markdown(last)
        else:
            st.caption("Tap the button to get an INTRADAY + SWING plan for this ticker.")

# ---------------- TAB: RANKED STOCK NEWS ----------------
with tab_news:
    st.header(f"📰 Latest Relevant News — {symbol}")
    st.caption("Fresh, deduplicated stories from all feeds, ranked by direct/indirect relevance, importance, and freshness.")
    include_indirect = st.checkbox("Include indirect sector/macro news", True)
    limit = st.slider("Number of ranked stories", 5, 30, 12)
    shown = [x for x in articles if include_indirect or x["direct"]]
    shown = [x for x in shown if (x["impact"]["label"] == "Bullish" and "🟢 Bullish" in sentiment_filter)
             or (x["impact"]["label"] == "Bearish" and "🔴 Bearish" in sentiment_filter)
             or (x["impact"]["label"] == "Neutral" and "🟡 Neutral" in sentiment_filter)]
    if shown:
        for article in shown[:limit]:
            render_news(article)
    else:
        st.warning("No relevant stories matched the current filters.")

# ---------------- TAB: AI CHAT ----------------
with tab_chat:
    st.header("💬 AI Chat Assistant")
    model_choice = st.selectbox("Choose Model", list(GROQ_MODELS) + list(GEMINI_MODELS))
    st.session_state.setdefault("chat_messages", [])
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about stock, levels, news, strategy...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        context = "\n".join(f"{x['title']} | {x['impact']['reason']}" for x in articles[:10])
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if model_choice in GROQ_MODELS:
                        answer = (
                            "⚠️ AI is not connected. Add `GROQ_KEY` to Replit Secrets, or add "
                            "`GROQ_KEY = \"your-key\"` to `.streamlit/secrets.toml` when running "
                            "this script outside Replit, then restart the app."
                            if not groq_client
                            else
                                  groq_client.chat.completions.create(
                                      model=GROQ_MODELS[model_choice],
                                      messages=[{"role": "system", "content": f"Trading assistant for {symbol}. Educational only. Context:\n{context}"},
                                                {"role": "user", "content": prompt}],
                                      temperature=.2, max_tokens=900).choices[0].message.content)
                    elif model_choice in GEMINI_MODELS:
                        if not gemini_client:
                            answer = "⚠️ Configure GEMINI_KEY in .streamlit/secrets.toml."
                        else:
                            resp = gemini_client.models.generate_content(
                                model=GEMINI_MODELS[model_choice],
                                contents=f"Trading assistant for {symbol}. Context:{context}\nQuestion:{prompt}")
                            answer = resp.text
                    else:
                        answer = "⚠️ Invalid model selected."
                except Exception as exc:
                    answer = f"⚠️ AI error: {type(exc).__name__}: {exc}"
                st.error(answer) if answer.startswith("⚠️") else st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

# ---------------- TAB: MARKET HEATMAP ----------------
with tab_heat:
    st.header("📊 Market Heatmap")
    heat_market = st.selectbox("Market", list(SECTOR_INDICES.keys()), key="heat_market")
    sub_sector, sub_stock = st.tabs(["📊 Sector Performance", "🔥 Stock Heatmap"])

    with sub_sector:
        rows = []
        for name, tkr in SECTOR_INDICES.get(heat_market, {}).items():
            last, change = fast_quote(tkr)
            if last is None:
                continue
            intensity = min(10, abs(change) / 0.5)
            bar = ("🟢" if change > 0 else "🔴") * max(1, int(intensity)) if change else "⚪"
            rows.append({"Sector": name, "Price": round(last, 2), "Change %": round(change, 2), "Bar": bar})
        if rows:
            for row in sorted(rows, key=lambda r: r["Change %"], reverse=True):
                c1, c2, c3 = st.columns([2, 1, 2])
                c1.markdown(f"### {row['Bar']} {row['Sector']}")
                c2.metric("Price", f"{row['Price']:.2f}")
                c3.metric("Change", f"{row['Change %']:.2f}%")
                st.markdown("---")
        else:
            st.warning("Could not fetch sector data right now — try again in a moment.")

    with sub_stock:
        stock_groups = SECTOR_STOCKS.get(heat_market, {})
        if not stock_groups:
            st.info("No stock-level breakdown for this market yet.")
        else:
            chosen_sector = st.selectbox("Select sector", list(stock_groups.keys()), key="heat_sector")
            rows = []
            for tkr in stock_groups[chosen_sector]:
                last, change = fast_quote(tkr)
                if last is None:
                    continue
                name = tkr.replace(".NS", "").replace("-USD", "").replace("=F", "").replace("=X", "")
                intensity = min(10, abs(change) / 0.3)
                bar = ("🟢" if change > 0 else ("🔴" if change < 0 else "⚪")) * max(1, int(intensity))
                rows.append({"Stock": name, "Price": round(last, 2), "Change %": round(change, 2), "Bar": bar})
            if rows:
                for row in sorted(rows, key=lambda r: r["Change %"], reverse=True):
                    c1, c2, c3 = st.columns([2, 1, 2])
                    c1.markdown(f"### {row['Bar']} {row['Stock']}")
                    c2.metric("Price", f"{row['Price']:.2f}")
                    c3.metric("Change", f"{row['Change %']:.2f}%")
                    st.markdown("---")
            else:
                st.warning("Could not fetch stock data right now — try again in a moment.")

# ---------------- TAB: WATCHLIST & PORTFOLIO ----------------
with tab_watch:
    st.header("📋 Watchlists")
    st.session_state.setdefault("watchlists", {"My Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS"], "US Tech": ["AAPL", "MSFT", "NVDA"], "Crypto": ["BTC-USD", "ETH-USD"]})
    st.session_state.setdefault("current_watchlist", "My Stocks")

    wcol1 = st.container()
    with wcol1:
        st.subheader("📋 Watchlists")
        new_list = st.text_input("New watchlist name", key="new_watchlist_name")
        if st.button("➕ Create watchlist", key="create_watchlist") and new_list and new_list not in st.session_state.watchlists:
            st.session_state.watchlists[new_list] = []
            st.session_state.current_watchlist = new_list
            st.rerun()
        names = list(st.session_state.watchlists.keys())
        chosen = st.selectbox("Select watchlist", names, index=names.index(st.session_state.current_watchlist) if st.session_state.current_watchlist in names else 0)
        st.session_state.current_watchlist = chosen

        add_ticker = st.text_input("Add ticker to this watchlist", key="watch_add")
        if st.button("➕ Add", key="watch_add_button") and add_ticker:
            t = add_ticker.upper().strip()
            if t and t not in st.session_state.watchlists[chosen]:
                st.session_state.watchlists[chosen].append(t)
                st.rerun()

        rows = []
        for ticker in st.session_state.watchlists[chosen]:
            last, change = fast_quote(ticker)
            rows.append({"Ticker": ticker, "Price": round(last, 2) if last is not None else "N/A",
                         "Change %": round(change, 2) if change is not None else "N/A"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.session_state.watchlists[chosen]:
            remove_choice = st.selectbox("Remove a ticker", ["—"] + st.session_state.watchlists[chosen], key="watch_remove_pick")
            if st.button("🗑️ Remove", key="watch_remove_button") and remove_choice != "—":
                st.session_state.watchlists[chosen].remove(remove_choice)
                st.rerun()

# ---------------- TAB: ALERTS & BACKTEST ----------------
with tab_alerts:
    st.header("🔔 Alerts & Backtest")
    st.session_state.setdefault("alerts", [])
    alert_default_price = max(0.01, round(float(price or 0.0), 2))
    if st.session_state.get("alert_value", alert_default_price) < 0.01:
        st.session_state["alert_value"] = alert_default_price

    x1, x2, x3 = st.columns(3)
    with x1: aticker = st.text_input("Alert ticker", value=symbol, key="alert_ticker")
    with x2:
        avalue = st.number_input(
            "Alert price",
            min_value=0.01,
            value=alert_default_price,
            key="alert_value",
        )
    with x3: adirection = st.selectbox("Condition", ["Above", "Below"], key="alert_direction")
    if st.button("Create alert", key="create_alert") and aticker:
        st.session_state.alerts.append({"ticker": aticker.upper(), "price": avalue, "direction": adirection, "notified": False})
        st.rerun()

    if st.session_state.alerts:
        st.subheader("📋 Active Reminders")
        for idx, alert in enumerate(list(st.session_state.alerts)):
            last, _ = fast_quote(alert["ticker"])
            last = last if last is not None else 0.0
            hit = (alert["direction"] == "Above" and last >= alert["price"]) or (alert["direction"] == "Below" and last <= alert["price"])
            c1, c2 = st.columns([5, 1])
            with c1:
                (st.error if hit else st.info)(f"{alert['ticker']} {alert['direction']} {alert['price']:.2f} · Current {last:.2f}")
            with c2:
                if st.button("❌ Cancel", key=f"cancel_alert_{idx}"):
                    st.session_state.alerts.pop(idx)
                    st.rerun()
            if hit and not alert.get("notified"):
                sent = send_telegram_alert(f"🚨 <b>PRICE ALERT</b>\n{alert['ticker']} is now {alert['direction']} {alert['price']:.2f}\nCurrent: {last:.2f}")
                if sent:
                    st.session_state.alerts[idx]["notified"] = True

    st.subheader("🧪 Backtesting")
    b1, b2 = st.columns(2)
    with b1: bticker = st.text_input("Backtest ticker", symbol, key="bticker")
    with b2: strategy = st.selectbox("Strategy", ["Buy at S1", "Sell at R1"], key="bstrategy")
    start = st.date_input("Start", datetime.now() - timedelta(days=90), key="bstart")
    end = st.date_input("End", datetime.now(), key="bend")
    if st.button("🚀 Run Backtest", key="run_backtest"):
        result = backtest(bticker, start, end, strategy)
        st.json(result) if result else st.error("Backtest failed — check the ticker.")

# ---------------- TAB: HISTORICAL DATA ----------------
with tab_history:
    st.header(f"📅 Historical Data — {symbol}")
    st.caption("Last 2 trading sessions (auto-adjusts across weekends/holidays — yfinance only returns real trading candles) plus long-term swing support/resistance.")

    days = get_two_day_intraday(symbol)
    if not days:
        st.warning("No intraday history available for this symbol.")
    else:
        for day in days:
            st.markdown(f"#### 📅 {day['date']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Open", f"{day['open']:.2f}")
            c2.metric("High", f"{day['high']:.2f}")
            c3.metric("Low", f"{day['low']:.2f}")
            c4.metric("Close", f"{day['close']:.2f}")
            rows = [{"Level": lvl["label"], "Price": round(lvl["value"], 2)} for lvl in reversed(day["ladder"])]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")

    st.subheader("📐 Long-Term Swing Support / Resistance (6-month)")
    swing_df = get_swing_data(symbol)
    if swing_df.empty or len(swing_df) < 30:
        st.info("Not enough long-term history to compute swing levels for this symbol.")
    else:
        res_levels, sup_levels = swing_levels(swing_df)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**🔴 Swing Resistance**")
            for r in res_levels:
                st.markdown(f"- {r:.2f}")
            if not res_levels:
                st.caption("None found.")
        with sc2:
            st.markdown("**🟢 Swing Support**")
            for s in sup_levels:
                st.markdown(f"- {s:.2f}")
            if not sup_levels:
                st.caption("None found.")

st.caption("Gann ladder, order blocks, golden pockets, news impact, and AI suggestions are heuristic/AI-assisted and are not investment advice.")
