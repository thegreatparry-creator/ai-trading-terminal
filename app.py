import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import email.utils
from datetime import datetime, timezone, timedelta
from groq import Groq
import google.generativeai as genai
import requests


# ================== API KEYS ==================
GEMINI_KEY = "AQ.Ab8RN6JtGdVf9VFtpeo2_7BYDuZQZZlhMIbQxKFX1noZ4UnTSQ"
GROQ_KEY = "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"
TELEGRAM_BOT_TOKEN = "8794257218:AAGYGDqPUEJdI3UahL07Pe86IgcLCfIn20g"
TELEGRAM_CHAT_ID = "8600332637"


# ================== TELEGRAM BOT CONFIG ==================
TELEGRAM_BOT_TOKEN = "8794257218:AAGYGDqPUEJdI3UahL07Pe86IgcLCfIn20g"
TELEGRAM_CHAT_ID = "8600332637"


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
genai.configure(api_key=GEMINI_KEY)


# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Institutional Terminal", layout="wide")
st.title("🎯 AI Trading Terminal - PRO")


# ================== SIDEBAR ==================
st.sidebar.header("🌐 Market Selection")
select_market = st.sidebar.selectbox("Asset Class / Country",
    ["India - NSE (.NS)", "India Index Benchmark", "United States (No Suffix)", "Cryptocurrency (-USD)", "Forex Currency (=X)", "Commodities", "Global Indices"])


default_symbol = "RELIANCE"
if select_market == "India Index Benchmark": default_symbol = "^NSEI"
elif select_market == "United States (No Suffix)": default_symbol = "AAPL"
elif select_market == "Cryptocurrency (-USD)": default_symbol = "BTC-USD"
elif select_market == "Forex Currency (=X)": default_symbol = "EURUSD=X"
elif select_market == "Commodities": default_symbol = "GC=F"
elif select_market == "Global Indices": default_symbol = "^GSPC"
else: default_symbol = "RELIANCE"


search_ticker = st.sidebar.text_input("Enter Ticker / Symbol", default_symbol).strip().upper()


if select_market == "India - NSE (.NS)" and not search_ticker.endswith(".NS"): full_symbol = f"{search_ticker}.NS"
elif select_market == "Cryptocurrency (-USD)" and not search_ticker.endswith("-USD"): full_symbol = f"{search_ticker}-USD"
elif select_market == "Forex Currency (=X)" and not search_ticker.endswith("=X"): full_symbol = f"{search_ticker}=X"
elif select_market == "Commodities" and not search_ticker.endswith("=F"): full_symbol = f"{search_ticker}=F"
else: full_symbol = search_ticker


st.sidebar.markdown("---")
st.sidebar.header("📡 Global News Sources")
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
def harvest_global_news():
    news_items = []
    for url in global_news_sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                root = ET.fromstring(resp.read())
                for item in root.findall(".//item")[:5]:
                    title = item.findtext("title", "").strip()
                    link = item.findtext("link", "").strip()
                    pub = item.findtext("pubDate", "").strip()
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
                    news_items.append({"title": title, "link": link, "source": source, "time": pub, "impact": score, "is_breaking": is_breaking})
        except Exception as e:
            print(f"News source error: {e}")
    return news_items[:20]


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
        res = groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return res.choices[0].message.content
    except: return "AI unavailable"


def get_chat_response(user_message, context_text, model_choice):
    system_prompt = f"Trading assistant. Context: {context_text}. Answer questions. Educational only."
    try:
        if model_choice == "Groq (Llama 3.1 8B)":
            response = groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}], temperature=0.3, max_tokens=800)
            return response.choices[0].message.content
        elif model_choice == "Groq (Gemma 2 9B)":
            response = groq_client.chat.completions.create(model="gemma2-9b-it", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}], temperature=0.3, max_tokens=800)
            return response.choices[0].message.content
        elif model_choice == "Groq (Mixtral 8x7B)":
            response = groq_client.chat.completions.create(model="mixtral-8x7b-32768", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}], temperature=0.3, max_tokens=800)
            return response.choices[0].message.content
        elif model_choice == "Groq (Llama 3.3 70B)":
            response = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}], temperature=0.3, max_tokens=800)
            return response.choices[0].message.content
        elif model_choice == "Gemini 1.5 Flash":
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(f"{system_prompt}\n\nQ: {user_message}")
            return response.text
        elif model_choice == "Gemini 1.5 Pro":
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(f"{system_prompt}\n\nQ: {user_message}")
            return response.text
        else:
            return "Invalid model selected"
    except Exception as e:
        print(f"AI Error: {e}")
        return "Error"


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
        st.error(f"❌ No data for {full_symbol}")
    else:
        df_15m, opening_15m = get_cached_15min_data(full_symbol)
        quant = calculate_levels(df, opening_15m)
        st.session_state.quant_data = quant
        news_list = get_cached_global_news()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Current Price", f"{quant['price']:.2f}")
        col2.metric("📊 Pattern", quant['pattern'])
        col3.metric("📈 Pivot", f"{quant['pivot']:.2f}")
        
        st.markdown("### 🎯 Gann Angle Levels (Based on 15m Open)")
        gann_rows = []
        current_price = quant['price']
        for deg in ["22.5°", "45°", "67.5°", "90°", "180°"]:
            r_key = f'R_{deg.replace("°", "")}'
            s_key = f'S_{deg.replace("°", "")}'
            r_raw = quant.get(r_key)
            s_raw = quant.get(s_key)
            try:
                r_val = float(r_raw) if r_raw is not None else 0.0
            except (ValueError, TypeError):
                r_val = 0.0
            try:
                s_val = float(s_raw) if s_raw is not None else 0.0
            except (ValueError, TypeError):
                s_val = 0.0
            gann_rows.append({"Angle": deg, "Resistance": r_val, "Support": s_val})
        
        all_levels = []
        for row in gann_rows:
            if row['Support'] > 0:
                all_levels.append((row['Support'], f"S {row['Angle']}"))
            if row['Resistance'] > 0:
                all_levels.append((row['Resistance'], f"R {row['Angle']}"))
        all_levels.sort(key=lambda x: x[0])
        
        price_between = ""
        for i in range(len(all_levels) - 1):
            lower_level = all_levels[i]
            upper_level = all_levels[i + 1]
            if lower_level[0] < current_price < upper_level[0]:
                price_between = f"🟡 Price (${current_price:.2f}) is between {lower_level[1]} (${lower_level[0]:.2f}) and {upper_level[1]} (${upper_level[0]:.2f})"
                break
        
        if current_price > all_levels[-1][0]:
            price_between = f"🟢 Price (${current_price:.2f}) is ABOVE all levels"
        elif current_price < all_levels[0][0]:
            price_between = f"🔴 Price (${current_price:.2f}) is BELOW all levels"
        
        st.info(f"**0° (Center):** ${quant.get('price', 0):.2f}\n\n{price_between}")
        
        gann_df = pd.DataFrame(gann_rows)
        st.dataframe(gann_df, use_container_width=True)
        
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
        
        st.markdown("### 📍 Price Position")
        price_status = []
        for level_name in ['r4', 'r3', 'r2', 'r1', 's1', 's2', 's3', 's4']:
            val = quant.get(level_name, 0)
            try:
                level_val = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                level_val = 0.0
            status = "ABOVE" if quant['price'] > level_val else "BELOW"
            price_status.append({"Level": level_name.upper(), "Price": f"${level_val:.2f}", "Current vs Level": status, "Gap": f"${abs(quant['price'] - level_val):.2f}"})
        st.dataframe(pd.DataFrame(price_status), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📰 Global News Feed")
        filtered_news = [n for n in news_list if (n["impact"] > 0 and "🟢 Bullish" in sentiment_filter) or (n["impact"] < 0 and "🔴 Bearish" in sentiment_filter) or (n["impact"] == 0 and "🟡 Neutral" in sentiment_filter)]
        if filtered_news:
            for n in filtered_news[:15]:
                badge = "🟢" if n['impact'] > 0 else ("🔴" if n['impact'] < 0 else "🟡")
                st.markdown(f"{badge} **[{n['source']}]** {n['title']}")
                st.caption(f"📅 {n['time']} | [🔗 Link]({n['link']})")
                st.markdown("---")
        else:
            st.info("No news matching your filters")
        
        st.markdown("---")
        st.subheader("💡 AI Trading Suggestions")
        news_text = "\n".join([n['title'] for n in news_list[:5]]) if news_list else "No news"
        if st.button("🚀 Generate AI Suggestions"):
            with st.spinner("AI analyzing..."):
                suggestions = get_groq_suggestions(full_symbol, quant, news_text)
                st.markdown(suggestions)


# ================== TAB 2: AI CHAT ==================
with tab2:
    st.header("💬 AI Chat Assistant")
    st.markdown("**Ask about:** Stock analysis, levels, news, strategy, market outlook")
    
    model = st.selectbox("Choose Model", [
        "Groq (Llama 3.1 8B)",
        "Groq (Gemma 2 9B)", 
        "Groq (Mixtral 8x7B)",
        "Groq (Llama 3.3 70B)",
        "Gemini 1.5 Flash",
        "Gemini 1.5 Pro"
    ])
    
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
                    try:
                        response = get_chat_response(prompt, st.session_state.chat_context, model)
                        if response and response != "Error" and len(response) > 10:
                            st.markdown(response)
                        else:
                            st.error(f"❌ AI Error - Response: {response}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        response = "Error occurred"
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
