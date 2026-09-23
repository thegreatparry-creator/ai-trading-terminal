import email.utils
import html as html_module
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from groq import Groq

st.set_page_config(page_title="AI Institutional Terminal", layout="wide")


def get_secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GROQ_KEY = get_secret("GROQ_KEY")
GEMINI_KEY = get_secret("GEMINI_KEY")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

GROQ_MODELS = {
    "Groq GPT-OSS 20B": "openai/gpt-oss-20b",
    "Groq GPT-OSS 120B": "openai/gpt-oss-120b",
    "Groq Qwen3 32B": "qwen/qwen3-32b",
    "Groq Kimi K2": "moonshotai/kimi-k2-instruct",
}
GEMINI_MODELS = {"Gemini Flash": "gemini-3.5-flash", "Gemini Pro": "gemini-3.1-pro-preview"}
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

st.markdown("""
<style>
html,body,[class*="css"]{font-size:16px}.stButton>button{border-radius:8px;padding:.5rem 1rem}
.zone{border:1px solid rgba(130,150,180,.35);border-radius:15px;padding:15px;margin:10px 0;background:rgba(20,30,45,.55)}
.zone-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.stat{padding:9px;border-radius:9px;background:rgba(0,0,0,.22)}.small{font-size:.75rem;opacity:.7}.value{font-weight:700;margin-top:3px}
.news-card{border:1px solid rgba(150,150,150,.28);border-radius:14px;padding:15px 16px;margin:0 0 13px;background:rgba(255,255,255,.025)}.news-card.direct{border-left:5px solid #26a269}.news-card.indirect{border-left:5px solid #e8a317}.news-head{display:flex;justify-content:space-between;gap:10px;align-items:center}.news-title{font-size:1.05rem;font-weight:700;line-height:1.4;margin:7px 0}.news-summary,.news-reason{line-height:1.55;opacity:.88}.news-meta{font-size:.78rem;opacity:.68;margin-top:9px}.badge{padding:5px 10px;border-radius:99px;font-size:.75rem;font-weight:700}.green{background:#164d35;color:#6ee7a0}.yellow{background:#604811;color:#ffd166}.red{background:#5b2020;color:#ff8b8b}
@media(max-width:640px){h1{font-size:1.4rem!important}.zone-grid{grid-template-columns:1fr 1fr}.news-title{font-size:1rem}}
</style>
""", unsafe_allow_html=True)
st.title("🎯 AI Trading Terminal - PRO")


def yahoo_search(query):
    try:
        response = requests.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": query, "quotesCount": 8, "newsCount": 0}, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        response.raise_for_status()
        return [{"symbol": x["symbol"], "name": x.get("shortname") or x.get("longname") or x["symbol"], "exchange": x.get("exchange", ""), "type": x.get("quoteType", "")} for x in response.json().get("quotes", []) if x.get("symbol")]
    except Exception:
        return []


st.sidebar.header("🔍 Search Any Stock / Index / Crypto")
query = st.sidebar.text_input("Company name or ticker", "RELIANCE", key="global_search_query").strip().upper()
search_results = yahoo_search(query) if query else []
market = "India - NSE (.NS)"
if search_results:
    options = [f"{x['symbol']} — {x['name']} ({x['exchange']} · {x['type']})" for x in search_results]
    picked = st.sidebar.radio("Matching symbols", options, key="global_search_pick")
    full_symbol = search_results[options.index(picked)]["symbol"]
    search_ticker = full_symbol
else:
    market = st.sidebar.selectbox("Market", ["India - NSE (.NS)", "United States", "Crypto", "Forex", "Commodities"])
    raw = st.sidebar.text_input("Exact ticker", query).strip().upper()
    suffix = {"India - NSE (.NS)": ".NS", "Crypto": "-USD", "Forex": "=X", "Commodities": "=F"}.get(market, "")
    full_symbol = raw if not suffix or raw.endswith(suffix) else raw + suffix
    search_ticker = full_symbol

sentiment_filter = st.sidebar.multiselect("News sentiment", ["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"], default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"])


def clean_text(value):
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def parse_time(value):
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=30)


@st.cache_data(ttl=180, show_spinner=False)
def harvest_news():
    unique = {}
    for source, url in NEWS_SOURCES:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=6) as response:
                root = ET.fromstring(response.read())
            for item in root.findall(".//item")[:20]:
                title = clean_text(item.findtext("title", ""))
                if not title:
                    continue
                link = (item.findtext("link", "") or "").strip()
                published = (item.findtext("pubDate", "") or "").strip()
                key = link or re.sub(r"[^a-z0-9]", "", title.lower())
                unique.setdefault(key, {"title": title, "summary": clean_text(item.findtext("description", "")) or title, "link": link, "source": source, "published": published, "published_dt": parse_time(published)})
        except Exception as exc:
            print(f"News source error ({source}): {exc}")
    return list(unique.values())


@st.cache_data(ttl=90, show_spinner=False)
def get_daily_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period="3mo", interval="1d"), ticker.info or {}
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=15, show_spinner=False)
def get_intraday_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period="2d", interval="1m", auto_adjust=False)
        if data.empty:
            data = yf.Ticker(symbol).history(period="5d", interval="15m", auto_adjust=False)
        return data
    except Exception:
        return pd.DataFrame()


def get_first_15m_open(data):
    if data is None or data.empty:
        return None
    dates = pd.Series(data.index.date, index=data.index)
    today = datetime.now().date()
    today_data = data[dates == today]
    if today_data.empty:
        today_data = data.tail(1)
    return float(today_data["Open"].iloc[0])


def calculate_pivots(df):
    high, low, close = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1]), float(df["Close"].iloc[-1])
    pivot = (high + low + close) / 3
    return {"r1": 2 * pivot - low, "r2": pivot + high - low, "r3": high + 2 * (pivot - low), "r4": high + 3 * (pivot - low), "pivot": pivot, "s1": 2 * pivot - high, "s2": pivot - high + low, "s3": low - 2 * (high - pivot), "s4": low - 3 * (high - pivot)}


def build_gann_ladder(anchor):
    levels = [{"label": "0° Open", "value": float(anchor), "side": "OPEN", "angle": 0.0}]
    root = math.sqrt(abs(float(anchor)))
    for angle in [22.5, 45, 67.5, 90, 180]:
        delta = angle / 180
        levels.append({"label": f"{angle:g}° S", "value": (root - delta) ** 2, "side": "S", "angle": angle})
        levels.append({"label": f"{angle:g}° R", "value": (root + delta) ** 2, "side": "R", "angle": angle})
    return sorted(levels, key=lambda item: item["value"])


def locate_band(ladder, price):
    if not ladder:
        return None, None
    for lower, upper in zip(ladder, ladder[1:]):
        if lower["value"] <= price <= upper["value"]:
            return lower, upper
    if price < ladder[0]["value"]:
        return None, ladder[0]
    return ladder[-1], None


def get_or_set_session_anchor(symbol, fallback):
    key = f"gann_anchor_{symbol}"
    if st.session_state.get("anchor_symbol") != symbol:
        st.session_state["anchor_symbol"] = symbol
        st.session_state[key] = fallback
    if key not in st.session_state:
        st.session_state[key] = fallback
    return st.session_state[key]


def level_rows(ladder, price):
    return [{"Level": level["label"], "Price": round(level["value"], 2), "Distance": round(level["value"] - price, 2), "Type": "Open" if level["side"] == "OPEN" else "Resistance" if level["side"] == "R" else "Support"} for level in ladder]


def stock_keywords(symbol, info):
    raw = re.sub(r"(\.NS|-USD|=X|=F)$", "", symbol.lower())
    name = (info.get("longName") or info.get("shortName") or "").lower()
    words = re.findall(r"[a-z0-9]+", name)
    return [x for x in {raw, name, *[w for w in words if len(w) > 2]} if x]


def ranked_news(news, symbol, info):
    direct_keys = stock_keywords(symbol, info)
    industry = f"{info.get('sector','')} {info.get('industry','')}".lower()
    related_words = {"bank", "rate", "inflation", "oil", "gas", "energy", "technology", "software", "chip", "semiconductor", "auto", "vehicle", "pharma", "healthcare", "metal", "steel", "crypto", "currency", "tariff", "regulation"}
    related_keys = [word for word in related_words if word in industry or word in f"{symbol} {info.get('longName','')}".lower()]
    positive = ["beat", "growth", "profit", "upgrade", "strong", "rises", "surge", "record", "contract", "expansion", "demand"]
    negative = ["miss", "loss", "downgrade", "weak", "falls", "drop", "lawsuit", "warning", "cuts", "default", "pressure"]
    output = []
    for article in news:
        text = f"{article['title']} {article['summary']}".lower()
        direct_hits = sum(k in text for k in direct_keys)
        related_hits = sum(k in text for k in related_keys)
        if not direct_hits and not related_hits:
            continue
        age = max(0, (datetime.now(timezone.utc) - article["published_dt"]).total_seconds() / 3600)
        freshness = max(0, 10 - age / 3)
        score = max(0, min(10, 5 + sum(x in text for x in positive) * 1.4 - sum(x in text for x in negative) * 1.6))
        label, tone = ("Bullish", "green") if score >= 6.5 else (("Bearish", "red") if score <= 3.5 else ("Neutral", "yellow"))
        article = dict(article)
        article.update({"direct": bool(direct_hits), "importance": round(min(100, direct_hits * 42 + related_hits * 14 + freshness), 1), "label": label, "tone": tone, "score": round(score, 1)})
        article["reason"] = f"Direct catalyst for {symbol}." if direct_hits else f"Indirect catalyst through {info.get('sector') or info.get('industry') or 'the related sector'}, which can affect {symbol} through demand, costs, rates, regulation, or risk appetite."
        output.append(article)
    return sorted(output, key=lambda x: (x["importance"], x["published_dt"]), reverse=True)


def render_news(article):
    st.markdown(f"""
<div class="news-card {'direct' if article['direct'] else 'indirect'}"><div class="news-head"><span class="badge {article['tone']}">{'DIRECT' if article['direct'] else 'INDIRECT'} · {article['label']}</span><b>Importance {article['importance']:.0f}/100 · Impact {article['score']:.1f}/10</b></div><div class="news-title">{html_module.escape(article['title'])}</div><div class="news-summary">{html_module.escape(article['summary'])}</div><div class="news-reason">🧠 <b>Why it matters:</b> {html_module.escape(article['reason'])}</div><div class="news-meta">📌 {html_module.escape(article['source'])} · 🕒 {html_module.escape(article['published'])} · <a href="{html_module.escape(article['link'])}" target="_blank">Read source →</a></div></div>
""", unsafe_allow_html=True)


def chat_response(prompt, context, model_name):
    system = f"You are an educational trading assistant for {full_symbol}. Use only the supplied context. Explain uncertainty. Context:\n{context}"
    try:
        if model_name in GROQ_MODELS:
            if not groq_client:
                return "AI is unavailable: GROQ_KEY is missing in .streamlit/secrets.toml."
            result = groq_client.chat.completions.create(model=GROQ_MODELS[model_name], messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], temperature=0.2, max_tokens=900)
            return result.choices[0].message.content
        if model_name in GEMINI_MODELS:
            if not GEMINI_KEY:
                return "AI is unavailable: GEMINI_KEY is missing in .streamlit/secrets.toml."
            return genai.GenerativeModel(GEMINI_MODELS[model_name]).generate_content(f"{system}\n\nQuestion: {prompt}").text
        return "AI is unavailable: no valid model was selected."
    except Exception as exc:
        return f"AI request failed ({type(exc).__name__}): {exc}"


# ================== DATA ==================
daily_df, stock_info = get_daily_data(full_symbol)
if daily_df.empty:
    st.error(f"No market data found for {full_symbol}.")
    st.stop()

news_items = ranked_news(harvest_news(), full_symbol, stock_info)
first_intraday = get_intraday_data(full_symbol)
initial_anchor = get_first_15m_open(first_intraday) or float(daily_df["Open"].iloc[-1])
anchor = get_or_set_session_anchor(full_symbol, initial_anchor)


def render_live_analysis():
    live = get_intraday_data(full_symbol)
    if live.empty:
        live = daily_df
    current_price = float(live["Close"].iloc[-1])
    ladder = build_gann_ladder(anchor)
    lower, upper = locate_band(ladder, current_price)
    pivots = calculate_pivots(live if len(live) >= 2 else daily_df)

    st.header(f"📊 Live Analysis: {full_symbol}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live Price", f"{current_price:.2f}")
    m2.metric("0° Open", f"{anchor:.2f}")
    m3.metric("Nearest Lower", lower["label"] if lower else "Below ladder")
    m4.metric("Nearest Upper", upper["label"] if upper else "Above ladder")

    if lower and upper:
        span = upper["value"] - lower["value"]
        from_lower = current_price - lower["value"]
        to_upper = upper["value"] - current_price
        position = from_lower / span if span else 0
        st.success(f"Current price **{current_price:.2f}** is between **{lower['label']} ({lower['value']:.2f})** and **{upper['label']} ({upper['value']:.2f})**")
        st.progress(max(0.0, min(1.0, position)), text=f"{position * 100:.1f}% through this band · {from_lower:.2f} above lower · {to_upper:.2f} below upper")
    elif upper:
        st.warning(f"Current price **{current_price:.2f}** is below the lowest calculated level: **{upper['label']} ({upper['value']:.2f})**")
    else:
        st.warning(f"Current price **{current_price:.2f}** is above the highest calculated level: **{lower['label']} ({lower['value']:.2f})**")

    fig = go.Figure(go.Candlestick(x=live.index, open=live["Open"], high=live["High"], low=live["Low"], close=live["Close"], name=full_symbol, increasing_line_color="#26a269", decreasing_line_color="#ef5350"))
    if lower and upper:
        fig.add_hrect(y0=lower["value"], y1=upper["value"], fillcolor="#f0c75e", opacity=0.12, line_width=0)
    for level in ladder:
        color = "#40c4ff" if level["side"] == "OPEN" else "#ef5350" if level["side"] == "R" else "#26a269"
        fig.add_hline(y=level["value"], line_dash="dot", line_width=1.2, line_color=color, annotation_text=level["label"], annotation_position="right")
    fig.add_hline(y=current_price, line_color="#ffffff", line_width=2, annotation_text=f"LIVE {current_price:.2f}", annotation_position="left")
    fig.update_layout(template="plotly_dark", height=560, hovermode="x unified", xaxis_rangeslider_visible=False, margin=dict(l=10, r=115, t=35, b=20), paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,.08)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')} · 0° anchor stays fixed for this session; market data refreshes every 15 seconds.")

    st.subheader("📐 Complete Gann Ladder")
    st.dataframe(pd.DataFrame(level_rows(ladder, current_price)), use_container_width=True, hide_index=True)
    st.subheader("🎯 Traditional Pivot Levels")
    pivot_rows = [{"Level": label, "Price": round(pivots[key], 2), "Type": "Resistance" if label.startswith("R") else "Support" if label.startswith("S") else "Pivot"} for label, key in [("R4", "r4"), ("R3", "r3"), ("R2", "r2"), ("R1", "r1"), ("Pivot", "pivot"), ("S1", "s1"), ("S2", "s2"), ("S3", "s3"), ("S4", "s4")]]
    st.dataframe(pd.DataFrame(pivot_rows), use_container_width=True, hide_index=True)


try:
    fragment = st.fragment
except AttributeError:
    fragment = None

if fragment:
    @st.fragment(run_every="15s")
    def live_section():
        render_live_analysis()
    live_section()
else:
    render_live_analysis()

# ================== OTHER FEATURES ==================
tab_news, tab_chat, tab_heatmap, tab_portfolio, tab_alerts, tab_history = st.tabs(["📰 Ranked Stock News", "💬 AI Chat", "📊 Market Heatmap", "📋 Watchlist & Portfolio", "🔔 Alerts & Backtest", "📅 Historical Data"])

with tab_news:
    st.header(f"📰 Latest Relevant News — {full_symbol}")
    include_indirect = st.checkbox("Include indirect sector/macro news", True)
    limit = st.slider("Number of ranked stories", 5, 30, 12)
    visible = [x for x in news_items if include_indirect or x["direct"]]
    visible = [x for x in visible if (x["label"] == "Bullish" and "🟢 Bullish" in sentiment_filter) or (x["label"] == "Bearish" and "🔴 Bearish" in sentiment_filter) or (x["label"] == "Neutral" and "🟡 Neutral" in sentiment_filter)]
    if visible:
        for article in visible[:limit]:
            render_news(article)
    else:
        st.info("No relevant news matched the selected filters.")

with tab_chat:
    st.header("💬 AI Chat Assistant")
    model_name = st.selectbox("Choose model", list(GROQ_MODELS) + list(GEMINI_MODELS), key="chat_model")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about price levels, news, or strategy")
    if prompt:
        context = "\n".join(f"{x['title']} | {x['reason']}" for x in news_items[:10])
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = chat_response(prompt, context, model_name)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

with tab_heatmap:
    st.header("📊 Market Heatmap")
    watch = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"] if market == "United States" else ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
    rows = []
    for ticker in watch:
        history, _ = get_daily_data(ticker)
        if len(history) >= 2:
            current = float(history["Close"].iloc[-1])
            change = current / float(history["Close"].iloc[-2]) * 100 - 100
            rows.append({"Stock": ticker, "Price": current, "Change %": change, "Intensity": min(10, abs(change) / 0.3)})
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Change %", ascending=False).style.format({"Price": "{:.2f}", "Change %": "{:.2f}%", "Intensity": "{:.1f}/10"}), use_container_width=True)

with tab_portfolio:
    st.header("📋 Watchlist & Portfolio")
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}
    ticker = st.text_input("Ticker", key="portfolio_ticker")
    quantity = st.number_input("Quantity", min_value=1, value=10, key="portfolio_quantity")
    buy_price = st.number_input("Buy price", min_value=0.01, value=100.0, key="portfolio_buy_price")
    if st.button("Add holding") and ticker:
        st.session_state.portfolio[ticker.upper()] = {"qty": quantity, "buy": buy_price}
    total_pnl = 0.0
    for held, data in st.session_state.portfolio.items():
        history, _ = get_daily_data(held)
        current = float(history["Close"].iloc[-1]) if not history.empty else 0.0
        pnl = (current - data["buy"]) * data["qty"]
        total_pnl += pnl
        st.metric(held, f"Current {current:.2f}", f"P&L {pnl:.2f}")
    if st.session_state.portfolio:
        st.metric("Total P&L", f"{total_pnl:.2f}")

with tab_alerts:
    st.header("🔔 Alerts & Backtest")
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    alert_symbol = st.text_input("Alert ticker", full_symbol, key="alert_symbol")
    alert_price = st.number_input("Alert price", min_value=0.01, value=float(price) if "price" in locals() else 1.0, key="alert_price")
    alert_type = st.selectbox("Trigger", ["Above", "Below"], key="alert_type")
    if st.button("Create alert"):
        st.session_state.alerts.append({"symbol": alert_symbol, "price": alert_price, "type": alert_type})
    for alert in st.session_state.alerts:
        history, _ = get_daily_data(alert["symbol"])
        current = float(history["Close"].iloc[-1]) if not history.empty else 0
        triggered = (alert["type"] == "Above" and current >= alert["price"]) or (alert["type"] == "Below" and current <= alert["price"])
        (st.error if triggered else st.info)(f"{alert['symbol']} {alert['type']} {alert['price']:.2f} · Current {current:.2f}")

with tab_history:
    st.header("📅 Historical Data")
    number_of_days = st.slider("Days", 1, 60, 10)
    history = daily_df.tail(number_of_days).copy()
    history["Change %"] = history["Close"].pct_change() * 100
    st.dataframe(history[["Open", "High", "Low", "Close", "Volume", "Change %"]].sort_index(ascending=False), use_container_width=True)

st.caption("News relevance and market levels are heuristic/AI-assisted and are not investment advice.")
