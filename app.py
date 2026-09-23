import email.utils
import html as html_module
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


def secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


GROQ_KEY = secret("GROQ_KEY")
GEMINI_KEY = secret("GEMINI_KEY")
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
.news-card{border:1px solid rgba(150,150,150,.28);border-radius:14px;padding:15px 16px;margin:0 0 13px;background:rgba(255,255,255,.025)}
.news-card.direct{border-left:5px solid #26a269}.news-card.indirect{border-left:5px solid #e8a317}
.news-head{display:flex;justify-content:space-between;gap:10px;align-items:center}.news-title{font-size:1.05rem;font-weight:700;line-height:1.4;margin:7px 0}.news-summary,.news-reason{line-height:1.55;opacity:.88}.news-meta{font-size:.78rem;opacity:.68;margin-top:9px}.badge{padding:5px 10px;border-radius:99px;font-size:.75rem;font-weight:700}.green{background:#164d35;color:#6ee7a0}.yellow{background:#604811;color:#ffd166}.red{background:#5b2020;color:#ff8b8b}
.zone{border:1px solid rgba(140,140,140,.3);border-radius:15px;padding:15px;margin:10px 0}.zone-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.stat{padding:9px;border-radius:9px;background:rgba(0,0,0,.08)}.small{font-size:.75rem;opacity:.7}.value{font-weight:700;margin-top:3px}@media(max-width:640px){h1{font-size:1.4rem!important}.zone-grid{grid-template-columns:1fr}.news-title{font-size:1rem}}
</style>
""", unsafe_allow_html=True)
st.title("🎯 AI Trading Terminal - PRO")


def yahoo_search(query):
    try:
        response = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=6,
        )
        response.raise_for_status()
        return [
            {"symbol": item["symbol"], "name": item.get("shortname") or item.get("longname") or item["symbol"], "exchange": item.get("exchange", ""), "type": item.get("quoteType", "")}
            for item in response.json().get("quotes", []) if item.get("symbol")
        ]
    except Exception:
        return []


st.sidebar.header("🔍 Search Any Stock / Index / Crypto")
query = st.sidebar.text_input("Company name or ticker", "RELIANCE", key="global_search_query").strip().upper()
search_results = yahoo_search(query) if query else []
market = "India - NSE (.NS)"
if search_results:
    options = [f"{item['symbol']} — {item['name']} ({item['exchange']} · {item['type']})" for item in search_results]
    selected = st.sidebar.radio("Matching symbols", options, key="global_search_pick")
    symbol = search_results[options.index(selected)]["symbol"]
else:
    market = st.sidebar.selectbox("Market", ["India - NSE (.NS)", "United States", "Crypto", "Forex", "Commodities"])
    raw = st.sidebar.text_input("Exact ticker", query).strip().upper()
    suffix = {"India - NSE (.NS)": ".NS", "Crypto": "-USD", "Forex": "=X", "Commodities": "=F"}.get(market, "")
    symbol = raw if not suffix or raw.endswith(suffix) else raw + suffix

sentiment_filter = st.sidebar.multiselect("Sentiment filter", ["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"], default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"])


def clean_text(value):
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def news_datetime(value):
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
                link = (item.findtext("link", "") or "").strip()
                if not title:
                    continue
                published = (item.findtext("pubDate", "") or "").strip()
                key = link or re.sub(r"[^a-z0-9]", "", title.lower())
                unique.setdefault(key, {"title": title, "summary": clean_text(item.findtext("description", "")) or title, "link": link, "source": source, "published": published, "published_dt": news_datetime(published)})
        except Exception as exc:
            print(f"News source error ({source}): {exc}")
    return list(unique.values())


@st.cache_data(ttl=120)
def get_stock_data(ticker):
    try:
        ticker_data = yf.Ticker(ticker)
        return ticker_data.history(period="3mo", interval="1d"), ticker_data.info or {}
    except Exception:
        return pd.DataFrame(), {}


def stock_keys(ticker, info):
    raw = re.sub(r"(\.NS|-USD|=X|=F)$", "", ticker.lower())
    name = (info.get("longName") or info.get("shortName") or "").lower()
    words = re.findall(r"[a-z0-9]+", name)
    return [value for value in {raw, name, *[word for word in words if len(word) > 2]} if value]


def indirect_keys(ticker, info):
    text = f"{ticker} {info.get('longName','')} {info.get('sector','')} {info.get('industry','')}".lower()
    groups = {
        "technology": ["technology", "software", "cloud", "semiconductor", "chip", "artificial intelligence", "ai"],
        "financial": ["bank", "banking", "finance", "interest rate", "credit", "loan", "insurance"],
        "energy": ["oil", "gas", "energy", "refinery", "crude", "lng", "solar", "renewable"],
        "automotive": ["auto", "automobile", "vehicle", "ev", "electric vehicle", "car", "motorcycle"],
        "pharma": ["pharma", "drug", "medicine", "healthcare", "fda", "clinical", "biotech"],
        "consumer": ["consumer", "retail", "fmcg", "inflation"],
        "metals": ["metal", "steel", "aluminium", "copper", "mining"],
        "crypto": ["crypto", "bitcoin", "ethereum", "blockchain"],
    }
    related = set()
    for words in groups.values():
        if any(word in text for word in words):
            related.update(words)
    return list(related)


def rank_news(article, direct_keys, related_keys):
    text = f"{article['title']} {article['summary']}".lower()
    direct_hits = sum(key in text for key in direct_keys)
    related_hits = sum(key in text for key in related_keys)
    catalysts = ["earnings", "revenue", "profit", "guidance", "regulator", "lawsuit", "acquisition", "merger", "tariff", "rate", "inflation", "sanction", "upgrade", "downgrade", "contract", "default", "warning", "forecast", "results"]
    catalyst_hits = sum(word in text for word in catalysts)
    age_hours = max(0, (datetime.now(timezone.utc) - article["published_dt"]).total_seconds() / 3600)
    freshness = max(0, 10 - min(age_hours / 3, 10))
    source_weight = {"CNBC": 1.7, "Economic Times": 1.5, "Moneycontrol": 1.4, "Yahoo Finance": 1.4, "MarketWatch": 1.3}.get(article["source"], 1)
    if not direct_hits and not related_hits:
        return None
    result = dict(article)
    result.update({"direct": direct_hits > 0, "importance": round(min(100, 42 * min(direct_hits, 2) + 14 * min(related_hits, 3) + 7 * min(catalyst_hits, 3) + freshness + source_weight * 4), 1)})
    return result


def impact(article, ticker, info):
    text = f"{article['title']} {article['summary']}".lower()
    positive = ["beat", "growth", "profit", "upgrade", "strong", "rises", "surge", "record", "contract", "expansion", "demand", "guidance raised"]
    negative = ["miss", "loss", "downgrade", "weak", "falls", "drop", "lawsuit", "warning", "cuts", "default", "pressure", "guidance cut"]
    score = max(0, min(10, 5 + sum(word in text for word in positive) * 1.4 - sum(word in text for word in negative) * 1.6))
    tone, label = ("green", "Bullish") if score >= 6.5 else (("red", "Bearish") if score <= 3.5 else ("yellow", "Neutral"))
    if article["direct"]:
        reason = f"Direct catalyst: the story mentions {ticker} or its company name and may affect earnings, valuation, demand, or risk sentiment directly."
    else:
        sector = info.get("sector") or info.get("industry") or "the company’s industry"
        reason = f"Indirect catalyst: this relates to {sector} or a macro factor. It may affect {ticker} through demand, costs, rates, regulation, currency, or investor risk appetite."
    return {"score": round(score, 1), "tone": tone, "label": label, "reason": reason}


def render_news(article):
    sentiment = article["impact"]
    relation = "DIRECT" if article["direct"] else "INDIRECT"
    st.markdown(f"""
<div class="news-card {'direct' if article['direct'] else 'indirect'}"><div class="news-head"><span class="badge {sentiment['tone']}">{relation} · {sentiment['label']}</span><b>Importance {article['importance']:.0f}/100 · Impact {sentiment['score']:.1f}/10</b></div><div class="news-title">{html_module.escape(article['title'])}</div><div class="news-summary">{html_module.escape(article['summary'])}</div><div class="news-reason">🧠 <b>Why it matters:</b> {html_module.escape(sentiment['reason'])}</div><div class="news-meta">📌 {html_module.escape(article['source'])} · 🕒 {html_module.escape(article['published'])} · <a href="{html_module.escape(article['link'])}" target="_blank">Read source →</a></div></div>
""", unsafe_allow_html=True)


df, info = get_stock_data(symbol)
if df.empty:
    st.error(f"No market data found for {symbol}.")
    st.stop()

price = float(df["Close"].iloc[-1])
all_news = harvest_news()
direct_keys = stock_keys(symbol, info)
related_keys = indirect_keys(symbol, info)
ranked_news = [rank_news(article, direct_keys, related_keys) for article in all_news]
ranked_news = [article for article in ranked_news if article]
for article in ranked_news:
    article["impact"] = impact(article, symbol, info)
ranked_news.sort(key=lambda article: (article["importance"], article["published_dt"]), reverse=True)

# ================== TABS ==================
tab_live, tab_news, tab_chat, tab_heatmap, tab_portfolio, tab_alerts, tab_history = st.tabs(["📊 Live Analysis", "📰 Ranked Stock News", "💬 AI Chat", "📊 Market Heatmap", "📋 Watchlist & Portfolio", "🔔 Alerts & Backtest", "📅 Historical Data"])

with tab_live:
    st.header(f"📊 Live Analysis: {symbol}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{price:.2f}")
    c2.metric("Company", info.get("longName", symbol))
    c3.metric("Related stories", len(ranked_news))
    chart = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol))
    chart.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(chart, use_container_width=True)
    st.subheader("🧠 Smart Money Zones")
    recent = df.tail(60)
    if len(recent) >= 15:
        high, low = float(recent["High"].max()), float(recent["Low"].min())
        gp_low, gp_high = low + (high - low) * 0.618, low + (high - low) * 0.65
        trend = "Bullish" if recent["Close"].iloc[-1] >= recent["Close"].iloc[0] else "Bearish"
        st.markdown(f"<div class='zone'><b>🌀 Golden Pocket · {trend}</b><div class='zone-grid'><div class='stat'><div class='small'>Current</div><div class='value'>{price:.2f}</div></div><div class='stat'><div class='small'>Zone low</div><div class='value'>{gp_low:.2f}</div></div><div class='stat'><div class='small'>Zone high</div><div class='value'>{gp_high:.2f}</div></div></div><p>Recent swing retracement zone. {'Price is inside the zone.' if gp_low <= price <= gp_high else 'Price is outside the zone.'}</p></div>", unsafe_allow_html=True)

with tab_news:
    st.header(f"📰 Latest Relevant News — {symbol}")
    st.caption("Fresh, deduplicated articles from all feeds, ranked by direct relevance, indirect sector relevance, catalyst importance, source quality, and freshness.")
    include_indirect = st.checkbox("Include indirect sector/macro news", True)
    limit = st.slider("Number of ranked stories", 5, 30, 12)
    shown = [article for article in ranked_news if include_indirect or article["direct"]]
    shown = [article for article in shown if (article["impact"]["label"] == "Bullish" and "🟢 Bullish" in sentiment_filter) or (article["impact"]["label"] == "Bearish" and "🔴 Bearish" in sentiment_filter) or (article["impact"]["label"] == "Neutral" and "🟡 Neutral" in sentiment_filter)]
    if shown:
        for article in shown[:limit]:
            render_news(article)
    else:
        st.warning("No relevant stories matched the current filters.")

with tab_chat:
    st.header("💬 AI Chat Assistant")
    model_name = st.selectbox("Choose model", list(GROQ_MODELS) + list(GEMINI_MODELS))
    prompt = st.chat_input("Ask about this stock, levels, news, or strategy")
    if prompt:
        context = "\n".join(f"{article['title']} | {article['impact']['reason']}" for article in ranked_news[:10])
        try:
            if model_name in GROQ_MODELS and groq_client:
                answer = groq_client.chat.completions.create(model=GROQ_MODELS[model_name], messages=[{"role": "system", "content": f"You are an educational trading assistant for {symbol}. Context:\n{context}"}, {"role": "user", "content": prompt}], temperature=0.2, max_tokens=900).choices[0].message.content
            elif model_name in GEMINI_MODELS and GEMINI_KEY:
                answer = genai.GenerativeModel(GEMINI_MODELS[model_name]).generate_content(f"You are an educational trading assistant for {symbol}. Context:\n{context}\nQuestion: {prompt}").text
            else:
                answer = "Configure the selected model API key in .streamlit/secrets.toml."
            st.chat_message("assistant").markdown(answer)
        except Exception as exc:
            st.error(f"AI error: {exc}")

with tab_heatmap:
    st.header("📊 Market Heatmap")
    # Fixed syntax: a single valid if/else expression.
    watch = (["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"] if market == "United States" else ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"])
    rows = []
    for ticker in watch:
        try:
            history, _ = get_stock_data(ticker)
            if len(history) < 2:
                continue
            current = float(history["Close"].iloc[-1])
            change = (current / float(history["Close"].iloc[-2]) - 1) * 100
            rows.append({"Stock": ticker, "Price": current, "Change %": change, "Intensity": min(10, abs(change) / 0.3)})
        except Exception:
            continue
    if rows:
        heatmap = pd.DataFrame(rows).sort_values("Change %", ascending=False)
        st.dataframe(heatmap.style.format({"Price": "{:.2f}", "Change %": "{:.2f}%", "Intensity": "{:.1f}/10"}), use_container_width=True)
    else:
        st.warning("Could not fetch heatmap data.")

with tab_portfolio:
    st.header("📋 Watchlist & Portfolio")
    if "watchlists" not in st.session_state:
        st.session_state.watchlists = {"My Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS"], "US": ["AAPL", "NVDA"], "Crypto": ["BTC-USD", "ETH-USD"]}
    st.write(st.session_state.watchlists)
    st.subheader("💼 Portfolio Tracker")
    holding = st.text_input("Ticker", key="holding_ticker")
    quantity = st.number_input("Quantity", min_value=1, max_value=100000, value=10, key="holding_qty")
    buy_price = st.number_input("Buy price", min_value=0.01, max_value=1000000.0, value=100.0, key="holding_buy")
    if st.button("Add holding") and holding:
        st.session_state.setdefault("portfolio", {})[holding.upper()] = {"qty": quantity, "buy": buy_price}
    total_pnl = 0.0
    for ticker, holding_data in st.session_state.get("portfolio", {}).items():
        history, _ = get_stock_data(ticker)
        current = float(history["Close"].iloc[-1]) if not history.empty else 0.0
        value = current * holding_data["qty"]
        pnl = value - holding_data["buy"] * holding_data["qty"]
        total_pnl += pnl
        st.metric(ticker, f"{value:.2f}", f"P&L {pnl:.2f}")
    if st.session_state.get("portfolio"):
        st.metric("Total P&L", f"{total_pnl:.2f}")

with tab_alerts:
    st.header("🔔 Alerts & Backtest")
    alert_ticker = st.text_input("Alert ticker", symbol, key="alert_ticker")
    alert_price = st.number_input("Alert price", min_value=0.01, max_value=1000000.0, value=price, key="alert_price")
    alert_type = st.selectbox("Trigger", ["Above", "Below"], key="alert_type")
    if st.button("Create price alert"):
        st.session_state.setdefault("alerts", []).append({"ticker": alert_ticker, "price": alert_price, "type": alert_type})
    for alert in st.session_state.get("alerts", []):
        history, _ = get_stock_data(alert["ticker"])
        current = float(history["Close"].iloc[-1]) if not history.empty else 0.0
        triggered = (alert["type"] == "Above" and current >= alert["price"]) or (alert["type"] == "Below" and current <= alert["price"])
        (st.error if triggered else st.info)(f"{alert['ticker']} {alert['type']} {alert['price']:.2f} | Current {current:.2f}")
    st.subheader("🧪 Backtesting")
    backtest_ticker = st.text_input("Backtest ticker", symbol, key="bt_ticker")
    strategy = st.selectbox("Strategy", ["Buy at S1", "Sell at R1"], key="bt_strategy")
    start = st.date_input("Start", datetime.now() - timedelta(days=180), key="bt_start")
    end = st.date_input("End", datetime.now(), key="bt_end")
    if st.button("Run backtest"):
        history, _ = get_stock_data(backtest_ticker)
        history = history[(history.index.date >= start) & (history.index.date <= end)]
        trades, profit = 0, 0.0
        for index in range(1, len(history)):
            pivot = (history.High.iloc[index - 1] + history.Low.iloc[index - 1] + history.Close.iloc[index - 1]) / 3
            level = (2 * pivot - history.High.iloc[index - 1]) if strategy == "Buy at S1" else (2 * pivot - history.Low.iloc[index - 1])
            if (strategy == "Buy at S1" and history.Low.iloc[index] <= level) or (strategy == "Sell at R1" and history.High.iloc[index] >= level):
                trades += 1
                profit += pivot - level
        st.success(f"Trades: {trades} · Approx. total: {profit:.2f}")

with tab_history:
    st.header("📅 Historical Data")
    number_of_days = st.slider("Days", 1, 60, 10)
    history = df.tail(number_of_days).copy()
    history["Change %"] = history["Close"].pct_change() * 100
    st.dataframe(history[["Open", "High", "Low", "Close", "Volume", "Change %"]].sort_index(ascending=False), use_container_width=True)

st.caption("News relevance and sentiment are heuristic/AI-assisted and are not investment advice.")
