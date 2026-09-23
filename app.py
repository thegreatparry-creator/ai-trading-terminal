import email.utils
import html as html_module
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

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


GROQ_KEY, GEMINI_KEY = secret("GROQ_KEY"), secret("GEMINI_KEY")
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = secret("TELEGRAM_BOT_TOKEN"), secret("TELEGRAM_CHAT_ID")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

GROQ_MODELS = {
    "Groq GPT-OSS 20B": "openai/gpt-oss-20b",
    "Groq GPT-OSS 120B": "openai/gpt-oss-120b",
    "Groq Qwen3 32B": "qwen/qwen3-32b",
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
.news-head{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.news-title{font-size:1.05rem;font-weight:700;line-height:1.4;margin:7px 0}.news-summary,.news-reason{line-height:1.55;opacity:.88}.news-meta{font-size:.78rem;opacity:.68;margin-top:9px}.badge{padding:5px 10px;border-radius:99px;font-size:.75rem;font-weight:700}.green{background:#164d35;color:#6ee7a0}.yellow{background:#604811;color:#ffd166}.red{background:#5b2020;color:#ff8b8b}
.zone{border:1px solid rgba(140,140,140,.3);border-radius:15px;padding:15px;margin:10px 0}.zone-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.stat{padding:9px;border-radius:9px;background:rgba(0,0,0,.08)}.small{font-size:.75rem;opacity:.7}.value{font-weight:700;margin-top:3px}
@media(max-width:640px){h1{font-size:1.4rem!important}.zone-grid{grid-template-columns:1fr}.news-title{font-size:1rem}}
</style>
""", unsafe_allow_html=True)
st.title("🎯 AI Trading Terminal - PRO")


def clean(value):
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def yahoo_search(query, limit=8):
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": query, "quotesCount": limit, "newsCount": 0}, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        r.raise_for_status()
        return [x for x in r.json().get("quotes", []) if x.get("symbol")]
    except Exception:
        return []


st.sidebar.header("🔍 Search Any Stock / Index / Crypto")
query = st.sidebar.text_input("Company name or ticker", "RELIANCE", key="global_search_query").strip().upper()
searches = yahoo_search(query) if query else []
if searches:
    labels = [f"{x['symbol']} — {x.get('shortname') or x.get('longname') or x['symbol']} ({x.get('exchange','')} · {x.get('quoteType','')})" for x in searches]
    picked = st.sidebar.radio("Matching symbols", labels, key="global_search_pick")
    symbol = searches[labels.index(picked)]["symbol"]
else:
    market = st.sidebar.selectbox("Market", ["India - NSE (.NS)", "United States", "Crypto", "Forex", "Commodities"])
    raw = st.sidebar.text_input("Exact ticker", query).strip().upper()
    suffix = {"India - NSE (.NS)": ".NS", "Crypto": "-USD", "Forex": "=X", "Commodities": "=F"}.get(market, "")
    symbol = raw if not suffix or raw.endswith(suffix) else raw + suffix

sentiment_filter = st.sidebar.multiselect("Sentiment filter", ["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"], default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"])


@st.cache_data(ttl=120)
def get_stock(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.history(period="3mo", interval="1d"), (t.info or {})
    except Exception:
        return pd.DataFrame(), {}


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
                unique.setdefault(key, {"title": title, "summary": clean(item.findtext("description", "")) or title, "link": link, "source": source, "published": published, "dt": dt})
        except Exception as exc:
            print(f"News source error ({source}): {exc}")
    return list(unique.values())


def keywords(ticker, info):
    raw = re.sub(r"(\.NS|-USD|=X|=F)$", "", ticker.lower())
    name = (info.get("longName") or info.get("shortName") or "").lower()
    return [x for x in {raw, name, *re.findall(r"[a-z0-9]+", name)} if len(x) >= 3]


def sector_words(ticker, info):
    text = f"{ticker} {info.get('longName','')} {info.get('sector','')} {info.get('industry','')}".lower()
    groups = ["technology", "software", "cloud", "semiconductor", "chip", "bank", "finance", "interest rate", "credit", "oil", "gas", "energy", "crude", "auto", "vehicle", "ev", "pharma", "drug", "healthcare", "fda", "consumer", "retail", "fmcg", "metal", "steel", "copper", "mining", "crypto", "bitcoin", "blockchain", "inflation", "tariff", "currency", "regulation"]
    return [w for w in groups if w in text]


def rank_article(article, direct_keys, indirect_keys):
    text = f"{article['title']} {article['summary']}".lower()
    d_hits, i_hits = sum(k in text for k in direct_keys), sum(k in text for k in indirect_keys)
    if not d_hits and not i_hits:
        return None
    catalysts = ["earnings", "revenue", "profit", "guidance", "regulator", "lawsuit", "acquisition", "merger", "tariff", "rate", "inflation", "sanction", "upgrade", "downgrade", "contract", "warning", "forecast", "results"]
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
    prompt = f"Explain the indirect connection of each news item to {ticker} ({info.get('longName','')}). Return one concise line as INDEX|EXPLANATION. Do not invent facts.\n{body}"
    try:
        answer = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": prompt}], temperature=.1, max_tokens=900).choices[0].message.content
        return {int(m.group(1)): m.group(2).strip() for line in answer.splitlines() if (m := re.match(r"\s*(\d+)\s*[|:-]\s*(.+)", line))}
    except Exception:
        return {}


def render_news(article):
    x = article["impact"]
    kind = "direct" if article["direct"] else "indirect"
    st.markdown(f"<div class='news-card {kind}'><div class='news-head'><span class='badge {x['tone']}'>{'DIRECT' if article['direct'] else 'INDIRECT'} · {x['label']}</span><b>Importance {article['importance']:.0f}/100 · Impact {x['score']:.1f}/10</b></div><div class='news-title'>{html_module.escape(article['title'])}</div><div class='news-summary'>{html_module.escape(article['summary'])}</div><div class='news-reason'>🧠 <b>Why it matters:</b> {html_module.escape(x['reason'])}</div><div class='news-meta'>📌 {html_module.escape(article['source'])} · 🕒 {html_module.escape(article['published'])} · <a href='{html_module.escape(article['link'])}' target='_blank'>Read source →</a></div></div>", unsafe_allow_html=True)


def levels(data):
    high, low, close = map(float, [data.High.iloc[-1], data.Low.iloc[-1], data.Close.iloc[-1]])
    p = (high + low + close) / 3
    return {"price": close, "pivot": round(p, 2), "r1": round(2*p-low, 2), "r2": round(p+high-low, 2), "r3": round(high+2*(p-low), 2), "r4": round(high+3*(p-low), 2), "s1": round(2*p-high, 2), "s2": round(p-high+low, 2), "s3": round(low-2*(high-p), 2), "s4": round(low-3*(high-p), 2)}


def backtest(ticker, start, end, strategy):
    try:
        data = yf.Ticker(ticker).history(start=start, end=end, interval="1d"); trades = []
        for i in range(1, len(data)):
            p = (data.High.iloc[i-1]+data.Low.iloc[i-1]+data.Close.iloc[i-1])/3; s1, r1 = 2*p-data.High.iloc[i-1], 2*p-data.Low.iloc[i-1]
            if strategy == "Buy at S1" and data.Low.iloc[i] <= s1: trades.append((s1, p))
            if strategy == "Sell at R1" and data.High.iloc[i] >= r1: trades.append((r1, p))
        profits = [x-e for e,x in trades]
        return {"Trades": len(profits), "Win %": round(sum(x > 0 for x in profits)/len(profits)*100, 1) if profits else 0, "Average profit": round(sum(profits)/len(profits), 2) if profits else 0, "Total profit": round(sum(profits), 2)}
    except Exception:
        return None


df, info = get_stock(symbol)
if df.empty:
    st.error(f"No market data found for {symbol}.")
    st.stop()
price = float(df.Close.iloc[-1])
articles = []
for raw in get_news():
    item = rank_article(raw, keywords(symbol, info), sector_words(symbol, info))
    if item:
        item["impact"] = impact(item, symbol, info); articles.append(item)
articles.sort(key=lambda x: (x["importance"], x["dt"]), reverse=True)
indirect = [x for x in articles if not x["direct"]][:10]
for idx, text in explain_indirect(indirect, symbol, info).items():
    if 0 <= idx < len(indirect): indirect[idx]["impact"]["reason"] = text

# ================== TABS ==================
tab_live, tab_news, tab_chat, tab_heat, tab_watch, tab_alerts, tab_history = st.tabs(["📊 Live Analysis", "📰 Ranked Stock News", "💬 AI Chat", "📊 Market Heatmap", "📋 Watchlist & Portfolio", "🔔 Alerts & Backtest", "📅 Historical Data"])

with tab_live:
    st.header(f"📊 Live Analysis: {symbol}")
    a, b, c = st.columns(3); a.metric("Current Price", f"{price:.2f}"); b.metric("Company", info.get("longName", symbol)); c.metric("Related stories", len(articles))
    fig = go.Figure(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=symbol)); fig.update_layout(height=460, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)
    st.subheader("🧠 Smart Money Zones")
    recent = df.tail(60)
    if len(recent) >= 15:
        hi, lo = float(recent.High.max()), float(recent.Low.min()); gp_low, gp_high = lo+(hi-lo)*.618, lo+(hi-lo)*.65
        trend = "Bullish" if recent.Close.iloc[-1] >= recent.Close.iloc[0] else "Bearish"
        st.markdown(f"<div class='zone'><b>🌀 Golden Pocket · {trend}</b><div class='zone-grid'><div class='stat'><div class='small'>Current</div><div class='value'>{price:.2f}</div></div><div class='stat'><div class='small'>Zone low</div><div class='value'>{gp_low:.2f}</div></div><div class='stat'><div class='small'>Zone high</div><div class='value'>{gp_high:.2f}</div></div></div><p>Recent swing retracement zone. {'Price is inside the zone.' if gp_low <= price <= gp_high else 'Price is outside the zone.'}</p></div>", unsafe_allow_html=True)
    q = levels(df); st.subheader("🎯 Traditional Pivot Levels"); st.dataframe(pd.DataFrame([{"Level": k.upper(), "Value": q[k]} for k in ["r4","r3","r2","r1","pivot","s1","s2","s3","s4"]]), use_container_width=True)

with tab_news:
    st.header(f"📰 Latest Relevant News — {symbol}"); st.caption("Fresh, deduplicated stories from all feeds, ranked by direct/indirect relevance, importance, and freshness.")
    include_indirect = st.checkbox("Include indirect sector/macro news", True); limit = st.slider("Number of ranked stories", 5, 30, 12)
    shown = [x for x in articles if include_indirect or x["direct"]]
    shown = [x for x in shown if (x["impact"]["label"] == "Bullish" and "🟢 Bullish" in sentiment_filter) or (x["impact"]["label"] == "Bearish" and "🔴 Bearish" in sentiment_filter) or (x["impact"]["label"] == "Neutral" and "🟡 Neutral" in sentiment_filter)]
    if shown:
        for article in shown[:limit]: render_news(article)
    else: st.warning("No relevant stories matched the current filters.")

with tab_chat:
    st.header("💬 AI Chat Assistant"); model_choice = st.selectbox("Choose Model", list(GROQ_MODELS) + list(GEMINI_MODELS)); prompt = st.chat_input("Ask about stock, levels, news, strategy...")
    if prompt:
        context = "\n".join(f"{x['title']} | {x['impact']['reason']}" for x in articles[:10])
        try:
            if model_choice in GROQ_MODELS and groq_client:
                answer = groq_client.chat.completions.create(model=GROQ_MODELS[model_choice], messages=[{"role":"system","content":f"Trading assistant for {symbol}. Educational only. Context:\n{context}"},{"role":"user","content":prompt}], temperature=.2, max_tokens=900).choices[0].message.content
            elif model_choice in GEMINI_MODELS and GEMINI_KEY:
                answer = genai.GenerativeModel(GEMINI_MODELS[model_choice]).generate_content(f"Trading assistant for {symbol}. Context:{context}\nQuestion:{prompt}").text
            else: answer = "⚠️ Configure the selected API key in .streamlit/secrets.toml."
            st.markdown(answer)
        except Exception as exc: st.error(f"AI error: {exc}")

with tab_heat:
    st.header("📊 Market Heatmap"); groups = {"Technology":["AAPL","MSFT","NVDA","GOOGL","META"],"Banking":["JPM","BAC","GS","MS"],"Energy":["XLE","CL=F","GC=F"],"Crypto":["BTC-USD","ETH-USD","SOL-USD"]}; group = st.selectbox("Select group", list(groups)); rows=[]
    for ticker in groups[group]:
        try:
            fast = yf.Ticker(ticker).fast_info; last=float(fast.get("last_price",0)); prev=float(fast.get("previous_close",last)); change=(last-prev)/prev*100 if prev else 0; rows.append({"Asset":ticker,"Price":last,"Change %":round(change,2),"Intensity":f"{min(10,abs(change)/.3):.1f}/10"})
        except Exception: pass
    st.dataframe(pd.DataFrame(rows).sort_values("Change %", ascending=False) if rows else pd.DataFrame(), use_container_width=True)

with tab_watch:
    st.header("📋 Watchlist & Portfolio"); st.session_state.setdefault("watchlist", ["RELIANCE.NS","TCS.NS","INFY.NS","AAPL","NVDA"]); add=st.text_input("Add ticker", key="watch_add")
    if st.button("➕ Add", key="watch_add_button") and add and add.upper() not in st.session_state.watchlist: st.session_state.watchlist.append(add.upper())
    rows=[]
    for ticker in st.session_state.watchlist:
        try:
            fast=yf.Ticker(ticker).fast_info; last=float(fast.get("last_price",0)); prev=float(fast.get("previous_close",last)); rows.append({"Ticker":ticker,"Price":last,"Change %":round((last-prev)/prev*100 if prev else 0,2)})
        except Exception: rows.append({"Ticker":ticker,"Price":"N/A","Change %":"N/A"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.subheader("💼 Portfolio Tracker"); st.session_state.setdefault("portfolio", {}); p1,p2,p3=st.columns(3)
    with p1: pticker=st.text_input("Ticker",key="portfolio_ticker")
    with p2: pqty=st.number_input("Quantity",min_value=1,value=1,key="portfolio_qty")
    with p3: pbuy=st.number_input("Buy price",min_value=.01,value=100.,key="portfolio_buy")
    if st.button("Add holding",key="portfolio_add") and pticker: st.session_state.portfolio[pticker.upper()]={"qty":pqty,"buy":pbuy}
    for ticker,data in st.session_state.portfolio.items():
        try:
            last=float(yf.Ticker(ticker).fast_info.get("last_price",0)); pnl=(last-data["buy"])*data["qty"]; st.metric(ticker,f"{last:.2f}",f"P&L {pnl:.2f}")
        except Exception: st.warning(f"Could not fetch {ticker}")

with tab_alerts:
    st.header("🔔 Alerts & Backtest"); st.session_state.setdefault("alerts", []); x1,x2,x3=st.columns(3)
    with x1: aticker=st.text_input("Alert ticker",key="alert_ticker")
    with x2: avalue=st.number_input("Alert price",min_value=.01,value=100.,key="alert_value")
    with x3: adirection=st.selectbox("Condition",["Above","Below"],key="alert_direction")
    if st.button("Create alert",key="create_alert") and aticker: st.session_state.alerts.append({"ticker":aticker.upper(),"price":avalue,"direction":adirection})
    for alert in st.session_state.alerts:
        try:
            last=float(yf.Ticker(alert["ticker"]).fast_info.get("last_price",0)); hit=(alert["direction"]=="Above" and last>=alert["price"]) or (alert["direction"]=="Below" and last<=alert["price"]); (st.error if hit else st.info)(f"{alert['ticker']} {alert['direction']} {alert['price']:.2f} · Current {last:.2f}")
        except Exception: st.warning(f"Could not fetch {alert['ticker']}")
    st.subheader("🧪 Backtesting"); b1,b2=st.columns(2)
    with b1: bticker=st.text_input("Backtest ticker",symbol,key="bticker")
    with b2: strategy=st.selectbox("Strategy",["Buy at S1","Sell at R1"],key="bstrategy")
    start=st.date_input("Start",datetime.now()-timedelta(days=90),key="bstart"); end=st.date_input("End",datetime.now(),key="bend")
    if st.button("🚀 Run Backtest",key="run_backtest"):
        result=backtest(bticker,start,end,strategy); st.json(result) if result else st.error("Backtest failed")

with tab_history:
    st.header("📅 Historical Data"); period=st.selectbox("Period",["1mo","3mo","6mo","1y"],key="history_period"); hist=yf.Ticker(symbol).history(period=period,interval="1d")
    if hist.empty: st.warning("No historical data")
    else: st.dataframe(hist[["Open","High","Low","Close","Volume"]].tail(100),use_container_width=True); st.line_chart(hist["Close"])

st.caption("News relevance, sentiment, and impact explanations are heuristic/AI-assisted and are not investment advice.")
