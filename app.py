import email.utils
import html as html_module
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import google.generativeai as genai
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from groq import Groq


# ================== CONFIGURATION ==================
st.set_page_config(page_title="AI Institutional Terminal", layout="wide")


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
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

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
.zone{border:1px solid rgba(140,140,140,.3);border-radius:15px;padding:15px;margin:10px 0}.zone-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.stat{padding:9px;border-radius:9px;background:rgba(0,0,0,.08)}.small{font-size:.75rem;opacity:.7}.value{font-weight:700;margin-top:3px}
@media(max-width:640px){h1{font-size:1.4rem!important}.zone-grid{grid-template-columns:1fr}.news-title{font-size:1rem}}
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Trading Terminal - PRO")


# ================== SYMBOL SELECTION ==================
def yahoo_symbol_search(query, limit=8):
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": query, "quotesCount": limit, "newsCount": 0}, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        r.raise_for_status()
        return [{"symbol": x.get("symbol"), "name": x.get("shortname") or x.get("longname") or x.get("symbol"), "exchange": x.get("exchange", ""), "type": x.get("quoteType", "")} for x in r.json().get("quotes", []) if x.get("symbol")]
    except Exception:
        return []


st.sidebar.header("🔍 Search Any Stock / Index / Crypto")
query = st.sidebar.text_input("Company name or ticker", "RELIANCE", key="global_search_query").strip().upper()
results = yahoo_symbol_search(query) if query else []
if results:
    choices = [f"{x['symbol']} — {x['name']} ({x['exchange']} · {x['type']})" for x in results]
    selected = st.sidebar.radio("Matching symbols", choices, key="global_search_pick")
    symbol = results[choices.index(selected)]["symbol"]
else:
    market = st.sidebar.selectbox("Market", ["India - NSE (.NS)", "United States", "Crypto", "Forex", "Commodities"])
    raw = st.sidebar.text_input("Exact ticker", query).strip().upper()
    suffix = {"India - NSE (.NS)": ".NS", "Crypto": "-USD", "Forex": "=X", "Commodities": "=F"}.get(market, "")
    symbol = raw if (not suffix or raw.endswith(suffix)) else raw + suffix

st.sidebar.markdown("---")
sentiment_filter = st.sidebar.multiselect("Sentiment filter", ["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"], default=["🟢 Bullish", "🔴 Bearish", "🟡 Neutral"])


# ================== DATA ==================
def clean_text(value):
    text = html_module.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    return re.sub(r"\s+", " ", text).strip()


def parse_news_time(value):
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=7)


@st.cache_data(ttl=180, show_spinner=False)
def harvest_news():
    """Read a fresh, wide pool from every configured source and remove duplicates."""
    unique = {}
    for source_name, url in NEWS_SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as response:
                root = ET.fromstring(response.read())
            for item in root.findall(".//item")[:20]:
                title = clean_text(item.findtext("title", ""))
                link = (item.findtext("link", "") or "").strip()
                summary = clean_text(item.findtext("description", ""))
                published = (item.findtext("pubDate", "") or "").strip()
                if not title:
                    continue
                key = link or re.sub(r"[^a-z0-9]", "", title.lower())
                unique.setdefault(key, {"title": title, "summary": summary or title, "link": link, "source": source_name, "published": published, "published_dt": parse_news_time(published)})
        except Exception as exc:
            print(f"News source error ({source_name}): {exc}")
    return list(unique.values())


@st.cache_data(ttl=120)
def stock_data(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.history(period="3mo", interval="1d"), t.info or {}
    except Exception:
        return None, {}


def stock_keywords(ticker, info):
    raw = re.sub(r"(\.NS|-USD|=X|=F)$", "", ticker.lower())
    name = (info.get("longName") or info.get("shortName") or "").lower()
    words = re.findall(r"[a-z0-9]+", name)
    values = {raw, name, *[w for w in words if len(w) >= 3]}
    return [x for x in values if x and len(x) >= 2]


def related_sector_keywords(ticker, info):
    text = f"{ticker} {info.get('longName','')} {info.get('sector','')} {info.get('industry','')}".lower()
    groups = {
        "technology": ["technology", "software", "cloud", "semiconductor", "chip", "it services", "artificial intelligence", "ai"],
        "financial": ["bank", "banking", "finance", "nbfc", "interest rate", "credit", "loan", "insurance"],
        "energy": ["oil", "gas", "energy", "refinery", "crude", "lng", "solar", "renewable"],
        "automotive": ["auto", "automobile", "vehicle", "ev", "electric vehicle", "car", "motorcycle"],
        "pharma": ["pharma", "drug", "medicine", "healthcare", "fda", "clinical", "biotech"],
        "consumer": ["consumer", "retail", "fmcg", "inflation", "discretionary"],
        "metals": ["metal", "steel", "aluminium", "copper", "iron ore", "mining"],
        "crypto": ["crypto", "bitcoin", "ethereum", "blockchain", "digital asset"],
    }
    found = set()
    for sector, words in groups.items():
        if any(word in text for word in words):
            found.update(words)
    return list(found)


def relevance_and_importance(item, direct_keys, indirect_keys):
    text = f"{item['title']} {item['summary']}".lower()
    direct_hits = sum(1 for k in direct_keys if k in text)
    indirect_hits = sum(1 for k in indirect_keys if k in text)
    catalyst = ["earnings", "revenue", "profit", "guidance", "regulator", "lawsuit", "acquisition", "merger", "tariff", "rate", "inflation", "sanction", "upgrade", "downgrade", "contract", "default", "warning", "forecast", "results"]
    catalyst_hits = sum(1 for word in catalyst if word in text)
    age_hours = max(0.0, (datetime.now(timezone.utc) - item["published_dt"]).total_seconds() / 3600)
    freshness = max(0.0, 10.0 - min(age_hours / 3.0, 10.0))
    source_weight = {"Reuters": 2.0, "CNBC": 1.7, "Economic Times": 1.5, "Moneycontrol": 1.4, "Yahoo Finance": 1.4, "MarketWatch": 1.3}.get(item["source"], 1.0)
    direct = direct_hits > 0
    related = direct or indirect_hits > 0
    importance = min(100.0, 42 * min(direct_hits, 2) + 14 * min(indirect_hits, 3) + 7 * min(catalyst_hits, 3) + freshness + source_weight * 4)
    if not related:
        return None
    item = dict(item)
    item.update({"direct": direct, "direct_hits": direct_hits, "indirect_hits": indirect_hits, "importance": round(importance, 1), "freshness": freshness})
    return item


def deterministic_impact(item, ticker, info):
    text = f"{item['title']} {item['summary']}".lower()
    positive = ["beat", "growth", "profit", "upgrade", "strong", "rises", "surge", "record", "contract", "expansion", "demand", "guidance raised"]
    negative = ["miss", "loss", "downgrade", "weak", "falls", "drop", "lawsuit", "warning", "cuts", "default", "pressure", "guidance cut"]
    p, n = sum(w in text for w in positive), sum(w in text for w in negative)
    score = max(0.0, min(10.0, 5 + p * 1.4 - n * 1.6))
    if score >= 6.5:
        tone, label = "green", "Bullish"
    elif score <= 3.5:
        tone, label = "red", "Bearish"
    else:
        tone, label = "yellow", "Neutral"
    if item["direct"]:
        reason = f"Direct catalyst: this article mentions {ticker} or its company name, so it can affect earnings, valuation, demand, or risk sentiment directly."
    else:
        sector = info.get("sector") or info.get("industry") or "the company’s industry"
        reason = f"Indirect catalyst: this story concerns {sector} or a related macro factor. It can affect {ticker} through demand, input costs, rates, regulation, currency, or investor risk appetite."
    return {"score": round(score, 1), "label": label, "tone": tone, "reason": reason}


def ai_indirect_explanations(items, ticker, info):
    """Ask the configured model to improve explanations for indirect stories in one call."""
    if not groq_client or not items:
        return {}
    payload = "\n".join(f"{i}: {x['title']} — {x['summary']}" for i, x in enumerate(items))
    prompt = f"For stock {ticker} ({info.get('longName','')}), explain why each indirectly related news item may affect the stock. Return one concise line per item in the exact format INDEX|EXPLANATION. Do not invent facts. Items:\n{payload}"
    try:
        response = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=900)
        explanations = {}
        for line in response.choices[0].message.content.splitlines():
            match = re.match(r"\s*(\d+)\s*[|:-]\s*(.+)", line)
            if match:
                explanations[int(match.group(1))] = match.group(2).strip()
        return explanations
    except Exception as exc:
        print(f"News explanation error: {exc}")
        return {}


def render_news(item):
    impact = item["impact"]
    st.markdown(f"""
    <div class="news-card {'direct' if item['direct'] else 'indirect'}">
      <div class="news-head"><span class="badge {impact['tone']}">{'DIRECT' if item['direct'] else 'INDIRECT'} · {impact['label']}</span><b>Importance {item['importance']:.0f}/100 · Impact {impact['score']:.1f}/10</b></div>
      <div class="news-title">{html_module.escape(item['title'])}</div>
      <div class="news-summary">{html_module.escape(item['summary'])}</div>
      <div class="news-reason">🧠 <b>Why it matters:</b> {html_module.escape(impact['reason'])}</div>
      <div class="news-meta">📌 {html_module.escape(item['source'])} · 🕒 {html_module.escape(item['published'])} · <a href="{html_module.escape(item['link'])}" target="_blank">Read source →</a></div>
    </div>
    """, unsafe_allow_html=True)


# ================== ANALYSIS ==================
df, info = stock_data(symbol)
if df is None or df.empty:
    st.error(f"No market data found for {symbol}.")
    st.stop()

price = float(df["Close"].iloc[-1])
news = harvest_news()
keys = stock_keywords(symbol, info)
indirect_keys = related_sector_keywords(symbol, info)
ranked_news = []
for article in news:
    ranked = relevance_and_importance(article, keys, indirect_keys)
    if ranked:
        ranked["impact"] = deterministic_impact(ranked, symbol, info)
        ranked_news.append(ranked)
ranked_news.sort(key=lambda x: (x["importance"], x["published_dt"]), reverse=True)

indirect = [x for x in ranked_news if not x["direct"]][:10]
for idx, explanation in ai_indirect_explanations(indirect, symbol, info).items():
    if 0 <= idx < len(indirect):
        indirect[idx]["impact"]["reason"] = explanation

# ================== TABS ==================
tab_live, tab_news, tab_chat = st.tabs(["📊 Live Analysis", "📰 Ranked Stock News", "💬 AI Chat"])
with tab_live:
    st.header(f"📊 {symbol}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{price:.2f}")
    c2.metric("Company", info.get("longName", symbol))
    c3.metric("Related stories", len(ranked_news))
    chart = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol))
    chart.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(chart, use_container_width=True)
    st.info("The chart is kept simple and responsive. News and smart-money interpretation are shown in the dedicated cards below.")

    st.subheader("🧠 Smart Money Zones")
    recent = df.tail(60).copy()
    if len(recent) >= 15:
        high, low = float(recent["High"].max()), float(recent["Low"].min())
        gp_low, gp_high = low + (high-low)*.618, low + (high-low)*.65
        tone = "Bullish" if recent["Close"].iloc[-1] >= recent["Close"].iloc[0] else "Bearish"
        st.markdown(f"<div class='zone'><b>🌀 Golden Pocket · {tone}</b><div class='zone-grid'><div class='stat'><div class='small'>Current</div><div class='value'>{price:.2f}</div></div><div class='stat'><div class='small'>Zone low</div><div class='value'>{gp_low:.2f}</div></div><div class='stat'><div class='small'>Zone high</div><div class='value'>{gp_high:.2f}</div></div></div><p>Retracement zone calculated from the recent swing range. {'Price is inside the zone.' if gp_low <= price <= gp_high else 'Price is outside the zone.'}</p></div>", unsafe_allow_html=True)
    else:
        st.caption("Not enough data for smart-money zones.")

with tab_news:
    st.header(f"📰 Latest Relevant News — {symbol}")
    st.caption("Fresh pool from all configured feeds. Articles are deduplicated, relevance-ranked, then sorted by importance and freshness. Direct stories appear before indirect sector or macro catalysts.")
    show_indirect = st.checkbox("Include indirect sector/macro news", True)
    limit = st.slider("Number of ranked stories", 5, 30, 12)
    shown = [x for x in ranked_news if show_indirect or x["direct"]]
    shown = [x for x in shown if ((x["impact"]["label"] == "Bullish" and "🟢 Bullish" in sentiment_filter) or (x["impact"]["label"] == "Bearish" and "🔴 Bearish" in sentiment_filter) or (x["impact"]["label"] == "Neutral" and "🟡 Neutral" in sentiment_filter))]
    if shown:
        for item in shown[:limit]:
            render_news(item)
    else:
        st.warning("No relevant stories matched the current filters. Try including indirect news or enabling all sentiments.")

with tab_chat:
    st.header("💬 AI Chat")
    prompt = st.chat_input("Ask about the stock or the ranked news...")
    if prompt:
        if not groq_client:
            st.warning("Add GROQ_KEY to .streamlit/secrets.toml to use AI chat.")
        else:
            context = "\n".join(f"{x['title']} | {x['impact']['reason']}" for x in ranked_news[:10])
            try:
                answer = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role":"system", "content": f"You are a careful financial news assistant for {symbol}. Explain direct and indirect catalysts. Educational only. Context:\n{context}"}, {"role":"user", "content": prompt}], temperature=.2, max_tokens=900).choices[0].message.content
                st.markdown(answer)
            except Exception as exc:
                st.error(f"AI error: {exc}")

st.caption("News relevance and sentiment are heuristic/AI-assisted and are not investment advice.")
