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

GROQ_MODELS = {"Groq GPT-OSS 20B": "openai/gpt-oss-20b", "Groq GPT-OSS 120B": "openai/gpt-oss-120b", "Groq Qwen3 32B": "qwen/qwen3-32b", "Groq Kimi K2": "moonshotai/kimi-k2-instruct"}
GEMINI_MODELS = {"Gemini Flash": "gemini-3.5-flash", "Gemini Pro": "gemini-3.1-pro-preview"}
NEWS_SOURCES = [("Moneycontrol", "https://www.moneycontrol.com/rss/latestnews.xml"), ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1998028306.cms"), ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories"), ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"), ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline"), ("Alpha Ideas", "https://alphaideas.in/feed"), ("Cointelegraph", "https://cointelegraph.com/feed"), ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss"), ("ForexLive", "https://www.forexlive.com/servicexml/xml.aspx?xml=1")]

st.markdown("""
<style>
html,body,[class*="css"]{font-size:16px}.stButton>button{border-radius:8px;padding:.5rem 1rem}
.news-card{border:1px solid rgba(150,150,150,.28);border-radius:14px;padding:15px 16px;margin:0 0 13px;background:rgba(255,255,255,.025)}
.news-card.direct{border-left:5px solid #26a269}.news-card.indirect{border-left:5px solid #e8a317}.news-head{display:flex;justify-content:space-between;gap:10px;align-items:center}.news-title{font-size:1.05rem;font-weight:700;line-height:1.4;margin:7px 0}.news-summary,.news-reason{line-height:1.55;opacity:.88}.news-meta{font-size:.78rem;opacity:.68;margin-top:9px}.badge{padding:5px 10px;border-radius:99px;font-size:.75rem;font-weight:700}.green{background:#164d35;color:#6ee7a0}.yellow{background:#604811;color:#ffd166}.red{background:#5b2020;color:#ff8b8b}.zone{border:1px solid rgba(140,140,140,.3);border-radius:15px;padding:15px;margin:10px 0}.zone-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.stat{padding:9px;border-radius:9px;background:rgba(0,0,0,.08)}.small{font-size:.75rem;opacity:.7}.value{font-weight:700;margin-top:3px}@media(max-width:640px){h1{font-size:1.4rem!important}.zone-grid{grid-template-columns:1fr}.news-title{font-size:1rem}}
</style>""", unsafe_allow_html=True)
st.title("🎯 AI Trading Terminal - PRO")


def yahoo_search(q):
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": q, "quotesCount": 8, "newsCount": 0}, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        return [{"symbol": x["symbol"], "name": x.get("shortname") or x.get("longname") or x["symbol"], "exchange": x.get("exchange", ""), "type": x.get("quoteType", "")} for x in r.json().get("quotes", []) if x.get("symbol")]
    except Exception:
        return []


st.sidebar.header("🔍 Search Any Stock / Index / Crypto")
query = st.sidebar.text_input("Company name or ticker", "RELIANCE", key="global_search_query").strip().upper()
search_results = yahoo_search(query) if query else []
if search_results:
    options = [f"{x['symbol']} — {x['name']} ({x['exchange']} · {x['type']})" for x in search_results]
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
        d = email.utils.parsedate_to_datetime(value)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=30)


@st.cache_data(ttl=180, show_spinner=False)
def harvest_news():
    unique = {}
    for source, url in NEWS_SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as response:
                root = ET.fromstring(response.read())
            for item in root.findall(".//item")[:20]:
                title, link = clean_text(item.findtext("title", "")), (item.findtext("link", "") or "").strip()
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
        t = yf.Ticker(ticker)
        return t.history(period="3mo", interval="1d"), t.info or {}
    except Exception:
        return pd.DataFrame(), {}


def stock_keys(ticker, info):
    raw = re.sub(r"(\.NS|-USD|=X|=F)$", "", ticker.lower())
    name = (info.get("longName") or info.get("shortName") or "").lower()
    words = re.findall(r"[a-z0-9]+", name)
    return [x for x in {raw, name, *[w for w in words if len(w) > 2]} if x]


def indirect_keys(ticker, info):
    text = f"{ticker} {info.get('longName','')} {info.get('sector','')} {info.get('industry','')}".lower()
    groups = {"technology":["technology","software","cloud","semiconductor","chip","artificial intelligence","ai"],"financial":["bank","banking","finance","interest rate","credit","loan","insurance"],"energy":["oil","gas","energy","refinery","crude","lng","solar","renewable"],"automotive":["auto","automobile","vehicle","ev","electric vehicle","car","motorcycle"],"pharma":["pharma","drug","medicine","healthcare","fda","clinical","biotech"],"consumer":["consumer","retail","fmcg","inflation"],"metals":["metal","steel","aluminium","copper","mining"],"crypto":["crypto","bitcoin","ethereum","blockchain"]}
    out = set()
    for words in groups.values():
        if any(w in text for w in words): out.update(words)
    return list(out)


def rank_news(article, direct_keys, related_keys):
    text = f"{article['title']} {article['summary']}".lower()
    direct_hits, related_hits = sum(k in text for k in direct_keys), sum(k in text for k in related_keys)
    catalysts = ["earnings","revenue","profit","guidance","regulator","lawsuit","acquisition","merger","tariff","rate","inflation","sanction","upgrade","downgrade","contract","default","warning","forecast","results"]
    catalyst_hits = sum(x in text for x in catalysts)
    age_hours = max(0, (datetime.now(timezone.utc)-article["published_dt"]).total_seconds()/3600)
    freshness = max(0, 10-min(age_hours/3, 10))
    source_weight = {"CNBC":1.7,"Economic Times":1.5,"Moneycontrol":1.4,"Yahoo Finance":1.4,"MarketWatch":1.3}.get(article["source"],1)
    if not direct_hits and not related_hits: return None
    result = dict(article)
    result.update({"direct":direct_hits > 0, "importance":round(min(100, 42*min(direct_hits,2)+14*min(related_hits,3)+7*min(catalyst_hits,3)+freshness+source_weight*4),1)})
    return result


def impact(article, ticker, info):
    text = f"{article['title']} {article['summary']}".lower()
    pos = ["beat","growth","profit","upgrade","strong","rises","surge","record","contract","expansion","demand","guidance raised"]
    neg = ["miss","loss","downgrade","weak","falls","drop","lawsuit","warning","cuts","default","pressure","guidance cut"]
    score = max(0, min(10, 5 + sum(x in text for x in pos)*1.4 - sum(x in text for x in neg)*1.6))
    tone, label = ("green","Bullish") if score >= 6.5 else (("red","Bearish") if score <= 3.5 else ("yellow","Neutral"))
    if article["direct"]:
        reason = f"Direct catalyst: the story mentions {ticker} or its company name and may affect earnings, valuation, demand, or risk sentiment directly."
    else:
        sector = info.get("sector") or info.get("industry") or "the company’s industry"
        reason = f"Indirect catalyst: this relates to {sector} or a macro factor. It may affect {ticker} through demand, costs, rates, regulation, currency, or investor risk appetite."
    return {"score":round(score,1),"tone":tone,"label":label,"reason":reason}


def explain_indirect(items, ticker, info):
    if not groq_client or not items: return {}
    payload = "\n".join(f"{i}|{x['title']} — {x['summary']}" for i,x in enumerate(items))
    prompt = f"For {ticker} ({info.get('longName','')}), explain each indirect financial news link in one careful concise sentence. Do not invent facts. Return INDEX|EXPLANATION only.\n{payload}"
    try:
        text = groq_client.chat.completions.create(model="openai/gpt-oss-20b", messages=[{"role":"user","content":prompt}], temperature=.1, max_tokens=800).choices[0].message.content
        return {int(m.group(1)):m.group(2).strip() for line in text.splitlines() if (m:=re.match(r"\s*(\d+)\s*[|:-]\s*(.+)",line))}
    except Exception:
        return {}


def render_news(article):
    x, i = article["impact"], article["importance"]
    st.markdown(f"""
<div class="news-card {'direct' if article['direct'] else 'indirect'}"><div class="news-head"><span class="badge {x['tone']}">{'DIRECT' if article['direct'] else 'INDIRECT'} · {x['label']}</span><b>Importance {i:.0f}/100 · Impact {x['score']:.1f}/10</b></div><div class="news-title">{html_module.escape(article['title'])}</div><div class="news-summary">{html_module.escape(article['summary'])}</div><div class="news-reason">🧠 <b>Why it matters:</b> {html_module.escape(x['reason'])}</div><div class="news-meta">📌 {html_module.escape(article['source'])} · 🕒 {html_module.escape(article['published'])} · <a href="{html_module.escape(article['link'])}" target="_blank">Read source →</a></div></div>
""", unsafe_allow_html=True)


df, info = get_stock_data(symbol)
if df.empty:
    st.error(f"No market data found for {symbol}.")
    st.stop()
price = float(df["Close"].iloc[-1])
all_news = harvest_news()
ranked_news = [rank_news(x, stock_keys(symbol, info), indirect_keys(symbol, info)) for x in all_news]
ranked_news = [x for x in ranked_news if x]
for x in ranked_news: x["impact"] = impact(x, symbol, info)
ranked_news.sort(key=lambda x:(x["importance"],x["published_dt"]), reverse=True)
indirect = [x for x in ranked_news if not x["direct"]][:10]
for n, explanation in explain_indirect(indirect, symbol, info).items():
    if 0 <= n < len(indirect): indirect[n]["impact"]["reason"] = explanation

# ================== LIVE ANALYSIS AND RANKED NEWS (preserved) ==================
tab_live, tab_news, tab_chat, tab_heatmap, tab_portfolio, tab_alerts, tab_history = st.tabs(["📊 Live Analysis","📰 Ranked Stock News","💬 AI Chat","📊 Market Heatmap","📋 Watchlist & Portfolio","🔔 Alerts & Backtest","📅 Historical Data"])
with tab_live:
    st.header(f"📊 Live Analysis: {symbol}")
    c1,c2,c3 = st.columns(3); c1.metric("Current Price",f"{price:.2f}"); c2.metric("Company",info.get("longName",symbol)); c3.metric("Related stories",len(ranked_news))
    chart=go.Figure(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name=symbol)); chart.update_layout(height=460,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=35,b=10)); st.plotly_chart(chart,use_container_width=True)
    st.subheader("🧠 Smart Money Zones")
    recent=df.tail(60)
    if len(recent)>=15:
        hi,lo=float(recent["High"].max()),float(recent["Low"].min()); gp_low,gp_high=lo+(hi-lo)*.618,lo+(hi-lo)*.65; trend="Bullish" if recent["Close"].iloc[-1]>=recent["Close"].iloc[0] else "Bearish"
        st.markdown(f"<div class='zone'><b>🌀 Golden Pocket · {trend}</b><div class='zone-grid'><div class='stat'><div class='small'>Current</div><div class='value'>{price:.2f}</div></div><div class='stat'><div class='small'>Zone low</div><div class='value'>{gp_low:.2f}</div></div><div class='stat'><div class='small'>Zone high</div><div class='value'>{gp_high:.2f}</div></div></div><p>Recent swing retracement zone. {'Price is inside the zone.' if gp_low<=price<=gp_high else 'Price is outside the zone.'}</p></div>",unsafe_allow_html=True)

with tab_news:
    st.header(f"📰 Latest Relevant News — {symbol}"); st.caption("Fresh, deduplicated articles from all feeds, ranked by direct relevance, indirect sector relevance, catalyst importance, source quality, and freshness.")
    include_indirect=st.checkbox("Include indirect sector/macro news",True); limit=st.slider("Number of ranked stories",5,30,12)
    shown=[x for x in ranked_news if include_indirect or x["direct"]]; shown=[x for x in shown if (x["impact"]["label"]=="Bullish" and "🟢 Bullish" in sentiment_filter) or (x["impact"]["label"]=="Bearish" and "🔴 Bearish" in sentiment_filter) or (x["impact"]["label"]=="Neutral" and "🟡 Neutral" in sentiment_filter)]
    if shown:
        for article in shown[:limit]: render_news(article)
    else: st.warning("No relevant stories matched the current filters.")

# ================== RESTORED FEATURES ==================
with tab_chat:
    st.header("💬 AI Chat Assistant"); model_name=st.selectbox("Choose model",list(GROQ_MODELS)+list(GEMINI_MODELS)); prompt=st.chat_input("Ask about this stock, levels, news, or strategy")
    if prompt:
        context="\n".join(f"{x['title']} | {x['impact']['reason']}" for x in ranked_news[:10])
        try:
            if model_name in GROQ_MODELS and groq_client:
                answer=groq_client.chat.completions.create(model=GROQ_MODELS[model_name],messages=[{"role":"system","content":f"You are an educational trading assistant for {symbol}. Context:\n{context}"},{"role":"user","content":prompt}],temperature=.2,max_tokens=900).choices[0].message.content
            elif model_name in GEMINI_MODELS and GEMINI_KEY:
                answer=genai.GenerativeModel(GEMINI_MODELS[model_name]).generate_content(f"You are an educational trading assistant for {symbol}. Context:\n{context}\nQuestion: {prompt}").text
            else: answer="Configure the selected model API key in .streamlit/secrets.toml."
            st.chat_message("assistant").markdown(answer)
        except Exception as exc: st.error(f"AI error: {exc}")

with tab_heatmap:
    st.header("📊 Market Heatmap")
    watch=["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA"] if market if 'market' in locals() and market=="United States" else ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS"]
    rows=[]
    for ticker in watch:
        try:
            h,_=get_stock_data(ticker); change=(float(h["Close"].iloc[-1])/float(h["Close"].iloc[-2])-1)*100; rows.append({"Stock":ticker,"Price":float(h["Close"].iloc[-1]),"Change %":change,"Intensity":min(10,abs(change)/.3)})
        except Exception: pass
    if rows:
        heat=pd.DataFrame(rows).sort_values("Change %",ascending=False); st.dataframe(heat.style.format({"Price":"{:.2f}","Change %":"{:.2f}%","Intensity":"{:.1f}/10"}),use_container_width=True)

with tab_portfolio:
    st.header("📋 Watchlist & Portfolio")
    if "watchlists" not in st.session_state: st.session_state.watchlists={"My Stocks":["RELIANCE.NS","TCS.NS","INFY.NS"],"US":["AAPL","NVDA"],"Crypto":["BTC-USD","ETH-USD"]}
    st.write(st.session_state.watchlists)
    st.subheader("💼 Portfolio Tracker"); holding=st.text_input("Ticker",key="holding_ticker"); qty=st.number_input("Quantity",1,100000,10,key="holding_qty"); buy=st.number_input("Buy price",.01,1000000.,100.,key="holding_buy")
    if st.button("Add holding") and holding: st.session_state.setdefault("portfolio",{})[holding.upper()]={"qty":qty,"buy":buy}
    total=0.
    for ticker,data in st.session_state.get("portfolio",{}).items():
        h,_=get_stock_data(ticker); current=float(h["Close"].iloc[-1]) if not h.empty else 0; value=current*data["qty"]; pnl=value-data["buy"]*data["qty"]; total+=pnl; st.metric(ticker,f"{value:.2f}",f"P&L {pnl:.2f}")
    if st.session_state.get("portfolio"): st.metric("Total P&L",f"{total:.2f}")

with tab_alerts:
    st.header("🔔 Alerts & Backtest")
    alert_ticker=st.text_input("Alert ticker",symbol,key="alert_ticker"); alert_price=st.number_input("Alert price",.01,1000000.,price,key="alert_price"); alert_type=st.selectbox("Trigger",["Above","Below"],key="alert_type")
    if st.button("Create price alert"): st.session_state.setdefault("alerts",[]).append({"ticker":alert_ticker,"price":alert_price,"type":alert_type})
    for a in st.session_state.get("alerts",[]):
        h,_=get_stock_data(a["ticker"]); current=float(h["Close"].iloc[-1]) if not h.empty else 0; triggered=(a["type"]=="Above" and current>=a["price"]) or (a["type"]=="Below" and current<=a["price"]); (st.error if triggered else st.info)(f"{a['ticker']} {a['type']} {a['price']:.2f} | Current {current:.2f}")
    st.subheader("🧪 Backtesting"); bt=st.text_input("Backtest ticker",symbol,key="bt_ticker"); strategy=st.selectbox("Strategy",["Buy at S1","Sell at R1"],key="bt_strategy"); start=st.date_input("Start",datetime.now()-timedelta(days=180),key="bt_start"); end=st.date_input("End",datetime.now(),key="bt_end")
    if st.button("Run backtest"):
        h,_=get_stock_data(bt); h=h[(h.index.date>=start)&(h.index.date<=end)]; trades=0; profit=0
        for i in range(1,len(h)):
            p=(h.High.iloc[i-1]+h.Low.iloc[i-1]+h.Close.iloc[i-1])/3; level=(2*p-h.High.iloc[i-1]) if strategy=="Buy at S1" else (2*p-h.Low.iloc[i-1]); target=p
            if (strategy=="Buy at S1" and h.Low.iloc[i]<=level) or (strategy=="Sell at R1" and h.High.iloc[i]>=level): trades+=1; profit+=target-level
        st.success(f"Trades: {trades} · Approx. total: {profit:.2f}")

with tab_history:
    st.header("📅 Historical Data"); days=st.slider("Days",1,60,10); history=df.tail(days).copy(); history["Change %"]=history["Close"].pct_change()*100; st.dataframe(history[["Open","High","Low","Close","Volume","Change %"]].sort_index(ascending=False),use_container_width=True)

st.caption("News relevance and sentiment are heuristic/AI-assisted and are not investment advice.")
