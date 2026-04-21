"""
S&P 500 Live Quant Analyzer - Mobile Optimized
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ---------- MOBILE-FIRST CONFIG ----------
st.set_page_config(
    page_title="Quant Signals",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    .stButton button { 
        height: 3rem; 
        font-size: 1.1rem; 
        font-weight: 600;
        border-radius: 12px;
    }
    .signal-card {
        background: #1E2128;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00D26A;
    }
    .sell-card { border-left-color: #FF4B4B !important; }
    .neutral-card { border-left-color: #FAA61A !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .dataframe { font-size: 0.85rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return table["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","WMT",
                "JNJ","PG","MA","HD","CVX","ABBV","PFE","KO","PEP","BAC"]

@st.cache_data(ttl=300)
def fetch_data(ticker, period="3mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

# ---------- SIGNALS ----------
def compute_signals(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["EMA9"]  = ta.trend.ema_indicator(c, 9)
    df["EMA21"] = ta.trend.ema_indicator(c, 21)
    df["SMA50"] = ta.trend.sma_indicator(c, 50)
    df["RSI"]   = ta.momentum.rsi(c, 14)
    macd = ta.trend.MACD(c)
    df["MACDh"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_H"], df["BB_L"] = bb.bollinger_hband(), bb.bollinger_lband()
    df["BB_pct"] = bb.bollinger_pband()
    df["ATR"] = ta.volatility.average_true_range(h, l, c, 14)
    df["VOL_MA"] = v.rolling(20).mean()
    df["Ret1"] = c.pct_change()
    df["Ret5"] = c.pct_change(5)
    df["Vol20"] = df["Ret1"].rolling(20).std() * np.sqrt(252)
    return df.dropna()

def score_stock(df):
    if df is None or df.empty: return None
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    score, reasons = 0, []

    if last["EMA9"] > last["EMA21"]: score += 2; reasons.append("Bullish trend")
    else: score -= 2; reasons.append("Bearish trend")
    if last["Close"] > last["SMA50"]: score += 1
    else: score -= 1

    if last["MACDh"] > 0 and prev["MACDh"] <= 0:
        score += 3; reasons.append("MACD bull cross")
    elif last["MACDh"] < 0 and prev["MACDh"] >= 0:
        score -= 3; reasons.append("MACD bear cross")
    elif last["MACDh"] > 0: score += 1
    else: score -= 1

    rsi = last["RSI"]
    if rsi >= 70: score -= 2; reasons.append(f"Overbought RSI {rsi:.0f}")
    elif rsi <= 30: score += 2; reasons.append(f"Oversold RSI {rsi:.0f}")
    elif 60 <= rsi < 70: score += 1
    elif 30 < rsi <= 40: score -= 1

    bbp = last["BB_pct"]
    if bbp < 0.1: score += 2; reasons.append("Near lower BB")
    elif bbp > 0.9: score -= 2; reasons.append("Near upper BB")

    if last["Volume"] > 1.5 * last["VOL_MA"]:
        if last["Ret1"] > 0: score += 2; reasons.append("Bull volume")
        else: score -= 2; reasons.append("Bear volume")

    if last["Ret5"] > 0.03: score += 1
    elif last["Ret5"] < -0.03: score -= 1

    return {"score": score, "reasons": reasons, "price": last["Close"],
            "rsi": last["RSI"], "atr": last["ATR"], "macd_h": last["MACDh"],
            "ret5": last["Ret5"]}

def recommendation(score):
    if score >= 7:  return "STRONG BUY", "5-10 days", "🟢"
    if score >= 4:  return "BUY", "3-7 days", "🟢"
    if score >= 2:  return "WEAK BUY", "1-3 days", "🟡"
    if score > -2:  return "HOLD", "Wait", "⚪"
    if score > -4:  return "WEAK SELL", "1-3 days", "🟡"
    if score > -7:  return "SELL", "3-7 days", "🔴"
    return "STRONG SELL", "5-10 days", "🔴"

def compute_targets(price, atr, score):
    if score > 0:
        stop, target = price - 1.5*atr, price + 2.5*atr
    else:
        stop, target = price + 1.5*atr, price - 2.5*atr
    rr = abs(target-price)/abs(price-stop) if price != stop else 0
    return stop, target, rr

def analyze_ticker(ticker):
    df = fetch_data(ticker)
    if df is None: return None
    try:
        df = compute_signals(df)
        s = score_stock(df)
        if s is None: return None
        action, hold, emoji = recommendation(s["score"])
        stop, target, rr = compute_targets(s["price"], s["atr"], s["score"])
        return {
            "Ticker": ticker, "Signal": action, "Emoji": emoji,
            "Score": s["score"], "Price": round(s["price"], 2),
            "Stop": round(stop, 2), "Target": round(target, 2),
            "R:R": round(rr, 2), "Hold": hold,
            "RSI": round(s["rsi"], 1), "Ret5": round(s["ret5"]*100, 2),
            "Reasons": " • ".join(s["reasons"][:3]), "_df": df
        }
    except Exception:
        return None

def run_scan(tickers):
    results = []
    prog = st.progress(0, text="Scanning...")
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(analyze_ticker, t): t for t in tickers}
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            if r: results.append(r)
            prog.progress((i+1)/len(tickers), text=f"Scanning... {i+1}/{len(tickers)}")
    prog.empty()
    return pd.DataFrame(results)

# ---------- UI ----------
st.title("📈 Quant Signals")
st.caption("Live S&P 500 short-term trade ideas")

with st.expander("⚙️ Settings", expanded=False):
    universe = st.radio("Universe",
        ["Top 30 (Fast)", "Top 100", "Full S&P 500", "Custom"],
        index=0)
    if universe == "Custom":
        custom = st.text_area("Tickers", "AAPL,MSFT,NVDA,TSLA")
        tickers = [t.strip().upper() for t in custom.split(",")]
    elif universe == "Top 30 (Fast)":
        tickers = get_sp500_tickers()[:30]
    elif universe == "Top 100":
        tickers = get_sp500_tickers()[:100]
    else:
        tickers = get_sp500_tickers()
    min_score = st.slider("Min signal strength", 0, 10, 4)

if st.button("🚀 Run Live Scan", type="primary", use_container_width=True):
    st.session_state.scan_results = run_scan(tickers)

if "scan_results" in st.session_state:
    df_res = st.session_state.scan_results
    if df_res.empty:
        st.error("No results. Try again.")
    else:
        df_res = df_res.sort_values("Score", key=abs, ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Buys", (df_res["Score"] >= 2).sum())
        c2.metric("🔴 Sells", (df_res["Score"] <= -2).sum())
        c3.metric("📊 Total", len(df_res))

        tab1, tab2, tab3 = st.tabs(["🎯 Top Picks", "📋 All", "📈 Chart"])

        with tab1:
            filt = df_res[df_res["Score"].abs() >= min_score]
            st.subheader(f"🟢 Long Ideas ({(filt['Score']>0).sum()})")
            for _, r in filt[filt["Score"] > 0].head(10).iterrows():
                st.markdown(f"""
                <div class='signal-card'>
                <h3 style='margin:0'>{r['Emoji']} {r['Ticker']} — {r['Signal']}</h3>
                <p style='margin:4px 0;color:#aaa'>Score: <b>{r['Score']:+d}</b> | Hold: <b>{r['Hold']}</b></p>
                <p style='margin:4px 0'>💰 ${r['Price']} → 🎯 ${r['Target']} | 🛑 ${r['Stop']} | R:R {r['R:R']}</p>
                <p style='margin:4px 0;font-size:0.85rem;color:#ccc'>{r['Reasons']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader(f"🔴 Short Ideas ({(filt['Score']<0).sum()})")
            for _, r in filt[filt["Score"] < 0].sort_values("Score").head(10).iterrows():
                st.markdown(f"""
                <div class='signal-card sell-card'>
                <h3 style='margin:0'>{r['Emoji']} {r['Ticker']} — {r['Signal']}</h3>
                <p style='margin:4px 0;color:#aaa'>Score: <b>{r['Score']:+d}</b> | Hold: <b>{r['Hold']}</b></p>
                <p style='margin:4px 0'>💰 ${r['Price']} → 🎯 ${r['Target']} | 🛑 ${r['Stop']} | R:R {r['R:R']}</p>
                <p style='margin:4px 0;font-size:0.85rem;color:#ccc'>{r['Reasons']}</p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            display = df_res[["Ticker","Signal","Score","Price","Hold","RSI","Ret5"]].copy()
            display.columns = ["Ticker","Signal","Score","$","Hold","RSI","5D%"]
            st.dataframe(display, use_container_width=True, hide_index=True, height=500)
            st.download_button("📥 CSV", df_res.drop(columns=["_df"]).to_csv(index=False),
                               "signals.csv", use_container_width=True)

        with tab3:
            pick = st.selectbox("Ticker", df_res["Ticker"].tolist())
            row = df_res[df_res["Ticker"] == pick].iloc[0]
            d = row["_df"].tail(60)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                         low=d["Low"], close=d["Close"], name="Price"))
            fig.add_trace(go.Scatter(x=d.index, y=d["EMA9"], name="EMA9",
                                     line=dict(color="orange", width=1.5)))
            fig.add_trace(go.Scatter(x=d.index, y=d["EMA21"], name="EMA21",
                                     line=dict(color="cyan", width=1.5)))
            fig.update_layout(
                title=f"{pick} — {row['Emoji']} {row['Signal']}",
                xaxis_rangeslider_visible=False,
                height=400, margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False, template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Entry", f"${row['Price']}")
            c2.metric("Stop", f"${row['Stop']}")
            c3.metric("Target", f"${row['Target']}")
            st.info(f"**Why:** {row['Reasons']}")
            st.caption(f"Hold ~{row['Hold']} | R:R {row['R:R']}")
else:
    st.info("👆 Tap **Run Live Scan** to get started")

st.caption("⚠️ Educational only. Not financial advice. Data: yfinance (15-min delayed)")