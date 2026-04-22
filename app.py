"""
S&P 500 Live Quant Analyzer - PRO EDITION (Diagnostic)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Quant Signals Pro",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 100%; }
    .stButton button { height: 3rem; font-size: 1.1rem; font-weight: 600; border-radius: 12px; }
    .signal-card {
        background: #1E2128; border-radius: 12px; padding: 1rem;
        margin: 0.5rem 0; border-left: 4px solid #00D26A;
    }
    .sell-card { border-left-color: #FF4B4B !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

# ---------- UNIVERSE ----------
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","WMT"]

@st.cache_data(ttl=86400)
def get_sector_map():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
        return dict(zip(table["Symbol"], table["GICS Sector"]))
    except Exception:
        return {}

# ---------- DATA ----------
@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True, threads=False)
        if df.empty or len(df) < 60:
            return None, f"insufficient_data({len(df)} rows)"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, "ok"
    except Exception as e:
        return None, f"fetch_error({str(e)[:50]})"

@st.cache_data(ttl=300)
def fetch_benchmark():
    df, _ = fetch_data("SPY")
    return df

# ---------- FACTORS ----------
def compute_all_factors(df, spy_df=None):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["EMA9"]  = ta.trend.ema_indicator(c, 9)
    df["EMA21"] = ta.trend.ema_indicator(c, 21)
    df["SMA50"] = ta.trend.sma_indicator(c, 50)
    sma200_win = min(200, len(c)-1)
    df["SMA200"] = ta.trend.sma_indicator(c, sma200_win)
    df["ADX"]   = ta.trend.adx(h, l, c, 14)

    df["RSI"]   = ta.momentum.rsi(c, 14)
    macd = ta.trend.MACD(c)
    df["MACD"]  = macd.macd()
    df["MACDh"] = macd.macd_diff()
    df["ROC10"] = ta.momentum.roc(c, 10)

    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()
    df["BB_pct"] = bb.bollinger_pband()
    df["Zscore"] = (c - c.rolling(20).mean()) / c.rolling(20).std()

    df["ATR"] = ta.volatility.average_true_range(h, l, c, 14)
    df["ATR_pct"] = df["ATR"] / c * 100
    df["Ret1"] = c.pct_change()

    df["VOL_MA20"] = v.rolling(20).mean()
    df["VOL_ratio"] = v / df["VOL_MA20"]
    df["OBV"] = ta.volume.on_balance_volume(c, v)
    df["OBV_slope"] = df["OBV"].diff(10)
    df["MFI"] = ta.volume.money_flow_index(h, l, c, v, 14)

    df["High20"] = h.rolling(20).max()
    df["Low20"] = l.rolling(20).min()
    high52_win = min(252, len(h)-1)
    df["High52w"] = h.rolling(high52_win).max()
    df["Low52w"] = l.rolling(high52_win).min()
    df["PctFromHigh52w"] = (c / df["High52w"] - 1) * 100
    df["PctFromLow52w"] = (c / df["Low52w"] - 1) * 100

    df["Ret5"] = c.pct_change(5)
    df["Ret20"] = c.pct_change(20)

    if spy_df is not None and len(spy_df) > 0:
        spy_close = spy_df["Close"].reindex(df.index).ffill()
        try:
            df["RS"] = (c / spy_close) / (c.iloc[0] / spy_close.iloc[0])
            df["RS_slope"] = df["RS"].diff(10)
        except Exception:
            df["RS_slope"] = 0
    else:
        df["RS_slope"] = 0

    df["Sharpe20"] = (df["Ret1"].rolling(20).mean() / df["Ret1"].rolling(20).std()) * np.sqrt(252)

    return df.dropna()

def score_stock(df, spy_df=None):
    if df is None or len(df) < 50:
        return None, "insufficient_rows"

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
    except Exception as e:
        return None, f"index_error({e})"

    factors = {}
    reasons = []

    # TREND
    ts = 0
    if last["EMA9"] > last["EMA21"] > last["SMA50"]:
        ts += 3; reasons.append("Full bull alignment")
    elif last["EMA9"] < last["EMA21"] < last["SMA50"]:
        ts -= 3; reasons.append("Full bear alignment")
    elif last["EMA9"] > last["EMA21"]:
        ts += 1
    else:
        ts -= 1

    if not pd.isna(last["SMA200"]):
        if last["Close"] > last["SMA200"]: ts += 2
        else: ts -= 2

    if last["ADX"] > 25:
        ts += 2 if ts > 0 else -2
        reasons.append(f"Strong trend ADX{last['ADX']:.0f}")
    elif last["ADX"] < 20:
        ts *= 0.5
    factors["trend"] = ts

    # MOMENTUM
    ms = 0
    if last["MACDh"] > 0 and prev["MACDh"] <= 0:
        ms += 4; reasons.append("MACD bull cross")
    elif last["MACDh"] < 0 and prev["MACDh"] >= 0:
        ms -= 4; reasons.append("MACD bear cross")
    elif last["MACDh"] > 0: ms += 1
    else: ms -= 1

    if last["ROC10"] > 5: ms += 2
    elif last["ROC10"] > 2: ms += 1
    elif last["ROC10"] < -5: ms -= 2
    elif last["ROC10"] < -2: ms -= 1

    if last["Ret5"] > 0.05: ms += 1
    elif last["Ret5"] < -0.05: ms -= 1
    factors["momentum"] = ms

    # MEAN REVERSION
    mrs = 0
    rsi = last["RSI"]
    if rsi <= 25: mrs += 3; reasons.append(f"Oversold RSI{rsi:.0f}")
    elif rsi <= 35: mrs += 1
    elif rsi >= 75: mrs -= 3; reasons.append(f"Overbought RSI{rsi:.0f}")
    elif rsi >= 65: mrs -= 1

    z = last["Zscore"]
    if z < -2: mrs += 2; reasons.append(f"Z{z:.1f}")
    elif z < -1.5: mrs += 1
    elif z > 2: mrs -= 2; reasons.append(f"Z{z:.1f}")
    elif z > 1.5: mrs -= 1

    if last["BB_pct"] < 0.05: mrs += 2
    elif last["BB_pct"] > 0.95: mrs -= 2
    factors["mean_rev"] = mrs

    # VOLUME
    vs = 0
    if last["VOL_ratio"] > 2.0:
        if last["Ret1"] > 0: vs += 3; reasons.append("Bull volume 2x+")
        else: vs -= 3; reasons.append("Bear volume 2x+")
    elif last["VOL_ratio"] > 1.5:
        vs += 2 if last["Ret1"] > 0 else -2

    if last["OBV_slope"] > 0 and last["Ret20"] > 0: vs += 1
    elif last["OBV_slope"] < 0 and last["Ret20"] < 0: vs -= 1

    mfi = last["MFI"]
    if mfi < 20: vs += 2
    elif mfi > 80: vs -= 2
    factors["volume"] = vs

    # BREAKOUT
    bos = 0
    if last["Close"] >= last["High20"] * 0.99:
        bos += 2; reasons.append("At 20d high")
    elif last["Close"] <= last["Low20"] * 1.01:
        bos -= 2; reasons.append("At 20d low")

    if last["PctFromHigh52w"] > -5:
        bos += 2; reasons.append("Near 52wH")
    elif last["PctFromLow52w"] < 10:
        bos -= 1
    factors["breakout"] = bos

    # RELATIVE STRENGTH
    rss = 0
    if not pd.isna(last.get("RS_slope", np.nan)):
        if last["RS_slope"] > 0.02: rss += 2; reasons.append("Beats SPY")
        elif last["RS_slope"] > 0: rss += 1
        elif last["RS_slope"] < -0.02: rss -= 2
        else: rss -= 1
    factors["rel_strength"] = rss

    # RISK-ADJUSTED
    ras = 0
    sh = last.get("Sharpe20", 0)
    if not pd.isna(sh):
        if sh > 1.5: ras += 2
        elif sh > 0.5: ras += 1
        elif sh < -1: ras -= 2
        elif sh < -0.3: ras -= 1
    factors["risk_adj"] = ras

    weights = {"trend":0.25,"momentum":0.20,"mean_rev":0.15,"volume":0.15,
               "breakout":0.10,"rel_strength":0.10,"risk_adj":0.05}
    for k in factors:
        factors[k] = max(-10, min(10, factors[k]))
    composite = round(sum(factors[k]*weights[k] for k in factors), 2)

    avg_dollar_vol = float(last["Close"] * last["VOL_MA20"])

    return {
        "score": composite,
        "factors": factors,
        "reasons": reasons,
        "price": float(last["Close"]),
        "rsi": float(last["RSI"]),
        "adx": float(last["ADX"]),
        "atr": float(last["ATR"]),
        "zscore": float(last["Zscore"]),
        "vol_ratio": float(last["VOL_ratio"]),
        "mfi": float(last["MFI"]),
        "ret5": float(last["Ret5"]),
        "ret20": float(last["Ret20"]),
        "pct_from_52w_high": float(last["PctFromHigh52w"]),
        "sharpe20": float(sh) if not pd.isna(sh) else 0,
        "dollar_vol_m": avg_dollar_vol / 1_000_000,
    }, "ok"

def recommendation(score, adx):
    if adx > 30: hm = 1.3
    elif adx < 20: hm = 0.6
    else: hm = 1.0

    if score >= 5:  return "STRONG BUY", f"{int(3*hm)}-{int(10*hm)}d", "🟢"
    if score >= 3:  return "BUY", f"{int(2*hm)}-{int(7*hm)}d", "🟢"
    if score >= 1.5:return "WEAK BUY", f"{int(1*hm)}-{int(3*hm)}d", "🟡"
    if score > -1.5:return "HOLD", "Wait", "⚪"
    if score > -3:  return "WEAK SELL", f"{int(1*hm)}-{int(3*hm)}d", "🟡"
    if score > -5:  return "SELL", f"{int(2*hm)}-{int(7*hm)}d", "🔴"
    return "STRONG SELL", f"{int(3*hm)}-{int(10*hm)}d", "🔴"

def compute_targets(price, atr, score):
    tmult = 2.5 + min(abs(score), 5) * 0.3
    if score > 0:
        stop = price - 1.5*atr; target = price + tmult*atr
    else:
        stop = price + 1.5*atr; target = price - tmult*atr
    rr = abs(target-price)/abs(price-stop) if price != stop else 0
    return stop, target, rr

def analyze_ticker(ticker, spy_df, sector_map):
    df, fetch_status = fetch_data(ticker)
    if df is None:
        return None, ticker, fetch_status

    try:
        df = compute_all_factors(df, spy_df)
        if df.empty:
            return None, ticker, "empty_after_factors"
        s, score_status = score_stock(df, spy_df)
        if s is None:
            return None, ticker, score_status

        action, hold, emoji = recommendation(s["score"], s["adx"])
        stop, target, rr = compute_targets(s["price"], s["atr"], s["score"])

        return {
            "Ticker": ticker,
            "Sector": sector_map.get(ticker, "Unknown"),
            "Signal": action, "Emoji": emoji,
            "Score": s["score"],
            "Price": round(s["price"], 2),
            "Stop": round(stop, 2),
            "Target": round(target, 2),
            "R:R": round(rr, 2),
            "Hold": hold,
            "RSI": round(s["rsi"], 1),
            "ADX": round(s["adx"], 1),
            "Z-Score": round(s["zscore"], 2),
            "MFI": round(s["mfi"], 1),
            "Vol Ratio": round(s["vol_ratio"], 2),
            "5D%": round(s["ret5"]*100, 2),
            "20D%": round(s["ret20"]*100, 2),
            "From 52w High%": round(s["pct_from_52w_high"], 1),
            "Sharpe": round(s["sharpe20"], 2),
            "$Vol (M)": round(s["dollar_vol_m"], 1),
            "Trend F": round(s["factors"]["trend"], 1),
            "Mom F": round(s["factors"]["momentum"], 1),
            "MR F": round(s["factors"]["mean_rev"], 1),
            "Vol F": round(s["factors"]["volume"], 1),
            "BO F": round(s["factors"]["breakout"], 1),
            "RS F": round(s["factors"]["rel_strength"], 1),
            "Reasons": " • ".join(s["reasons"][:4]),
            "_df": df
        }, ticker, "ok"
    except Exception as e:
        return None, ticker, f"analysis_error({str(e)[:60]})"

def run_full_scan(tickers, spy_df, sector_map, max_workers=20):
    results = []
    errors = []
    prog = st.progress(0, text="Initializing scan...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_ticker, t, spy_df, sector_map): t for t in tickers}
        for i, f in enumerate(as_completed(futures)):
            r, tkr, status = f.result()
            if r:
                results.append(r)
            else:
                errors.append({"Ticker": tkr, "Reason": status})
            prog.progress((i+1)/len(tickers),
                         text=f"Scanning {i+1}/{len(tickers)} - {len(results)} valid, {len(errors)} errors")
    prog.empty()
    return pd.DataFrame(results), pd.DataFrame(errors)

def diversify_picks(df, max_per_sector=3):
    diversified = []
    sector_count = {}
    for _, row in df.iterrows():
        sec = row["Sector"]
        if sector_count.get(sec, 0) < max_per_sector:
            diversified.append(row)
            sector_count[sec] = sector_count.get(sec, 0) + 1
    return pd.DataFrame(diversified)

# ==========================================================
# UI
# ==========================================================
st.title("📈 Quant Signals Pro")
st.caption("Full S&P 500 multi-factor scanner")

with st.expander("⚙️ Settings", expanded=True):
    scan_size = st.radio(
        "Scan Size",
        ["Sample 30 (quick test, 15s)",
         "Sample 100 (45s)",
         "Full S&P 500 (~500, 3-5 min)",
         "Custom"],
        index=0
    )
    
    all_tickers_list = get_sp500_tickers()
    
    if scan_size == "Custom":
        custom = st.text_area("Tickers (comma-separated)", "AAPL,MSFT,NVDA")
        tickers_to_scan = [t.strip().upper() for t in custom.split(",")]
    elif scan_size.startswith("Full"):
        tickers_to_scan = all_tickers_list
    elif "100" in scan_size:
        rng = np.random.default_rng(42)
        picked_idx = rng.choice(len(all_tickers_list), size=min(100, len(all_tickers_list)), replace=False)
        tickers_to_scan = [all_tickers_list[i] for i in picked_idx]
    else:
        rng = np.random.default_rng(42)
        picked_idx = rng.choice(len(all_tickers_list), size=min(30, len(all_tickers_list)), replace=False)
        tickers_to_scan = [all_tickers_list[i] for i in picked_idx]

    st.caption(f"Will scan {len(tickers_to_scan)} tickers")

    st.markdown("---")
    min_score = st.slider("Min |score| to display in Top Picks", 0.0, 8.0, 0.0, 0.5)
    max_per_sector = st.slider("Max picks per sector", 1, 10, 3)
    min_dollar_vol = st.slider("Min daily $ volume (millions) - 0 = no filter", 0, 500, 0, 5)
    show_factors = st.checkbox("Show factor scores in tables", False)

if st.button("🚀 Run Scan", type="primary", use_container_width=True):
    with st.spinner("Loading SPY benchmark..."):
        spy_df = fetch_benchmark()
        sector_map = get_sector_map()
    df_res, df_err = run_full_scan(tickers_to_scan, spy_df, sector_map)
    st.session_state.scan_results = df_res
    st.session_state.scan_errors = df_err
    st.session_state.scan_time = datetime.now().strftime("%Y-%m-%d %H:%M")

if "scan_results" in st.session_state:
    df_res = st.session_state.scan_results
    df_err = st.session_state.scan_errors

    st.caption(f"📅 Scan: {st.session_state.scan_time} | "
               f"Valid: {len(df_res)} | Errors: {len(df_err)}")

    if df_res.empty:
        st.error("❌ No stocks successfully analyzed. See Diagnostics tab below.")
        if not df_err.empty:
            st.subheader("🔧 What went wrong:")
            st.dataframe(df_err, use_container_width=True, height=400)
    else:
        if min_dollar_vol > 0:
            before = len(df_res)
            df_res = df_res[df_res["$Vol (M)"] >= min_dollar_vol].copy()
            st.info(f"Liquidity filter removed {before - len(df_res)} stocks below ${min_dollar_vol}M/day")

        df_res = df_res.sort_values("Score", key=abs, ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Buys", (df_res["Score"] >= 1.5).sum())
        c2.metric("🔴 Sells", (df_res["Score"] <= -1.5).sum())
        c3.metric("⭐ Strong", (df_res["Score"].abs() >= 3).sum())
        c4.metric("📊 Total", len(df_res))

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🎯 Top Picks", "📊 Factors", "📋 All", "📈 Chart", "🔧 Diagnostics"])

        with tab1:
            filt = df_res[df_res["Score"].abs() >= min_score]
            longs = filt[filt["Score"] > 0].sort_values("Score", ascending=False)
            shorts = filt[filt["Score"] < 0].sort_values("Score")

            longs_div = diversify_picks(longs, max_per_sector)
            shorts_div = diversify_picks(shorts, max_per_sector)

            st.subheader(f"🟢 Long Ideas ({len(longs_div)})")
            if len(longs_div) == 0:
                st.info("No long signals above threshold.")
            for _, r in longs_div.head(15).iterrows():
                st.markdown(f"""
                <div class='signal-card'>
                <h3 style='margin:0'>{r['Emoji']} {r['Ticker']} — {r['Signal']} <span style='font-size:0.7em;color:#888'>[{r['Sector']}]</span></h3>
                <p style='margin:4px 0;color:#aaa'>Score: <b>{r['Score']:+.1f}</b> | Hold: <b>{r['Hold']}</b> | R:R: <b>{r['R:R']}</b></p>
                <p style='margin:4px 0'>💰 ${r['Price']} → 🎯 ${r['Target']} | 🛑 ${r['Stop']}</p>
                <p style='margin:4px 0;font-size:0.8rem;color:#aaa'>RSI {r['RSI']} • ADX {r['ADX']} • Z {r['Z-Score']:+.1f} • 5D {r['5D%']:+.1f}%</p>
                <p style='margin:4px 0;font-size:0.85rem;color:#ccc'>{r['Reasons']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader(f"🔴 Short Ideas ({len(shorts_div)})")
            if len(shorts_div) == 0:
                st.info("No short signals above threshold.")
            for _, r in shorts_div.head(15).iterrows():
                st.markdown(f"""
                <div class='signal-card sell-card'>
                <h3 style='margin:0'>{r['Emoji']} {r['Ticker']} — {r['Signal']} <span style='font-size:0.7em;color:#888'>[{r['Sector']}]</span></h3>
                <p style='margin:4px 0;color:#aaa'>Score: <b>{r['Score']:+.1f}</b> | Hold: <b>{r['Hold']}</b> | R:R: <b>{r['R:R']}</b></p>
                <p style='margin:4px 0'>💰 ${r['Price']} → 🎯 ${r['Target']} | 🛑 ${r['Stop']}</p>
                <p style='margin:4px 0;font-size:0.8rem;color:#aaa'>RSI {r['RSI']} • ADX {r['ADX']} • Z {r['Z-Score']:+.1f} • 5D {r['5D%']:+.1f}%</p>
                <p style='margin:4px 0;font-size:0.85rem;color:#ccc'>{r['Reasons']}</p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.subheader("Sector Overview")
            sec_summary = df_res.groupby("Sector").agg(
                Count=("Ticker", "count"),
                Avg_Score=("Score", "mean"),
                Bullish=("Score", lambda x: (x > 1.5).sum()),
                Bearish=("Score", lambda x: (x < -1.5).sum())
            ).round(2).sort_values("Avg_Score", ascending=False)
            st.dataframe(sec_summary, use_container_width=True)

            st.subheader("Factor Breakdown (Top 20)")
            fcols = ["Ticker","Sector","Score","Trend F","Mom F","MR F","Vol F","BO F","RS F"]
            st.dataframe(df_res.head(20)[fcols], use_container_width=True, hide_index=True)

        with tab3:
            cols = ["Ticker","Sector","Signal","Score","Price","Stop","Target","R:R",
                    "Hold","RSI","ADX","Z-Score","5D%","20D%","$Vol (M)"]
            if show_factors:
                cols += ["Trend F","Mom F","MR F","Vol F","BO F","RS F"]
            st.dataframe(df_res[cols], use_container_width=True, hide_index=True, height=500)
            st.download_button("📥 Download CSV",
                df_res.drop(columns=["_df"]).to_csv(index=False),
                f"signals_{st.session_state.scan_time}.csv",
                use_container_width=True)

        with tab4:
            pick = st.selectbox("Select ticker", df_res["Ticker"].tolist())
            row = df_res[df_res["Ticker"] == pick].iloc[0]
            d = row["_df"].tail(90)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                         low=d["Low"], close=d["Close"], name="Price"))
            fig.add_trace(go.Scatter(x=d.index, y=d["EMA9"], name="EMA9",
                                     line=dict(color="orange", width=1.5)))
            fig.add_trace(go.Scatter(x=d.index, y=d["EMA21"], name="EMA21",
                                     line=dict(color="cyan", width=1.5)))
            fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name="SMA50",
                                     line=dict(color="purple", width=1, dash="dot")))
            fig.update_layout(
                title=f"{pick} [{row['Sector']}] — {row['Emoji']} {row['Signal']}",
                xaxis_rangeslider_visible=False, height=500,
                margin=dict(l=10,r=10,t=40,b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{row['Score']:+.1f}")
            c2.metric("Entry", f"${row['Price']}")
            c3.metric("Stop", f"${row['Stop']}")
            c4.metric("Target", f"${row['Target']}")
            st.info(f"**Reasons:** {row['Reasons']}")

        with tab5:
            st.subheader("🔧 Scan Diagnostics")
            total = len(df_res) + len(df_err)
            st.metric("Success Rate", f"{len(df_res)}/{total} "
                     f"({100*len(df_res)/max(1,total):.0f}%)")
            if not df_err.empty:
                st.subheader("Failed Tickers")
                reason_counts = df_err["Reason"].value_counts()
                st.bar_chart(reason_counts)
                st.dataframe(df_err, use_container_width=True, height=300)

else:
    st.info("👆 Tap **Run Scan** to start")

st.caption("⚠️ Educational only. Not financial advice.")
