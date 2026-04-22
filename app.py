"""
S&P 500 Live Quant Analyzer - PROFESSIONAL EDITION
Multi-factor model with unbiased full-universe scanning.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIG ----------
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
    .neutral-card { border-left-color: #FAA61A !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .dataframe { font-size: 0.85rem; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    .factor-pill {
        display: inline-block; padding: 2px 8px; margin: 2px;
        border-radius: 12px; font-size: 0.75rem; background: #2D3139;
    }
</style>
""", unsafe_allow_html=True)

# ---------- TICKER UNIVERSE ----------
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """Fetch the full S&P 500 list — unbiased."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except Exception:
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM"]

@st.cache_data(ttl=86400)
def get_sector_map():
    """Map each ticker to its GICS sector for sector-relative analysis."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
        return dict(zip(table["Symbol"], table["GICS Sector"]))
    except Exception:
        return {}

# ---------- DATA FETCH ----------
@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    """Fetch 6 months for robust indicator calculation."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_benchmark():
    """Fetch SPY as market benchmark."""
    return fetch_data("SPY", period="6mo")

# ---------- FACTOR CALCULATIONS ----------
def compute_all_factors(df, spy_df=None):
    """Compute the full suite of quantitative factors."""
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

    # --- TREND FACTORS ---
    df["EMA9"]  = ta.trend.ema_indicator(c, 9)
    df["EMA21"] = ta.trend.ema_indicator(c, 21)
    df["SMA50"] = ta.trend.sma_indicator(c, 50)
    df["SMA200"] = ta.trend.sma_indicator(c, min(200, len(c)-1))
    df["ADX"]   = ta.trend.adx(h, l, c, 14)  # trend strength 0-100

    # --- MOMENTUM FACTORS ---
    df["RSI"]   = ta.momentum.rsi(c, 14)
    macd = ta.trend.MACD(c)
    df["MACD"]  = macd.macd()
    df["MACDs"] = macd.macd_signal()
    df["MACDh"] = macd.macd_diff()
    df["STOCH"] = ta.momentum.stoch(h, l, c, 14)
    df["ROC10"] = ta.momentum.roc(c, 10)  # rate of change

    # --- MEAN REVERSION ---
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_H"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()
    df["BB_M"] = bb.bollinger_mavg()
    df["BB_pct"] = bb.bollinger_pband()
    df["BB_width"] = (df["BB_H"] - df["BB_L"]) / df["BB_M"]  # volatility squeeze
    # Z-score: how far from 20-day mean in std deviations
    df["Zscore"] = (c - c.rolling(20).mean()) / c.rolling(20).std()

    # --- VOLATILITY ---
    df["ATR"] = ta.volatility.average_true_range(h, l, c, 14)
    df["ATR_pct"] = df["ATR"] / c * 100  # normalized
    df["Ret1"] = c.pct_change()
    df["RealVol20"] = df["Ret1"].rolling(20).std() * np.sqrt(252) * 100

    # --- VOLUME ---
    df["VOL_MA20"] = v.rolling(20).mean()
    df["VOL_ratio"] = v / df["VOL_MA20"]
    df["OBV"] = ta.volume.on_balance_volume(c, v)
    df["OBV_slope"] = df["OBV"].diff(10)  # accumulation/distribution
    df["MFI"] = ta.volume.money_flow_index(h, l, c, v, 14)  # volume-weighted RSI

    # --- BREAKOUT / HIGHS & LOWS ---
    df["High20"] = h.rolling(20).max()
    df["Low20"] = l.rolling(20).min()
    df["High52w"] = h.rolling(min(252, len(h)-1)).max()
    df["Low52w"] = l.rolling(min(252, len(l)-1)).min()
    df["PctFromHigh52w"] = (c / df["High52w"] - 1) * 100
    df["PctFromLow52w"] = (c / df["Low52w"] - 1) * 100

    # --- RETURNS ---
    df["Ret5"] = c.pct_change(5)
    df["Ret10"] = c.pct_change(10)
    df["Ret20"] = c.pct_change(20)

    # --- RELATIVE STRENGTH vs SPY ---
    if spy_df is not None and len(spy_df) > 0:
        spy_aligned = spy_df["Close"].reindex(df.index).ffill()
        df["RS"] = (c / spy_aligned) / (c.iloc[0] / spy_aligned.iloc[0])
        df["RS_slope"] = df["RS"].diff(10)  # is stock gaining vs market?

    # --- RISK-ADJUSTED RETURN (Sharpe-like) ---
    df["Sharpe20"] = (df["Ret1"].rolling(20).mean() / df["Ret1"].rolling(20).std()) * np.sqrt(252)

    return df.dropna()

# ---------- MULTI-FACTOR SCORING ----------
def score_stock(df, spy_df=None):
    """
    Professional multi-factor scoring.
    Returns dict with score (-100 to +100 scale, normalized to -10 to +10).
    """
    if df is None or len(df) < 50:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    prev5 = df.iloc[-6] if len(df) > 5 else last

    factors = {}  # individual factor scores
    reasons = []

    # =========================================================
    # 1. TREND FACTORS (weight: 25%)
    # =========================================================
    trend_score = 0
    # EMA alignment
    if last["EMA9"] > last["EMA21"] > last["SMA50"]:
        trend_score += 3; reasons.append("Full bull alignment")
    elif last["EMA9"] < last["EMA21"] < last["SMA50"]:
        trend_score -= 3; reasons.append("Full bear alignment")
    elif last["EMA9"] > last["EMA21"]:
        trend_score += 1
    else:
        trend_score -= 1

    # Long-term trend (vs SMA200)
    if "SMA200" in df.columns and not pd.isna(last["SMA200"]):
        if last["Close"] > last["SMA200"]:
            trend_score += 2
        else:
            trend_score -= 2

    # ADX trend strength
    if last["ADX"] > 25:
        if trend_score > 0:
            trend_score += 2; reasons.append(f"Strong trend (ADX {last['ADX']:.0f})")
        else:
            trend_score -= 2
    elif last["ADX"] < 20:
        trend_score *= 0.5  # weak trend, reduce conviction
        reasons.append(f"Weak trend (ADX {last['ADX']:.0f})")

    factors["trend"] = trend_score

    # =========================================================
    # 2. MOMENTUM FACTORS (weight: 20%)
    # =========================================================
    mom_score = 0
    # MACD crossover
    if last["MACDh"] > 0 and prev["MACDh"] <= 0:
        mom_score += 4; reasons.append("Fresh MACD bull cross")
    elif last["MACDh"] < 0 and prev["MACDh"] >= 0:
        mom_score -= 4; reasons.append("Fresh MACD bear cross")
    elif last["MACDh"] > 0:
        mom_score += 1
    else:
        mom_score -= 1

    # Rate of change
    if last["ROC10"] > 5:
        mom_score += 2
    elif last["ROC10"] > 2:
        mom_score += 1
    elif last["ROC10"] < -5:
        mom_score -= 2
    elif last["ROC10"] < -2:
        mom_score -= 1

    # 5-day momentum
    if last["Ret5"] > 0.05:
        mom_score += 1
    elif last["Ret5"] < -0.05:
        mom_score -= 1

    factors["momentum"] = mom_score

    # =========================================================
    # 3. MEAN REVERSION (weight: 15%)
    # =========================================================
    mr_score = 0
    # RSI
    rsi = last["RSI"]
    if rsi <= 25:
        mr_score += 3; reasons.append(f"Extreme oversold RSI {rsi:.0f}")
    elif rsi <= 35:
        mr_score += 1
    elif rsi >= 75:
        mr_score -= 3; reasons.append(f"Extreme overbought RSI {rsi:.0f}")
    elif rsi >= 65:
        mr_score -= 1

    # Z-score (statistical overextension)
    z = last["Zscore"]
    if z < -2:
        mr_score += 2; reasons.append(f"2σ below mean (Z={z:.1f})")
    elif z < -1.5:
        mr_score += 1
    elif z > 2:
        mr_score -= 2; reasons.append(f"2σ above mean (Z={z:.1f})")
    elif z > 1.5:
        mr_score -= 1

    # Bollinger position
    bbp = last["BB_pct"]
    if bbp < 0.05:
        mr_score += 2
    elif bbp > 0.95:
        mr_score -= 2

    factors["mean_rev"] = mr_score

    # =========================================================
    # 4. VOLUME / ACCUMULATION (weight: 15%)
    # =========================================================
    vol_score = 0
    # Volume surge
    if last["VOL_ratio"] > 2.0:
        if last["Ret1"] > 0:
            vol_score += 3; reasons.append("Heavy bull volume 2x+")
        else:
            vol_score -= 3; reasons.append("Heavy bear volume 2x+")
    elif last["VOL_ratio"] > 1.5:
        if last["Ret1"] > 0:
            vol_score += 2
        else:
            vol_score -= 2

    # OBV slope (accumulation)
    if last["OBV_slope"] > 0 and last["Ret10"] > 0:
        vol_score += 1; reasons.append("OBV accumulation")
    elif last["OBV_slope"] < 0 and last["Ret10"] < 0:
        vol_score -= 1

    # Money Flow Index
    mfi = last["MFI"]
    if mfi < 20:
        vol_score += 2
    elif mfi > 80:
        vol_score -= 2

    factors["volume"] = vol_score

    # =========================================================
    # 5. BREAKOUT / POSITION (weight: 10%)
    # =========================================================
    bo_score = 0
    # 52-week positioning
    pct_from_high = last["PctFromHigh52w"]
    pct_from_low = last["PctFromLow52w"]

    if last["Close"] >= last["High20"] * 0.99:
        bo_score += 2; reasons.append("Near 20-day high")
    elif last["Close"] <= last["Low20"] * 1.01:
        bo_score -= 2; reasons.append("Near 20-day low")

    if pct_from_high > -5:  # within 5% of 52w high
        bo_score += 2; reasons.append(f"Within 5% of 52w high")
    elif pct_from_low < 10:  # within 10% of 52w low
        bo_score -= 1; reasons.append(f"Near 52w low")

    factors["breakout"] = bo_score

    # =========================================================
    # 6. RELATIVE STRENGTH vs SPY (weight: 10%)
    # =========================================================
    rs_score = 0
    if "RS_slope" in df.columns and not pd.isna(last["RS_slope"]):
        if last["RS_slope"] > 0.02:
            rs_score += 2; reasons.append("Outperforming SPY")
        elif last["RS_slope"] > 0:
            rs_score += 1
        elif last["RS_slope"] < -0.02:
            rs_score -= 2; reasons.append("Underperforming SPY")
        else:
            rs_score -= 1

    factors["rel_strength"] = rs_score

    # =========================================================
    # 7. RISK-ADJUSTED (weight: 5%)
    # =========================================================
    ra_score = 0
    if not pd.isna(last["Sharpe20"]):
        if last["Sharpe20"] > 1.5:
            ra_score += 2; reasons.append(f"Strong Sharpe ({last['Sharpe20']:.1f})")
        elif last["Sharpe20"] > 0.5:
            ra_score += 1
        elif last["Sharpe20"] < -1:
            ra_score -= 2
        elif last["Sharpe20"] < -0.3:
            ra_score -= 1
    factors["risk_adj"] = ra_score

    # =========================================================
    # WEIGHTED COMPOSITE
    # =========================================================
    # Normalize each factor to -10 to +10 approximately
    weights = {
        "trend": 0.25,
        "momentum": 0.20,
        "mean_rev": 0.15,
        "volume": 0.15,
        "breakout": 0.10,
        "rel_strength": 0.10,
        "risk_adj": 0.05
    }

    # Cap individual factors at ±10
    for k in factors:
        factors[k] = max(-10, min(10, factors[k]))

    composite = sum(factors[k] * weights[k] for k in factors)
    composite = round(composite, 2)

    # =========================================================
    # LIQUIDITY FILTER (hard filter)
    # =========================================================
    avg_dollar_volume = (last["Close"] * last["VOL_MA20"])
    liquid_enough = avg_dollar_volume > 10_000_000  # $10M daily minimum

    return {
        "score": composite,
        "factors": factors,
        "reasons": reasons,
        "price": float(last["Close"]),
        "rsi": float(last["RSI"]),
        "adx": float(last["ADX"]),
        "atr": float(last["ATR"]),
        "atr_pct": float(last["ATR_pct"]),
        "zscore": float(last["Zscore"]),
        "vol_ratio": float(last["VOL_ratio"]),
        "mfi": float(last["MFI"]),
        "ret5": float(last["Ret5"]),
        "ret20": float(last["Ret20"]),
        "pct_from_52w_high": float(last["PctFromHigh52w"]),
        "sharpe20": float(last["Sharpe20"]) if not pd.isna(last["Sharpe20"]) else 0,
        "dollar_vol_m": avg_dollar_volume / 1_000_000,
        "liquid": liquid_enough,
    }

# ---------- RECOMMENDATION ----------
def recommendation(score, adx):
    """Map composite score to action. ADX adjusts holding period."""
    # Strong trend = longer hold; weak trend = shorter
    if adx > 30:
        hold_mult = 1.3
    elif adx < 20:
        hold_mult = 0.6
    else:
        hold_mult = 1.0

    if score >= 5:
        base_hold = 7
        return "STRONG BUY", f"{int(3*hold_mult)}-{int(10*hold_mult)} days", "🟢"
    if score >= 3:
        return "BUY", f"{int(2*hold_mult)}-{int(7*hold_mult)} days", "🟢"
    if score >= 1.5:
        return "WEAK BUY", f"{int(1*hold_mult)}-{int(3*hold_mult)} days", "🟡"
    if score > -1.5:
        return "HOLD", "Wait", "⚪"
    if score > -3:
        return "WEAK SELL", f"{int(1*hold_mult)}-{int(3*hold_mult)} days", "🟡"
    if score > -5:
        return "SELL", f"{int(2*hold_mult)}-{int(7*hold_mult)} days", "🔴"
    return "STRONG SELL", f"{int(3*hold_mult)}-{int(10*hold_mult)} days", "🔴"

def compute_targets(price, atr, score):
    """ATR-based stops & targets with score-adjusted R:R."""
    # Higher conviction = wider target
    target_mult = 2.5 + min(abs(score), 5) * 0.3  # 2.5-4.0x ATR
    stop_mult = 1.5

    if score > 0:
        stop = price - stop_mult * atr
        target = price + target_mult * atr
    else:
        stop = price + stop_mult * atr
        target = price - target_mult * atr
    rr = abs(target - price) / abs(price - stop) if price != stop else 0
    return stop, target, rr

# ---------- TICKER ANALYSIS ----------
def analyze_ticker(ticker, spy_df, sector_map):
    df = fetch_data(ticker)
    if df is None:
        return None
    try:
        df = compute_all_factors(df, spy_df)
        s = score_stock(df, spy_df)
        if s is None or not s["liquid"]:
            return None

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
        }
    except Exception as e:
        return None

def run_full_scan(tickers, spy_df, sector_map, max_workers=20):
    results = []
    prog = st.progress(0, text="Initializing scan...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_ticker, t, spy_df, sector_map): t for t in tickers}
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            if r:
                results.append(r)
            prog.progress((i+1)/len(tickers),
                         text=f"Scanning {i+1}/{len(tickers)} — {len(results)} valid")
    prog.empty()
    return pd.DataFrame(results)

# ---------- CORRELATION FILTER ----------
def diversify_picks(df, max_per_sector=3):
    """Limit picks from the same sector to avoid concentration risk."""
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
st.caption("Full S&P 500 multi-factor scanner — unbiased, sector-diversified")

with st.expander("⚙️ Settings", expanded=False):
    scan_size = st.radio(
        "Scan Size",
        ["Full S&P 500 (~500, 3-5 min)",
         "Sample 100 (random, 45s)",
         "Sample 50 (random, 25s)",
         "Custom"],
        index=0
    )
    if scan_size == "Custom":
        custom = st.text_area("Tickers (comma-separated)", "AAPL,MSFT,NVDA")
        tickers_to_scan = [t.strip().upper() for t in custom.split(",")]
    elif scan_size.startswith("Full"):
        tickers_to_scan = get_sp500_tickers()
    elif "100" in scan_size:
        np.random.seed(None)
        tickers_to_scan = list(np.random.choice(get_sp500_tickers(), 100, replace=False))
    else:
        tickers_to_scan = list(np.random.choice(get_sp500_tickers(), 50, replace=False))

    st.markdown("---")
    min_score = st.slider("Min |score| to display", 0.0, 8.0, 2.5, 0.5)
    max_per_sector = st.slider("Max picks per sector (diversification)", 1, 10, 3)
    min_dollar_vol = st.slider("Min daily $ volume (millions)", 5, 500, 20, 5)

    st.markdown("---")
    st.markdown("**Factor Weights (advanced)**")
    show_factors = st.checkbox("Show individual factor scores in table", False)

if st.button("🚀 Run Full Scan", type="primary", use_container_width=True):
    with st.spinner("Loading SPY benchmark..."):
        spy_df = fetch_benchmark()
        sector_map = get_sector_map()
    st.session_state.scan_results = run_full_scan(tickers_to_scan, spy_df, sector_map)
    st.session_state.scan_time = datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------- RESULTS ----------
if "scan_results" in st.session_state:
    df_res = st.session_state.scan_results
    if df_res.empty:
        st.error("No valid results. Check your internet or try again.")
    else:
        # Apply liquidity filter
        df_res = df_res[df_res["$Vol (M)"] >= min_dollar_vol].copy()
        df_res = df_res.sort_values("Score", key=abs, ascending=False)

        st.caption(f"📅 Last scan: {st.session_state.scan_time} | "
                   f"Scanned {len(df_res)} liquid stocks")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Buys", (df_res["Score"] >= 1.5).sum())
        c2.metric("🔴 Sells", (df_res["Score"] <= -1.5).sum())
        c3.metric("⭐ Strong", (df_res["Score"].abs() >= 5).sum())
        c4.metric("📊 Total", len(df_res))

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Top Picks", "📊 Factor Breakdown", "📋 All Data", "📈 Chart"])

        # --- TAB 1: DIVERSIFIED TOP PICKS ---
        with tab1:
            filt = df_res[df_res["Score"].abs() >= min_score]
            longs = filt[filt["Score"] > 0].sort_values("Score", ascending=False)
            shorts = filt[filt["Score"] < 0].sort_values("Score")

            longs_div = diversify_picks(longs, max_per_sector)
            shorts_div = diversify_picks(shorts, max_per_sector)

            st.subheader(f"🟢 Long Ideas ({len(longs_div)} diversified)")
            for _, r in longs_div.head(15).iterrows():
                st.markdown(f"""
                <div class='signal-card'>
                <h3 style='margin:0'>{r['Emoji']} {r['Ticker']} — {r['Signal']} <span style='font-size:0.7em;color:#888'>[{r['Sector']}]</span></h3>
                <p style='margin:4px 0;color:#aaa'>Score: <b>{r['Score']:+.1f}</b> | Hold: <b>{r['Hold']}</b> | R:R: <b>{r['R:R']}</b></p>
                <p style='margin:4px 0'>💰 ${r['Price']} → 🎯 ${r['Target']} | 🛑 ${r['Stop']}</p>
                <p style='margin:4px 0;font-size:0.8rem;color:#aaa'>
                RSI {r['RSI']} • ADX {r['ADX']} • Z {r['Z-Score']:+.1f} • 
                5D {r['5D%']:+.1f}% • From 52wH {r['From 52w High%']:+.1f}%
                </p>
                <p style='margin:4px 0;font-size:0.85rem;color:#ccc'>{r['Reasons']}</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader(f"🔴 Short Ideas ({len(shorts_div)} diversified)")
            for _, r in shorts_div.head(15).iterrows():
                st.markdown(f"""
                <div class='signal-card sell-card'>
                <h3 style='margin:0'>{r['Emoji']} {r['Ticker']} — {r['Signal']} <span style='font-size:0.7em;color:#888'>[{r['Sector']}]</span></h3>
                <p style='margin:4px 0;color:#aaa'>Score: <b>{r['Score']:+.1f}</b> | Hold: <b>{r['Hold']}</b> | R:R: <b>{r['R:R']}</b></p>
                <p style='margin:4px 0'>💰 ${r['Price']} → 🎯 ${r['Target']} | 🛑 ${r['Stop']}</p>
                <p style='margin:4px 0;font-size:0.8rem;color:#aaa'>
                RSI {r['RSI']} • ADX {r['ADX']} • Z {r['Z-Score']:+.1f} • 
                5D {r['5D%']:+.1f}% • From 52wH {r['From 52w High%']:+.1f}%
                </p>
                <p style='margin:4px 0;font-size:0.85rem;color:#ccc'>{r['Reasons']}</p>
                </div>
                """, unsafe_allow_html=True)

        # --- TAB 2: FACTOR BREAKDOWN ---
        with tab2:
            st.subheader("Factor Contribution (top signals)")
            top_for_factors = df_res.head(20)
            factor_cols = ["Ticker","Score","Trend F","Mom F","MR F","Vol F","BO F","RS F"]
            st.dataframe(top_for_factors[factor_cols], use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Factor Legend:**
            - **Trend F**: EMA alignment + ADX + SMA200 (25%)
            - **Mom F**: MACD + ROC + 5-day return (20%)
            - **MR F**: RSI + Z-score + Bollinger position (15%)
            - **Vol F**: Volume ratio + OBV + MFI (15%)
            - **BO F**: 20-day & 52-week breakout levels (10%)
            - **RS F**: Relative strength vs SPY (10%)
            """)
            
            # Sector heatmap
            st.subheader("Signal by Sector")
            sec_summary = df_res.groupby("Sector").agg(
                Count=("Ticker", "count"),
                Avg_Score=("Score", "mean"),
                Bullish=("Score", lambda x: (x > 1.5).sum()),
                Bearish=("Score", lambda x: (x < -1.5).sum())
            ).round(2).sort_values("Avg_Score", ascending=False)
            st.dataframe(sec_summary, use_container_width=True)

        # --- TAB 3: ALL DATA ---
        with tab3:
            base_cols = ["Ticker","Sector","Signal","Score","Price","Stop","Target","R:R",
                         "Hold","RSI","ADX","Z-Score","5D%","20D%","From 52w High%","$Vol (M)"]
            if show_factors:
                base_cols += ["Trend F","Mom F","MR F","Vol F","BO F","RS F"]
            st.dataframe(df_res[base_cols], use_container_width=True, hide_index=True, height=500)
            st.download_button("📥 Download CSV",
                               df_res.drop(columns=["_df"]).to_csv(index=False),
                               f"signals_{st.session_state.scan_time}.csv",
                               use_container_width=True)

        # --- TAB 4: CHART ---
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
            fig.add_trace(go.Scatter(x=d.index, y=d["BB_H"], name="BB High",
                                     line=dict(color="gray", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=d.index, y=d["BB_L"], name="BB Low",
                                     line=dict(color="gray", width=1, dash="dot")))
            fig.update_layout(
                title=f"{pick} [{row['Sector']}] — {row['Emoji']} {row['Signal']}",
                xaxis_rangeslider_visible=False, height=500,
                margin=dict(l=10,r=10,t=40,b=10),
                showlegend=True, template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{row['Score']:+.1f}")
            c2.metric("Entry", f"${row['Price']}")
            c3.metric("Stop", f"${row['Stop']}")
            c4.metric("Target", f"${row['Target']}")

            st.info(f"**Reasons:** {row['Reasons']}")
            st.caption(f"Sector: {row['Sector']} | Hold: {row['Hold']} | R:R: {row['R:R']} | "
                       f"Liquidity: ${row['$Vol (M)']:.0f}M/day")

else:
    st.info("👆 Configure settings and tap **Run Full Scan** to analyze all 500 stocks")
    st.markdown("""
    ### What this scanner does:
    - ✅ Scans **all 500 S&P stocks** (no mega-cap bias)
    - ✅ Uses **7 factor categories**: Trend, Momentum, Mean Reversion, Volume, Breakout, Relative Strength, Risk-Adjusted
    - ✅ Filters out **illiquid stocks** (min $20M daily volume)
    - ✅ **Sector diversification** (max 3 picks per sector by default)
    - ✅ Compares to **SPY benchmark** for alpha detection
    - ✅ **ATR-based stops** with score-adjusted targets
    """)

st.caption("⚠️ Educational only. Not financial advice. Data: yfinance (15-min delayed)")
