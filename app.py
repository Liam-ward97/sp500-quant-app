
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
