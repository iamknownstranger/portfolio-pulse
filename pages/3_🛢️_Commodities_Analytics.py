import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Commodities Analytics", layout="wide")
st.title("🛢️ Commodities Analytics")

# List of top commodities (Yahoo Finance tickers)
COMMODITIES = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Wheat": "ZW=F"
}

# Add DXY for correlation
DXY_TICKER = "DX-Y.NYB"

with st.sidebar:
    st.header("Commodities Selection")
    selected_commodities = st.multiselect(
        "Select commodities to analyze:",
        options=list(COMMODITIES.keys()),
        default=["Gold", "Silver", "Crude Oil (WTI)"]
    )
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=3*365))
    end_date = st.date_input("End date", value=date.today())
    analyze_button = st.button("Analyze")

st.markdown("""
<style>
.metric-label { font-size: 1.1em; color: #888; }
.metric-value { font-size: 1.5em; font-weight: bold; }
.section-title { font-size: 1.3em; font-weight: bold; margin-top: 2em; }
</style>
""", unsafe_allow_html=True)

if analyze_button and selected_commodities:
    tickers = [COMMODITIES[c] for c in selected_commodities]
    df = yf.download(tickers, start=start_date, end=end_date)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna()

    st.markdown('<div class="section-title">Price History</div>', unsafe_allow_html=True)
    st.dataframe(df)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title="Historical Close Prices", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Daily returns
    daily_returns = df.pct_change().dropna()
    st.markdown('<div class="section-title">Daily Returns</div>', unsafe_allow_html=True)
    st.line_chart(daily_returns)

    # Cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()
    st.markdown('<div class="section-title">Cumulative Returns</div>', unsafe_allow_html=True)
    st.line_chart(cumulative_returns)

    # Seasonality: Average monthly returns
    st.markdown('<div class="section-title">Seasonality: Average Monthly Returns</div>', unsafe_allow_html=True)
    monthly = daily_returns.copy()
    monthly['month'] = pd.to_datetime(monthly.index).month
    seasonality = monthly.groupby('month').mean() * 100
    st.bar_chart(seasonality)

    # Rolling volatility (30d)
    st.markdown('<div class="section-title">Rolling 30-Day Volatility</div>', unsafe_allow_html=True)
    rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252) * 100
    st.line_chart(rolling_vol)

    # Skewness & Kurtosis
    st.markdown('<div class="section-title">Distribution Metrics</div>', unsafe_allow_html=True)
    skewness = daily_returns.skew()
    kurtosis = daily_returns.kurtosis()
    cols = st.columns(len(skewness) + 1)
    for col, (label, value) in zip(cols[1:], skewness.items()):
        col.metric(f"{label} Skew", f"{value:.2f}")
    cols2 = st.columns(len(kurtosis) + 1)
    for col, (label, value) in zip(cols2[1:], kurtosis.items()):
        col.metric(f"{label} Kurtosis", f"{value:.2f}")

    # Drawdown stats
    st.markdown('<div class="section-title">Drawdown Analysis</div>', unsafe_allow_html=True)
    drawdown = (df / df.cummax() - 1) * 100
    st.line_chart(drawdown)
    max_drawdown = drawdown.min()
    cols_dd = st.columns(len(max_drawdown) + 1)
    for col, (label, value) in zip(cols_dd[1:], max_drawdown.items()):
        col.metric(f"{label} Max Drawdown", f"{value:.2f}%")

    # Correlation with DXY (USD Index)
    st.markdown('<div class="section-title">Correlation with US Dollar Index (DXY)</div>', unsafe_allow_html=True)
    dxy = yf.download(DXY_TICKER, start=start_date, end=end_date)["Close"].pct_change().dropna()
    dxy = dxy.reindex(daily_returns.index).dropna()
    corr_dxy = daily_returns.corrwith(dxy)
    cols_corr = st.columns(len(corr_dxy) + 1)
    for col, (label, value) in zip(cols_corr[1:], corr_dxy.items()):
        col.metric(f"{label} vs DXY", f"{value:.2f}")

    # Boxplot of daily returns
    st.markdown('<div class="section-title">Risk Distribution (Boxplot)</div>', unsafe_allow_html=True)
    st.plotly_chart(px.box(daily_returns, title="Daily Returns Boxplot"), use_container_width=True)

    # --- Term Structure: Contango/Backwardation ---
    st.markdown('<div class="section-title">Term Structure: Contango vs. Backwardation</div>', unsafe_allow_html=True)
    term_structure_results = {}
    for name, ticker in zip(selected_commodities, tickers):
        # Try to get next-month contract by convention (e.g., CL=F for front, CLM25.NYM for next month)
        # yfinance may not always provide next-month contracts, so fallback to front if not found
        next_month_ticker = ticker.replace('=F', 'M25.NYM') if '=F' in ticker else ticker
        try:
            front = yf.Ticker(ticker).history(period="5d")['Close'].iloc[-1]
            next_month = yf.Ticker(next_month_ticker).history(period="5d")['Close'].iloc[-1]
            if np.isnan(next_month) or next_month == 0:
                raise Exception('No next-month data')
            if front < next_month:
                structure = 'Contango'
            elif front > next_month:
                structure = 'Backwardation'
            else:
                structure = 'Flat'
            term_structure_results[name] = (structure, front, next_month)
        except Exception:
            term_structure_results[name] = ("N/A", front, None)
    cols_ts = st.columns(len(term_structure_results) + 1)
    for col, (name, (structure, front, next_month)) in zip(cols_ts[1:], term_structure_results.items()):
        if structure == 'Contango':
            color = 'orange'
        elif structure == 'Backwardation':
            color = 'green'
        elif structure == 'Flat':
            color = 'gray'
        else:
            color = 'lightgray'
        col.markdown(f"<div style='color:{color};font-weight:bold'>{name}: {structure}</div>", unsafe_allow_html=True)
        col.caption(f"Front: {front:.2f}  Next: {next_month if next_month else 'N/A'}")

    # --- Additional Informative Metrics ---
    st.markdown('<div class="section-title">Additional Commodity Metrics</div>', unsafe_allow_html=True)
    for name, ticker in zip(selected_commodities, tickers):
        st.markdown(f"**{name} ({ticker})**")
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            st.warning(f"No data for {name}")
            continue
        last_close = hist['Close'].iloc[-1]
        high_1y = hist['Close'].max()
        low_1y = hist['Close'].min()
        avg_vol = hist['Volume'].mean()
        pos = (last_close - low_1y) / (high_1y - low_1y) * 100 if high_1y != low_1y else 0
        vol_30d = hist['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100
        vol_1y = hist['Close'].pct_change().std() * np.sqrt(252) * 100
        vol_skew = vol_30d / vol_1y if vol_1y else np.nan
        # Rolling correlation with DXY
        dxy_hist = yf.Ticker(DXY_TICKER).history(start=hist.index[0], end=hist.index[-1])['Close']
        dxy_ret = dxy_hist.pct_change().dropna()
        comm_ret = hist['Close'].pct_change().dropna()
        join_idx = comm_ret.index.intersection(dxy_ret.index)
        rolling_corr = comm_ret[join_idx].rolling(60).corr(dxy_ret[join_idx])
        st.write(f"Last Close: {last_close:.2f} | 1Y High: {high_1y:.2f} | 1Y Low: {low_1y:.2f} | Position in 1Y Range: {pos:.1f}%")
        st.write(f"Avg Daily Volume: {avg_vol:,.0f}")
        st.write(f"30D Volatility: {vol_30d:.2f}% | 1Y Volatility: {vol_1y:.2f}% | Volatility Skew (30D/1Y): {vol_skew:.2f}")
        st.line_chart(rolling_corr, height=100)
        st.caption("60-day rolling correlation with DXY (USD Index)")
        st.markdown("---")