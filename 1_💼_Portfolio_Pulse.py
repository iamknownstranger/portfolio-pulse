from datetime import date, timedelta
import random

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from streamlit_tags import st_tags
from common.sidebar import render_sidebar

st.set_page_config(page_title="Portfolio Analyzer", page_icon="💼", layout="wide")


st.title('Portfolio Pulse 💼')

# Render the common sidebar filters
symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()

if not start_date:
    start_date = date.today() - timedelta(days=3*365)
if not end_date:
    end_date = date.today()
if start_date == end_date:
    st.warning("Start date and end date can't be the same")
elif start_date > end_date:
    st.warning("Start date cannot be greater than end date")

symbols_string = ", ".join(symbols)
st.write(f"Your portfolio consists of {len(symbols)} stocks and their symbols are **{symbols_string}**")

@st.cache_data(ttl=86400)
def get_data(tickers, start, end, max_retries=3, retry_delay=5):
    import duckdb
    try:
        con = duckdb.connect("data/market_data.db")
        query = f"""
        SELECT date, symbol, close_price FROM market_data
        WHERE symbol IN ({', '.join([f'\'{ticker}\'' for ticker in tickers])})
        AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date;
        """
        df_duckdb = pd.DataFrame(con.execute(query).fetchall(), columns=['date', 'symbol', 'close_price'])
        con.close()
        if not df_duckdb.empty:
            df_duckdb = df_duckdb.pivot(index='date', columns='symbol', values='close_price')
            return df_duckdb
    except Exception as e:
        st.warning(f"DuckDB data fetch failed: {e}. Falling back to yfinance.")

    # Fallback to yfinance with retry logic
    for attempt in range(max_retries):
        try:
            df_yf = yf.download(tickers, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
            df_yf.index = df_yf.index.date
            return df_yf.dropna(axis=1, how='all')
        except Exception as e:
            if 'rate limit' in str(e).lower() or 'too many requests' in str(e).lower():
                if attempt < max_retries - 1:
                    st.warning(f"Yahoo Finance rate limited. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error("Too Many Requests from Yahoo Finance. Please try after a while.")
                    return pd.DataFrame()
            else:
                st.error(f"Error fetching data from yfinance: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

df = get_data(symbols, start_date, end_date)

# --- Fetch benchmark index data ---
def get_top100us_index(start_date, end_date):
    df = pd.read_csv("data/largest-companies-in-the-usa-by-market-cap.csv")
    df = df.sort_values("marketcap", ascending=False).head(100)
    tickers = df["Symbol"].tolist()
    try:
        df_yf = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)["Close"]
        df_yf.index = pd.to_datetime(df_yf.index)
        index_val = df_yf.mean(axis=1)
        return index_val.dropna()
    except Exception as e:
        st.warning(f"Failed to fetch Top 100 US index data: {e}")
        return pd.Series(dtype=float)

benchmark_df = None
if benchmark_symbol == "TOP100US":
    benchmark_df = get_top100us_index(start_date, end_date)
elif benchmark_symbol:
    try:
        benchmark_df = yf.download(benchmark_symbol, start=start_date, end=end_date)["Close"]
        benchmark_df = benchmark_df.dropna()
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
    except Exception as e:
        st.warning(f"Failed to fetch benchmark data: {e}")

if not df.empty and benchmark_df is not None and not benchmark_df.empty:
    st.subheader("Historical close price data")
    st.dataframe(df)
    st.subheader("Closing price chart")
    st.line_chart(df)

    # Calculate and display analytics
    st.subheader("Correlation Matrix")
    st.write("""A Coefficient of **correlation** is a statistical measure of the relationship between two variables. It varies from -1 to 1, with 1 or -1 indicating perfect correlation. A correlation value close to 0 indicates no association between the variables. A correlation matrix is a table showing correlation coefficients between variables.""")
    correlation_matrix = df.corr(method='pearson')
    correlation_heatmap = px.imshow(correlation_matrix, title='Correlation between Stocks in your portfolio')
    st.plotly_chart(correlation_heatmap, use_container_width=True)

    st.subheader("Daily Simple Returns")
    st.write("**Daily Simple Returns** is the percentage change in prices calculated daily.")
    daily_simple_return = df.pct_change(1)
    daily_simple_return.dropna(inplace=True)
    st.dataframe(daily_simple_return)

    daily_simple_return_plot = px.line(daily_simple_return, x=daily_simple_return.index,
                                       y=daily_simple_return.columns,
                                       title="Volatility in Daily Simple Returns",
                                       labels={"x": "Date", "y": "Daily Simple Returns"})
    st.plotly_chart(daily_simple_return_plot, use_container_width=True)

    st.subheader("Average Daily Returns")
    daily_avg = daily_simple_return.mean() * 100
    cols_daily_avg = st.columns(len(daily_avg) + 1)
    for col, (label, value) in zip(cols_daily_avg[1:], daily_avg.items()):
        col.metric(label, f"{value:.2f}%")

    daily_simple_return_boxplot = px.box(daily_simple_return, title="Risk Box Plot")
    st.plotly_chart(daily_simple_return_boxplot, use_container_width=True)

    st.subheader("Annualized Standard Deviation")
    st.write("**Annualized Standard Deviation** (Volatility, 252 trading days) of individual stocks in your portfolio.")
    annual_std = daily_simple_return.std() * np.sqrt(252) * 100
    cols_std = st.columns(len(annual_std) + 1)
    for col, (label, value) in zip(cols_std[1:], annual_std.items()):
        col.metric(label, f"{value:.2f}%")
    
    st.subheader("Return Per Unit Of Risk")
    st.write("After adjusting for a risk-free rate, this ratio (also called Sharpe Ratio) measures risk-adjusted return.")
    return_per_unit_risk = daily_avg / (daily_simple_return.std() * np.sqrt(252)) * 100
    cols_risk = st.columns(len(return_per_unit_risk) + 1)
    for col, (label, value) in zip(cols_risk[1:], return_per_unit_risk.items()):
        col.metric(label, f"{value:.2f}")

    daily_returns = df.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    st.subheader("📈 Cumulative Returns")
    st.line_chart(cumulative_returns)
    
    st.subheader("Modern Portfolio Theory")
    st.write("""
    **Modern Portfolio Theory** (MPT) is a mathematical framework for constructing a portfolio that maximizes expected return for a given level of risk. It explains how investors can construct portfolios to optimize or maximize expected return based on a given level of market risk.
    """)
    st.write("""
    An **Efficient Frontier** represents all possible portfolio combinations. It shows the maximum return portfolio at one end and the minimum variance portfolio at the other.
    """)
    st.image("efficient_frontier.png")

    mean = expected_returns.mean_historical_return(df)
    st.subheader("Mean Historical Return")
    cols_mean_return = st.columns(len(mean) + 1)
    for col, (label, value) in zip(cols_mean_return[1:], mean.items()):
        col.metric(label, f"{value:.2f}")

    st.subheader("Sample Covariance Matrix")
    sample_covariance_matrix = risk_models.sample_cov(df)
    st.dataframe(sample_covariance_matrix)
    sample_covariance_matrix_heatmap = px.imshow(sample_covariance_matrix, title="Sample Covariance Matrix")
    st.plotly_chart(sample_covariance_matrix_heatmap, use_container_width=True)

    ef = EfficientFrontier(mean, sample_covariance_matrix)
    try:
        weights = ef.max_sharpe()
    except ValueError as e:
        st.error(f"Error optimizing portfolio using max_sharpe: {e}. Falling back to min_volatility.")
        weights = ef.min_volatility()
    st.subheader("Optimized Weights (Maximize Sharpe Ratio)")
    cols_weights = st.columns(len(weights) + 1)
    for col, (label, value) in zip(cols_weights[1:], weights.items()):
        col.metric(label, f"{value:.2f}")
    cleaned_weights = ef.clean_weights()
    labels = list(cleaned_weights.keys())
    values = list(cleaned_weights.values())
    pie_chart = px.pie(df, values=values, names=labels, title='Optimized Portfolio Allocation')
    st.plotly_chart(pie_chart, use_container_width=True)

    st.subheader("Portfolio Performance")
    mu, sigma, sharpe = ef.portfolio_performance()
    cols_perf = st.columns(4)
    cols_perf[1].metric("Expected Annual Return", f"{100 * mu:.2f}%")
    cols_perf[2].metric("Annualized Volatility", f"{100 * sigma:.2f}%")
    cols_perf[3].metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.balloons()

    # --- New Portfolio Performance Metrics ---
    portfolio_daily = df.pct_change().dropna().mean(axis=1)
    portfolio_daily.name = "portfolio_daily"  # assign name for merging
    portfolio_cum = (1 + portfolio_daily).cumprod()
    portfolio_vol = portfolio_daily.std() * np.sqrt(252) * 100
    portfolio_max_dd = ((portfolio_cum.cummax() - portfolio_cum) / portfolio_cum.cummax() * 100).max()
    downside_std_pf = portfolio_daily[portfolio_daily < 0].std() * np.sqrt(252)
    portfolio_sortino = (portfolio_daily.mean() * 252) / downside_std_pf if downside_std_pf != 0 else np.nan

    sp500_pf = yf.download("^GSPC", start=start_date, end=end_date)
    sp500_pf["daily_return"] = sp500_pf["Close"].pct_change()
    sp500_daily = sp500_pf["daily_return"].dropna()
    # --- New: Convert indices to datetime for merging ---
    portfolio_daily.index = pd.to_datetime(portfolio_daily.index)
    sp500_daily.index = pd.to_datetime(sp500_daily.index)
    portfolio_daily.index.name = "date"
    sp500_daily.index.name = "date"
    merged_pf = pd.merge(portfolio_daily.reset_index(),
                         sp500_daily.rename("sp500_return").reset_index(),
                         on="date", how="inner")
    beta_pf = merged_pf["portfolio_daily"].cov(merged_pf["sp500_return"]) / merged_pf["sp500_return"].var() if not merged_pf.empty else np.nan

    st.subheader("📊 Portfolio Performance Metrics")
    cols_pf = st.columns(5)
    cols_pf[0].metric("Cumulative Return", f"{(portfolio_cum.iloc[-1]-1)*100:.2f}%")
    cols_pf[1].metric("Annualized Vol", f"{portfolio_vol:.2f}%")
    cols_pf[2].metric("Max Drawdown", f"{portfolio_max_dd:.2f}%")
    cols_pf[3].metric("Sortino Ratio", f"{portfolio_sortino:.2f}")
    cols_pf[4].metric("Beta vs S&P500", f"{beta_pf:.2f}")

    st.subheader("📈 Cumulative Returns (Portfolio)")
    st.line_chart(portfolio_cum)

    # --- New: Cumulative Returns: Portfolio vs Benchmark ---
    daily_returns = df.pct_change().dropna()
    portfolio_daily = daily_returns.mean(axis=1)
    portfolio_cum = (1 + portfolio_daily).cumprod()
    benchmark_daily = benchmark_df.pct_change().dropna()
    benchmark_cum = (1 + benchmark_daily).cumprod()
    print(">>> benchmark_cum", benchmark_cum, benchmark_cum)
    # Align indices for plotting
    cum_df = pd.concat([
        portfolio_cum,
        benchmark_cum
    ], axis=1, join='inner')

    st.subheader("📈 Cumulative Returns: Portfolio vs Benchmark")
    st.line_chart(cum_df)

    # --- Metrics comparison ---
    portfolio_vol = portfolio_daily.std() * np.sqrt(252) * 100
    benchmark_vol = benchmark_daily.std() * np.sqrt(252) * 100
    portfolio_cum_return = (portfolio_cum.iloc[-1] - 1) * 100
    benchmark_cum_return = (benchmark_cum.iloc[-1] - 1) * 100
    portfolio_max_dd = ((portfolio_cum.cummax() - portfolio_cum) / portfolio_cum.cummax() * 100).max()
    benchmark_max_dd = ((benchmark_cum.cummax() - benchmark_cum) / benchmark_cum.cummax() * 100).max()
    downside_std_pf = portfolio_daily[portfolio_daily < 0].std() * np.sqrt(252)
    downside_std_bm = benchmark_daily[benchmark_daily < 0].std() * np.sqrt(252)
    portfolio_sortino = (portfolio_daily.mean() * 252) / downside_std_pf if downside_std_pf != 0 else np.nan
    benchmark_sortino = (benchmark_daily.mean() * 252) / downside_std_bm if downside_std_bm != 0.0 and not np.isnan(downside_std_bm) else np.nan
    merged = pd.merge(portfolio_daily.reset_index(), benchmark_daily.rename("benchmark_daily").reset_index(), on="index", how="inner")
    beta_pf = merged["mean"].cov(merged["benchmark_daily"]) / merged["benchmark_daily"].var() if not merged.empty else np.nan

    st.subheader("📊 Performance Metrics Comparison")
    cols = st.columns(6)
    cols[0].metric("Portfolio Cum. Return", f"{portfolio_cum_return:.2f}%")
    cols[1].metric("Benchmark Cum. Return", f"{benchmark_cum_return:.2f}%")
    cols[2].metric("Portfolio Volatility", f"{portfolio_vol:.2f}%")
    cols[3].metric("Benchmark Volatility", f"{benchmark_vol:.2f}%")
    cols[4].metric("Portfolio Sortino", f"{portfolio_sortino:.2f}")
    cols[5].metric("Benchmark Sortino", f"{benchmark_sortino:.2f}")
    st.metric("Portfolio Beta vs Benchmark", f"{beta_pf:.2f}")

    st.subheader("Export Data")
    st.download_button(
        label="Download Portfolio Data (CSV)",
        data=df.to_csv().encode('utf-8'),
        file_name="portfolio_data.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Benchmark Data (CSV)",
        data=benchmark_df.to_csv().encode('utf-8'),
        file_name="benchmark_data.csv",
        mime="text/csv"
    )
    # --- Enhancement: Add toggle for dark/light theme for charts ---
    theme = st.radio("Chart Theme", ["Light", "Dark"], index=1, horizontal=True)
    chart_template = "plotly_dark" if theme == "Dark" else "plotly_white"
    
    correlation_heatmap = px.imshow(correlation_matrix, title='Correlation between Stocks in your portfolio', template=chart_template)
    st.plotly_chart(correlation_heatmap, use_container_width=True)
    
    daily_simple_return_plot = px.line(daily_simple_return, x=daily_simple_return.index,
                                       y=daily_simple_return.columns,
                                       title="Volatility in Daily Simple Returns",
                                       labels={"x": "Date", "y": "Daily Simple Returns"},
                                       template=chart_template)
    st.plotly_chart(daily_simple_return_plot, use_container_width=True)
    
    daily_simple_return_boxplot = px.box(daily_simple_return, title="Risk Box Plot", template=chart_template)
    st.plotly_chart(daily_simple_return_boxplot, use_container_width=True)
    
    sample_covariance_matrix_heatmap = px.imshow(sample_covariance_matrix, title="Sample Covariance Matrix", template=chart_template)
    st.plotly_chart(sample_covariance_matrix_heatmap, use_container_width=True)
    
    pie_chart = px.pie(df, values=values, names=labels, title='Optimized Portfolio Allocation', template=chart_template)
    st.plotly_chart(pie_chart, use_container_width=True)
    
    st.line_chart(cumulative_returns, use_container_width=True)
    
    st.line_chart(portfolio_cum, use_container_width=True)
    
    st.line_chart(cum_df, use_container_width=True)
    
# --- Enhancement: Add app info/help section ---
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**\n
- Select your portfolio stocks and date range in the sidebar.\n- Choose a benchmark for comparison.\n- Analyze performance, risk, and optimization results.\n- Download your data for further analysis.\n\n**Tip:** Use the chart theme toggle for better visibility in different environments.
""")

