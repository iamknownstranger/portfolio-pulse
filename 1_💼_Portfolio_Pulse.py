from datetime import date, timedelta
import random
import time

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
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
    """
    Fetches historical stock data from DuckDB or Yahoo Finance with retry logic.
    """
    # First, try to fetch data from the local DuckDB database.
    try:
        con = duckdb.connect("data/market_data.db")
        query = f"""
        SELECT date, symbol, close_price FROM market_data
        WHERE symbol IN ({', '.join([f"'{ticker}'" for ticker in tickers])})
        AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date;
        """
        df_duckdb = pd.DataFrame(con.execute(query).fetchall(), columns=['date', 'symbol', 'close_price'])
        con.close()
        if not df_duckdb.empty:
            # Pivot the table to have symbols as columns
            df_duckdb = df_duckdb.pivot(index='date', columns='symbol', values='close_price')
            return df_duckdb
    except Exception as e:
        st.warning(f"DuckDB data fetch failed: {e}. Falling back to yfinance.")

    # Fallback to yfinance if DuckDB fails or returns no data.
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
                else:
                    st.error("Too Many Requests from Yahoo Finance. Please try again later.")
                    return pd.DataFrame()
            else:
                st.error(f"Error fetching data from yfinance: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

df = get_data(symbols, start_date, end_date)

@st.cache_data(ttl=86400)
def get_benchmark_data(symbol, start, end):
    """
    Fetches historical data for the selected benchmark index.
    """
    if symbol == "TOP100US":
        # Logic to get the top 100 US companies by market cap and calculate an equal-weighted index.
        df_companies = pd.read_csv("data/largest-companies-in-the-usa-by-market-cap.csv")
        df_companies = df_companies.sort_values("marketcap", ascending=False).head(100)
        tickers = df_companies["Symbol"].tolist()
        try:
            df_yf = yf.download(tickers, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
            df_yf.index = pd.to_datetime(df_yf.index)
            index_val = df_yf.mean(axis=1)
            return index_val.dropna()
        except Exception as e:
            st.warning(f"Failed to fetch Top 100 US index data: {e}")
            return pd.Series(dtype=float)
    else:
        try:
            df_yf = yf.download(symbol, start=start, end=end)["Close"]
            df_yf = df_yf.dropna()
            df_yf.index = pd.to_datetime(df_yf.index)
            return df_yf
        except Exception as e:
            st.warning(f"Failed to fetch benchmark data for {symbol}: {e}")
            return pd.Series(dtype=float)


benchmark_df = get_benchmark_data(benchmark_symbol, start_date, end_date)

if not df.empty and benchmark_df is not None and not benchmark_df.empty:
    st.subheader("Historical Close Price Data")
    st.dataframe(df)
    st.subheader("Closing Price Chart")
    st.line_chart(df)

    # --- ENHANCEMENT: Added a toggle for chart themes for better readability ---
    theme = st.radio("Chart Theme", ["Light", "Dark"], index=1, horizontal=True)
    chart_template = "plotly_dark" if theme == "Dark" else "plotly_white"

    st.subheader("Correlation Matrix")
    st.write("A **correlation coefficient** is a statistical measure of the relationship between two variables. It ranges from -1 to 1. A value of 1 or -1 indicates a perfect correlation, while a value close to 0 suggests no association. A correlation matrix helps in understanding the strength of relationships between stocks in a portfolio, which is crucial for effective diversification.")
    correlation_matrix = df.corr(method='pearson')
    correlation_heatmap = px.imshow(correlation_matrix, title='Correlation Between Stocks in Your Portfolio', template=chart_template)
    st.plotly_chart(correlation_heatmap, use_container_width=True)

    st.subheader("Daily Simple Returns")
    st.write("**Daily Simple Returns** represent the daily percentage change in the stock prices.")
    daily_simple_return = df.pct_change(1).dropna()
    st.dataframe(daily_simple_return)

    daily_simple_return_plot = px.line(daily_simple_return, x=daily_simple_return.index,
                                       y=daily_simple_return.columns,
                                       title="Volatility in Daily Simple Returns",
                                       labels={"x": "Date", "y": "Daily Simple Returns"},
                                       template=chart_template)
    st.plotly_chart(daily_simple_return_plot, use_container_width=True)

    st.subheader("Average Daily Returns")
    daily_avg = daily_simple_return.mean() * 100
    # --- ENHANCEMENT: Using columns for a cleaner metric display ---
    cols_daily_avg = st.columns(len(daily_avg))
    for col, (label, value) in zip(cols_daily_avg, daily_avg.items()):
        col.metric(label, f"{value:.2f}%")

    daily_simple_return_boxplot = px.box(daily_simple_return, title="Risk Box Plot", template=chart_template)
    st.plotly_chart(daily_simple_return_boxplot, use_container_width=True)

    st.subheader("Annualized Standard Deviation (Volatility)")
    st.write("This metric shows the volatility of individual stocks in your portfolio over a year (252 trading days).")
    annual_std = daily_simple_return.std() * np.sqrt(252) * 100
    cols_std = st.columns(len(annual_std))
    for col, (label, value) in zip(cols_std, annual_std.items()):
        col.metric(label, f"{value:.2f}%")

    st.subheader("Return Per Unit Of Risk (Sharpe Ratio)")
    st.write("This ratio measures risk-adjusted return. A higher value is better, as it indicates more return for the amount of risk taken.")
    return_per_unit_risk = (daily_simple_return.mean() * 252) / (daily_simple_return.std() * np.sqrt(252))
    cols_risk = st.columns(len(return_per_unit_risk))
    for col, (label, value) in zip(cols_risk, return_per_unit_risk.items()):
        col.metric(label, f"{value:.2f}")

    cumulative_returns = (1 + daily_simple_return).cumprod()
    st.subheader("📈 Cumulative Returns of Individual Stocks")
    st.line_chart(cumulative_returns)

    st.subheader("Modern Portfolio Theory")
    st.write("""
    **Modern Portfolio Theory (MPT)** is a framework for creating portfolios that maximize expected return for a given level of risk. It's based on the idea that investors can build diversified portfolios to optimize returns while managing risk.
    The **Efficient Frontier** represents the set of optimal portfolios that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return.
    """)
    st.image("efficient_frontier.png", caption="The Efficient Frontier")

    mean_returns = expected_returns.mean_historical_return(df)
    st.subheader("Mean Historical Return")
    cols_mean_return = st.columns(len(mean_returns))
    for col, (label, value) in zip(cols_mean_return, mean_returns.items()):
        col.metric(label, f"{value*100:.2f}%")

    st.subheader("Sample Covariance Matrix")
    sample_covariance_matrix = risk_models.sample_cov(df)
    st.dataframe(sample_covariance_matrix)
    sample_covariance_matrix_heatmap = px.imshow(sample_covariance_matrix, title="Sample Covariance Matrix", template=chart_template)
    st.plotly_chart(sample_covariance_matrix_heatmap, use_container_width=True)

    ef = EfficientFrontier(mean_returns, sample_covariance_matrix)
    try:
        weights = ef.max_sharpe()
    except Exception as e:
        st.error(f"Error optimizing portfolio with max_sharpe: {e}. Falling back to min_volatility.")
        weights = ef.min_volatility()

    st.subheader("Optimized Weights (Maximize Sharpe Ratio)")
    cleaned_weights = ef.clean_weights()
    cols_weights = st.columns(len(cleaned_weights))
    for col, (label, value) in zip(cols_weights, cleaned_weights.items()):
        col.metric(label, f"{value:.2%}")

    pie_chart = px.pie(values=list(cleaned_weights.values()), names=list(cleaned_weights.keys()),
                       title='Optimized Portfolio Allocation', template=chart_template)
    st.plotly_chart(pie_chart, use_container_width=True)

    st.subheader("Optimized Portfolio Performance")
    mu, sigma, sharpe = ef.portfolio_performance()
    cols_perf = st.columns(3)
    cols_perf[0].metric("Expected Annual Return", f"{100 * mu:.2f}%")
    cols_perf[1].metric("Annualized Volatility", f"{100 * sigma:.2f}%")
    cols_perf[2].metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.balloons()

    # --- ENHANCEMENT: Added more detailed portfolio vs. benchmark comparison ---
    st.subheader("📊 Portfolio vs. Benchmark Performance")
    portfolio_daily_returns = df.pct_change().mean(axis=1).dropna()
    portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()

    benchmark_daily_returns = benchmark_df.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()

    comparison_df = pd.DataFrame({
        'Portfolio': portfolio_cumulative_returns,
        'Benchmark': benchmark_cumulative_returns
    }).dropna()

    st.line_chart(comparison_df)

    st.subheader("📊 Performance Metrics Comparison")
    # Calculate portfolio metrics
    portfolio_annual_return = portfolio_daily_returns.mean() * 252
    portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
    portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility != 0 else 0

    # Calculate benchmark metrics
    benchmark_annual_return = benchmark_daily_returns.mean() * 252
    benchmark_volatility = benchmark_daily_returns.std() * np.sqrt(252)
    benchmark_sharpe = benchmark_annual_return / benchmark_volatility if benchmark_volatility != 0 else 0

    cols_comparison = st.columns(2)
    with cols_comparison[0]:
        st.subheader("Portfolio")
        st.metric("Annual Return", f"{portfolio_annual_return*100:.2f}%")
        st.metric("Annual Volatility", f"{portfolio_volatility*100:.2f}%")
        st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")

    with cols_comparison[1]:
        st.subheader("Benchmark")
        st.metric("Annual Return", f"{benchmark_annual_return*100:.2f}%")
        st.metric("Annual Volatility", f"{benchmark_volatility*100:.2f}%")
        st.metric("Sharpe Ratio", f"{benchmark_sharpe:.2f}")

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