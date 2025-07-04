import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import norm
from common.sidebar import render_sidebar
import yfinance as yf

st.set_page_config(page_title="Risk Wall", page_icon="⚠️", layout="wide")
st.title("⚠️ Risk Wall")

symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()

@st.cache_data(ttl=86400)
def get_data(tickers, start, end):
    try:
        # Attempt to fetch data from DuckDB
        query = f"""
        SELECT date, symbol, close_price FROM market_data
        WHERE symbol IN ({', '.join([f'\'{ticker}\'' for ticker in tickers])})
        AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date;
        """
        df_duckdb = pd.DataFrame(con.execute(query).fetchall(), columns=['date', 'symbol', 'close_price'])
        if not df_duckdb.empty:
            df_duckdb = df_duckdb.pivot(index='date', columns='symbol', values='close_price')
            return df_duckdb
    except Exception as e:
        st.warning(f"DuckDB data fetch failed: {e}. Falling back to yfinance.")

    # Fallback to yfinance if DuckDB fails
    try:
        df_yf = yf.download(tickers, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
        df_yf.index = df_yf.index.date
        return df_yf.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame()

df = get_data(symbols, start_date, end_date)

# Load actual data
df_companies = pd.read_csv("data/companiesmarketcap.com - Largest American companies by market capitalization.csv")
df_companies["marketcap"] = pd.to_numeric(df_companies["marketcap"], errors='coerce')
df_companies["price (INR)"] = pd.to_numeric(df_companies["price (INR)"], errors='coerce')

# FILTER: keep only selected symbols if provided and if the CSV has a "Symbol" column.
if "Symbol" in df_companies.columns and symbols:
    df_companies = df_companies[df_companies["Symbol"].isin(symbols)]

# --- Risk Metrics ---
# Compute daily returns and portfolio returns
returns = df.pct_change().dropna()
portfolio_returns = returns.mean(axis=1)

# Annualized volatility from daily returns
volatility = portfolio_returns.std() * np.sqrt(252)

# Historical VaR (95% confidence, i.e. 5th percentile) and CVaR
var_value = -np.percentile(portfolio_returns, 5) * 100
cvar_value = -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * 100

# EWMA volatility and VaR using decay factor lambda = 0.94
lambda_ = 0.94
# Weights: most recent day gets weight 1, older days get lower weights
weights = np.array([lambda_**i for i in range(len(portfolio_returns)-1, -1, -1)])
sigma_ewma = np.sqrt(np.sum(weights * portfolio_returns**2) / np.sum(weights)) * np.sqrt(252)
ewma_var = abs(norm.ppf(0.05)) * sigma_ewma * 100

# Stress Loss: worst daily return in percentage terms
stress_loss = -np.min(portfolio_returns) * 100

# Sharpe Ratio (assuming risk-free rate = 0)
sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility != 0 else np.nan

# Display risk metrics in two rows
cols1 = st.columns(3)
cols1[0].metric("Portfolio Volatility", f"{volatility*100:.2f}%")
cols1[1].metric("Historical VaR (95%)", f"{var_value:.2f}%")
cols1[2].metric("CVaR (95%)", f"{cvar_value:.2f}%")

cols2 = st.columns(3)
cols2[0].metric("EWMA VaR (95%)", f"{ewma_var:.2f}%")
cols2[1].metric("Stress Loss", f"{stress_loss:.2f}%")
cols2[2].metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

# --- Drawdown Analysis (using portfolio returns) ---
cum_returns = (1 + portfolio_returns).cumprod()
running_max = cum_returns.cummax()
drawdown_series = (cum_returns - running_max) / running_max * 100
max_drawdown = drawdown_series.min()
st.subheader("Drawdown Analysis")
fig_dd = px.line(x=cum_returns.index, y=drawdown_series, 
                 title="Portfolio Drawdown Over Time",
                 labels={"x": "Date", "y": "Drawdown (%)"})
st.plotly_chart(fig_dd, use_container_width=True)
st.metric("Max Drawdown", f"{max_drawdown:.2f}%")


# Calculate and display analytics
st.subheader("Correlation Matrix")
st.write("""A Coefficient of **correlation** is a statistical measure of the relationship between two variables. It varies from -1 to 1, with 1 or -1 indicating perfect correlation. A correlation value close to 0 indicates no association between the variables. A correlation matrix is a table showing correlation coefficients between variables.""")
correlation_matrix = df.corr(method='pearson')
correlation_heatmap = px.imshow(correlation_matrix, title='Correlation between Stocks in your portfolio')
st.plotly_chart(correlation_heatmap, use_container_width=True)


# --- Stress Testing (using a -10% shock on portfolio returns) ---
stress_returns = portfolio_returns - 0.10
stress_cum_returns = (1 + stress_returns).cumprod()
stress_drawdown = (stress_cum_returns - stress_cum_returns.cummax()) / stress_cum_returns.cummax() * 100
max_stress_drawdown = stress_drawdown.min()
st.subheader("Stress Testing")
st.write("Stress Test: Scenario applying a -10% shock to portfolio daily returns")
fig_stress = px.line(x=stress_cum_returns.index, y=stress_drawdown, 
                     title="Stress Test Drawdown",
                     labels={"x": "Date", "y": "Drawdown (%)"})
st.plotly_chart(fig_stress, use_container_width=True)
st.metric("Max Stress Drawdown", f"{max_stress_drawdown:.2f}%")

# --- Portfolio Volatility Plot ---
rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
st.subheader("Rolling Portfolio Volatility (30-day)")
fig_vol = px.line(x=rolling_vol.index, y=rolling_vol, 
                  title="30-Day Rolling Portfolio Volatility",
                  labels={"x": "Date", "y": "Volatility (%)"})
st.plotly_chart(fig_vol, use_container_width=True)

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

