import streamlit as st
import pandas as pd
import numpy as np
from common.data import get_price_data, load_companies_csv
from common.sidebar import render_sidebar
from pypfopt import expected_returns, risk_models, EfficientFrontier, CLA
import plotly.graph_objects as go

st.set_page_config(page_title="Rebalancing & Optimization", page_icon="🔄", layout="wide")
st.title("🔄 Rebalancing & Optimization")

symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()

# Load actual data
df_companies = load_companies_csv()

# FILTER by sidebar selection if available
if "Symbol" in df_companies.columns and symbols:
    df_companies = df_companies[df_companies["Symbol"].isin(symbols)]

total = df_companies["marketcap"].sum()

# Use market cap proportion as current allocation percentage
df_companies["Current %"] = df_companies["marketcap"] / total * 100
# Assume target allocation is the median of current percentages
target = df_companies["Current %"].median()
df_companies["Target %"] = target

# --- New: Rebalancing Metrics ---
df_companies["Allocation Diff"] = abs(df_companies["Current %"] - df_companies["Target %"])
avg_diff = df_companies["Allocation Diff"].mean()
st.subheader("Rebalancing Metrics")
st.metric("Avg Allocation Deviation", f"{avg_diff:.2f}%")

# Show top 10 companies for rebalancing analysis
df_alloc_top10 = df_companies.sort_values("marketcap", ascending=False).head(10)[["Name", "Current %", "Target %"]]
st.subheader("Current vs Target Allocation")
st.dataframe(df_alloc_top10, hide_index=True)

st.subheader("Rebalancing Suggestions")
st.write("Companies with Current % above target may be overrepresented. Consider selling some shares, while companies below target may be underweighted.")

# Updated to calculate real values for Pre-Optimization and Post-Optimization Sharpe Ratios
def calculate_sharpe_ratio(df):
    daily_returns = df.pct_change().dropna()
    portfolio_daily = daily_returns.mean(axis=1)
    annual_return = portfolio_daily.mean() * 252
    annual_volatility = portfolio_daily.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan
    return sharpe_ratio

# Fetch data for the selected symbols
data = get_price_data(symbols, start_date, end_date)
if not data.empty:
    pre_optimization_sharpe = calculate_sharpe_ratio(data)

    # Perform portfolio optimization using PyPortfolioOpt
    mean_returns = expected_returns.mean_historical_return(data)
    covariance_matrix = risk_models.sample_cov(data)
    ef = EfficientFrontier(mean_returns, covariance_matrix)
    try:
        ef.max_sharpe()
    except Exception as e:
        st.warning(f"Max-Sharpe optimization failed ({e}). Falling back to minimum volatility.")
        ef.min_volatility()
    cleaned_weights = ef.clean_weights()

    # Calculate Post-Optimization Sharpe Ratio
    post_optimization_sharpe = ef.portfolio_performance()[2]

    # Update the metrics display
    st.subheader("Impact on Key Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Pre-Optimization Sharpe", f"{pre_optimization_sharpe:.2f}")
    col2.metric("Post-Optimization Sharpe", f"{post_optimization_sharpe:.2f}")

    # Updated to display Optimized Portfolio Weights using st.metric
    # Display optimized weights
    st.subheader("Optimized Portfolio Weights")
    for symbol, weight in cleaned_weights.items():
        st.metric(label=symbol, value=f"{weight:.2%}")

    # Generate Efficient Frontier using PyPortfolioOpt
    try:
        cla = CLA(mean_returns, covariance_matrix)
        cla.max_sharpe()
        efficient_frontier = cla.efficient_frontier()

        # Extract risks and returns correctly
        risks = [point[0] for point in efficient_frontier]
        returns = [point[1] for point in efficient_frontier]

        # Plot the Efficient Frontier using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=risks, y=returns, mode='lines', name='Efficient Frontier'))
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.subheader("Efficient Frontier")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute the efficient frontier for this selection: {e}")

else:
    st.warning("No price data available for the selected stocks and date range, so optimization is unavailable.")

st.subheader("Optimization Tools")
st.write("User-defined constraints and advanced optimization options will be provided here.")

