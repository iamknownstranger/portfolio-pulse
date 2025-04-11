import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from common.sidebar import render_sidebar

st.set_page_config(page_title="Rebalancing & Optimization", page_icon="🔄", layout="wide")
st.title("🔄 Rebalancing & Optimization")

symbols, start_date, end_date = render_sidebar()

# Load actual data
df_companies = pd.read_csv("/home/ubuntu/portfolio-pulse/data/companiesmarketcap.com - Largest American companies by market capitalization.csv")
df_companies["marketcap"] = pd.to_numeric(df_companies["marketcap"], errors='coerce')

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

st.subheader("Impact on Key Metrics")
# ...existing code...
col1, col2 = st.columns(2)
col1.metric("Pre-Optimization Sharpe", "X.XX")
col2.metric("Post-Optimization Sharpe", "Y.YY")

st.subheader("Efficient Frontier")
# Simulate an efficient frontier using random data (as actual optimization data is not available in the CSV)
df_frontier = pd.DataFrame({
    "Risk": np.linspace(0.1, 0.5, 50),
    "Return": np.linspace(0.05, 0.25, 50)
})
fig_frontier = px.scatter(df_frontier, x="Risk", y="Return",
                          title="Efficient Frontier", trendline="ols")
st.plotly_chart(fig_frontier, use_container_width=True)

st.subheader("Optimization Tools")
st.write("User-defined constraints and advanced optimization options will be provided here.")

