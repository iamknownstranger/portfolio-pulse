import streamlit as st
import pandas as pd
import plotly.express as px

from common.sidebar import render_sidebar


st.set_page_config(page_title="Holdings & Exposure", page_icon="📋", layout="wide")
st.title("📋 Holdings & Exposure")

symbols, start_date, end_date = render_sidebar()


# Load actual holdings data from CSV
df_holdings = pd.read_csv("/home/ubuntu/portfolio-pulse/data/companiesmarketcap.com - Largest American companies by market capitalization.csv")
df_holdings["marketcap"] = pd.to_numeric(df_holdings["marketcap"], errors='coerce')

# FILTER by symbols if available
if "Symbol" in df_holdings.columns and symbols:
    df_holdings = df_holdings[df_holdings["Symbol"].isin(symbols)]

# --- New: Key Holdings Metrics ---
total_market_cap = df_holdings["marketcap"].sum()
avg_market_cap = df_holdings["marketcap"].mean()
st.subheader("Key Holdings Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Market Cap", f"${total_market_cap:,.0f}")
col2.metric("Avg Market Cap", f"${avg_market_cap:,.0f}")

# Display Holdings Table (show top 10 for brevity)
st.subheader("Your Holdings")
st.dataframe(df_holdings.head(10), hide_index=True)

# Geographical Exposure based on the 'country' column
st.subheader("Geographical Exposure")
geo = df_holdings["country"].value_counts().reset_index()
geo.columns = ["Country", "Count"]
fig_geo = px.pie(geo, names="Country", values="Count", title="Geographical Exposure")
st.plotly_chart(fig_geo, use_container_width=True)

# Market Cap Distribution by Category
st.subheader("Market Cap Distribution")
# Define categories from quantiles of marketcap
quantiles = df_holdings["marketcap"].quantile([0.33, 0.66]).values
def cap_category(x):
    if x < quantiles[0]:
        return "Small Cap"
    elif x < quantiles[1]:
        return "Mid Cap"
    else:
        return "Large Cap"
df_holdings["Category"] = df_holdings["marketcap"].apply(cap_category)
cap_dist = df_holdings["Category"].value_counts().reset_index()
cap_dist.columns = ["Category", "Count"]
fig_caps = px.bar(cap_dist, x="Category", y="Count", title="Market Cap Distribution")
st.plotly_chart(fig_caps, use_container_width=True)

