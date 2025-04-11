import duckdb
import polars as pl
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from common.sidebar import render_sidebar

con = duckdb.connect("data/market_cap_data.db")
INCEPTION_DATE = con.execute("SELECT min(date) FROM market_cap_data").fetchone()[0]

def compute_equal_weighted_index(df):
    df_sorted = df.sort(["date", "market_cap"], descending=[True, True])
    df_top100 = df_sorted.group_by("date").agg(
        pl.col("market_cap").head(100).alias("top_100_market_cap"),
        pl.col("symbol").head(100).alias("top_100_symbols")
    ).sort("date")
    return df_top100.with_columns(
        (pl.col("top_100_market_cap").list.sum() /
         pl.col("top_100_market_cap").list.len()).alias("index_value")
    )

def detect_composition_changes(df):
    df_sorted = df.sort("market_cap", descending=True)
    df_top100 = df_sorted.group_by("date").agg(
        pl.col("symbol").head(100).alias("top_100_symbols"),
        pl.col("market_cap").head(100).alias("top_100_market_cap")
    ).sort("date")
    df_top100 = df_top100.with_columns(
        pl.col("top_100_symbols").shift(1).alias("prev_top_100_symbols")
    ).with_columns([
        pl.struct(["top_100_symbols", "prev_top_100_symbols"]).map_elements(
            lambda s: set(s["top_100_symbols"]) - set(s["prev_top_100_symbols"]
                                                      ) if s["prev_top_100_symbols"] else set()
        ).alias("additions"),
        pl.struct(["top_100_symbols", "prev_top_100_symbols"]).map_elements(
            lambda s: set(s["prev_top_100_symbols"]) -
            set(s["top_100_symbols"]) if s["prev_top_100_symbols"] else set()
        ).alias("removals")
    ])
    return df_top100

def get_date_range(period, end_date):
    if period == "WTD":
        return end_date - timedelta(days=end_date.weekday()), end_date
    elif period == "MTD":
        return end_date.replace(day=1), end_date
    elif period == "YTD":
        return end_date.replace(month=1, day=1), end_date 
    return INCEPTION_DATE, end_date

def fetch_data(con, start_date, end_date, symbols_filter=None):
    # Build symbol filter if symbols are specified.
    symbol_clause = ""
    if symbols_filter:
        # Format each symbol with quotes and comma‐separated
        symbols_str = ", ".join([f"'{s}'" for s in symbols_filter])
        symbol_clause = f" AND symbol IN ({symbols_str})"
    query = f"""
            SELECT * FROM market_cap_data
            WHERE date BETWEEN '{start_date}' AND '{end_date}' {symbol_clause}
            ORDER BY date, market_cap DESC
        """
    return pl.DataFrame(con.execute(query).fetchall(), schema=["symbol", "date", "market_cap"])

# === App Layout ===
st.set_page_config(page_title="Index Insights", page_icon="📈", layout="wide")
st.title("📊 Index Insights")
symbols, start_date, end_date = render_sidebar()

# Move DB connection and max_date definition before filters
max_date = con.execute("SELECT MAX(date) FROM market_cap_data").fetchone()[0]
INCEPTION_DATE = con.execute("SELECT min(date) FROM market_cap_data").fetchone()[0]

with st.sidebar.form('index_insights_form'):
    selected_date = st.date_input("Select a date:", max_value=max_date)
    period = st.radio(
        "Period", ["WTD", "MTD", "YTD", "ITD", "Custom"], horizontal=True, index=3)
    if period == "Custom":
        custom_start_date = st.date_input(
            "Custom Start Date", value=max_date - timedelta(days=30), max_value=max_date)
        custom_end_date = st.date_input(
            "Custom End Date", value=max_date, min_value=custom_start_date, max_value=max_date)
    analyze_button = st.form_submit_button('Analyze')

selected_date_ts = pd.to_datetime(selected_date)

if period == "Custom":
    start_date, end_date = pd.to_datetime(
        custom_start_date), pd.to_datetime(custom_end_date)
else:
    start_date, end_date = get_date_range(period, selected_date_ts)

data = fetch_data(con, start_date, end_date, symbols_filter=symbols)

# === Calculations ===

df_index = compute_equal_weighted_index(data).sort("date")
df_composition_changes = detect_composition_changes(data)
df_index_pd = df_index.to_pandas().sort_values("date")
df_index_pd["daily_pct_change"] = df_index_pd["index_value"].pct_change()

month_mask = (df_index_pd["date"] >= (pd.to_datetime(selected_date) - timedelta(days=365*3))) & (df_index_pd["date"] <= pd.to_datetime(selected_date))
df_index_month = df_index_pd[month_mask]

cumulative_return = (df_index_month["index_value"].iloc[-1] / df_index_month["index_value"].iloc[0] - 1) * 100 if not df_index_month.empty else 0.0
volatility = df_index_pd["daily_pct_change"].std() * (252 ** 0.5) * 100
sharpe_ratio = cumulative_return / volatility if volatility else 0
max_drawdown = (df_index_pd["index_value"].cummax() - df_index_pd["index_value"]).max()

days_held = (df_index_pd["date"].iloc[-1] - df_index_pd["date"].iloc[0]).days
cagr = ((df_index_pd["index_value"].iloc[-1] / df_index_pd["index_value"].iloc[0]) ** (365/days_held) - 1) * 100 if days_held else 0

df_index_pd["rolling_volatility_30d"] = df_index_pd["daily_pct_change"].rolling(window=30).std() * (252 ** 0.5) * 100
df_index_pd["drawdown"] = (df_index_pd["index_value"] - df_index_pd["index_value"].cummax()) / df_index_pd["index_value"].cummax() * 100

# --- New Metrics ---
import numpy as np
# Downside volatility for Sortino
index_daily = df_index_pd["daily_pct_change"].dropna()
downside_std = index_daily[index_daily < 0].std() * np.sqrt(252)
sortino_ratio = (df_index_pd["daily_pct_change"].mean() * 252) / downside_std if downside_std != 0 else np.nan
calmar_ratio = (cagr / abs(max_drawdown)) if max_drawdown != 0 else np.nan

# Beta and correlation versus S&P500
sp500_data = yf.download("^GSPC", start=df_index_pd["date"].iloc[0], end=df_index_pd["date"].iloc[-1])
sp500_data["daily_return"] = sp500_data["Close"].pct_change()
sp500_daily = sp500_data["daily_return"].dropna()
merged = pd.merge(
    df_index_pd[["date", "daily_pct_change"]],
    sp500_daily.rename("sp500_return").reset_index().rename(columns={'Date': 'date'}),
    on="date", how="inner"
)
if not merged.empty:
    beta = merged["daily_pct_change"].cov(merged["sp500_return"]) / merged["sp500_return"].var()
    corr_sp500 = merged["daily_pct_change"].corr(merged["sp500_return"])
else:
    beta, corr_sp500 = np.nan, np.nan

# === Summary Metrics Display ===
st.subheader("📊 Summary Metrics")
cols = st.columns(6)
cols[0].metric("Cum. Return (1M)", f"{cumulative_return:.2f}%")
cols[1].metric("Annualized Vol", f"{volatility:.2f}%")
cols[2].metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
cols[3].metric("Max Drawdown", f"{max_drawdown:.2f}")
cols[4].metric("CAGR", f"{cagr:.2f}%")
cols[5].metric("Sortino Ratio", f"{sortino_ratio:.2f}")

cols2 = st.columns(2)
cols2[0].metric("Beta vs S&P500", f"{beta:.2f}")
cols2[1].metric("Corr with S&P500", f"{corr_sp500:.2f}")

# === Visualizations ===

st.subheader("📈 Index Performance")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_index_month["date"],
    y=df_index_month["index_value"],
    mode="lines",
    name="Index Value",
    line=dict(color="cyan")
))
fig.update_layout(
    title="Equal-Weighted Index Performance",
    xaxis_title="Date",
    yaxis_title="Index Value",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# === Rolling Volatility Chart ===
st.subheader("📉 30-Day Rolling Volatility")
vol_fig = go.Figure()
vol_fig.add_trace(go.Scatter(
    x=df_index_pd["date"],
    y=df_index_pd["rolling_volatility_30d"],
    mode="lines",
    name="Rolling Volatility",
    line=dict(color="orange")
))
vol_fig.update_layout(template="plotly_dark",
                      xaxis_title="Date", yaxis_title="Volatility (%)")
st.plotly_chart(vol_fig, use_container_width=True)

# === Drawdown Chart ===
st.subheader("📉 Drawdown Timeline")
dd_fig = go.Figure()
dd_fig.add_trace(go.Scatter(
    x=df_index_pd["date"],
    y=df_index_pd["drawdown"],
    fill='tozeroy',
    name="Drawdown",
    line=dict(color="red")
))
dd_fig.update_layout(template="plotly_dark",
                     xaxis_title="Date", yaxis_title="Drawdown (%)")
st.plotly_chart(dd_fig, use_container_width=True)

# === Index Composition Table ===
st.subheader("📋 Index Composition")
index_composition = data.to_pandas()
index_composition["date"] = index_composition["date"].dt.date
st.dataframe(index_composition, hide_index=True)

# === Composition Changes Table ===
st.subheader("📅 Historical Composition Changes")
df_composition_changes_pd = df_composition_changes.to_pandas()
df_composition_changes_pd["date"] = df_composition_changes_pd["date"].dt.date
df_composition_changes = st.dataframe(df_composition_changes_pd[[
             "date", "top_100_symbols", "additions", "removals"]], hide_index=True, on_select='rerun', selection_mode="single-row")
selected_row = df_composition_changes.selection.rows 

# === Treemap Visualization ===
st.subheader("🧱 Stock Weights Heatmap")
if selected_row:
    selected_row_index = selected_row[0]
    selected_date = df_composition_changes_pd.iloc[selected_row_index]["date"]
else:
    selected_date = df_composition_changes_pd["date"].max()

selected_comp = df_composition_changes_pd[
    df_composition_changes_pd["date"] == selected_date]

selected_comp = selected_comp.dropna()
if not selected_comp.empty:
    df_expanded = selected_comp.explode(
        ["top_100_symbols", "top_100_market_cap"])
    treemap_fig = go.Figure(go.Treemap(
        labels=df_expanded["top_100_symbols"],
        parents=["Index"] * len(df_expanded),
        values=df_expanded["top_100_market_cap"],
        marker=dict(
            colors=df_expanded["top_100_market_cap"], colorscale="RdYlGn"),
        textinfo="label+value",
        hovertemplate="Stock: %{label}<br>Weight: %{value:.2f}%<extra></extra>"
    ))
    treemap_fig.update_layout(
        height=800,
        margin=dict(t=50, b=80),
        annotations=[
            dict(
                text=f"<b>Index Market Cap Composition for {selected_date}</b>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.025,
                xanchor="center", yanchor="top",
                font=dict(size=16, color="white"),
                align="center"
            )
        ]
    )
    st.plotly_chart(treemap_fig, use_container_width=True)
else:
    st.warning("No valid data available for treemap. Please check the selected date.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-footer">Built with 💓 by Chandra Sekhar Mullu</div>', unsafe_allow_html=True)
