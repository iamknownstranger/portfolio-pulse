import duckdb
import polars as pl
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


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
    return None, end_date


def fetch_data(con, start_date, end_date):
    if start_date:
        return pl.DataFrame(con.execute(f"""
            SELECT * FROM market_cap_data
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date, market_cap DESC
        """).fetchall(), schema=["symbol", "date", "market_cap"])
    return pl.DataFrame(con.execute("""
        SELECT * FROM market_cap_data
        ORDER BY date, market_cap DESC
    """).fetchall(), schema=["symbol", "date", "market_cap"])

# === App Layout ===


st.set_page_config("Equal-Weighted Index Dashboard", layout="wide")
st.title("📊 Custom Equal-Weighted Index Dashboard")

# Move DB connection and max_date definition before filters
con = duckdb.connect("data/market_cap_data.db")
max_date = con.execute("SELECT MAX(date) FROM market_cap_data").fetchone()[0]

# Track if form has been submitted
if 'index_insights_submitted' not in st.session_state:
    st.session_state['index_insights_submitted'] = False

with st.sidebar.form('index_insights_form'):
    selected_date = st.date_input("Select a date:", max_value=max_date, value=max_date)
    period = st.radio(
        "Period", ["WTD", "MTD", "YTD", "ITD", "Custom"], horizontal=True, index=2)
    if period == "Custom":
        custom_start_date = st.date_input(
            "Custom Start Date", value=max_date - timedelta(days=30), max_value=max_date)
        custom_end_date = st.date_input(
            "Custom End Date", value=max_date, min_value=custom_start_date, max_value=max_date)
    analyze_button = st.form_submit_button('Analyze')
    if analyze_button:
        st.session_state['index_insights_submitted'] = True

selected_date_ts = pd.to_datetime(selected_date)

if period == "Custom":
    start_date, end_date = pd.to_datetime(
        custom_start_date), pd.to_datetime(custom_end_date)
else:
    start_date, end_date = get_date_range(period, selected_date_ts)
data = fetch_data(con, start_date, end_date)

# === Calculations ===
# Only run analytics if the form was submitted or it's the first load
if st.session_state['index_insights_submitted'] or not st.session_state.get('has_loaded_once', False):
    st.session_state['has_loaded_once'] = True
    df_index = compute_equal_weighted_index(data).sort("date")
    df_composition_changes = detect_composition_changes(data)

    df_index_pd = df_index.to_pandas().sort_values("date")
    df_index_pd["daily_pct_change"] = df_index_pd["index_value"].pct_change()

    month_mask = (df_index_pd["date"] >= (
        selected_date_ts - timedelta(days=365*3))) & (df_index_pd["date"] <= selected_date_ts)
    df_index_month = df_index_pd[month_mask]

    cumulative_return = (
        (df_index_month["index_value"].iloc[-1] /
         df_index_month["index_value"].iloc[0] - 1) * 100
        if not df_index_month.empty else 0.0
    )
    volatility = df_index_pd["daily_pct_change"].std() * (252 ** 0.5) * 100
    sharpe_ratio = cumulative_return / volatility if volatility else 0
    max_drawdown = (df_index_pd["index_value"].cummax() -
                    df_index_pd["index_value"]).max()

    # CAGR
    days_held = (df_index_pd["date"].iloc[-1] - df_index_pd["date"].iloc[0]).days
    cagr = ((df_index_pd["index_value"].iloc[-1] / df_index_pd["index_value"].iloc[0])
            ** (365.0 / days_held) - 1) * 100 if days_held else 0

    # Rolling Volatility
    df_index_pd["rolling_volatility_30d"] = df_index_pd["daily_pct_change"].rolling(
        window=30).std() * (252 ** 0.5) * 100

    # Drawdown
    df_index_pd["cum_max"] = df_index_pd["index_value"].cummax()
    df_index_pd["drawdown"] = (df_index_pd["index_value"] -
                               df_index_pd["cum_max"]) / df_index_pd["cum_max"] * 100

    # Turnover
    df_composition_changes_pd = df_composition_changes.to_pandas()
    df_composition_changes_pd["turnover_count"] = df_composition_changes_pd["additions"].apply(
        lambda x: len(x) if isinstance(x, set) else 0)
    avg_turnover = df_composition_changes_pd["turnover_count"].mean()

    # Benchmark: S&P 500
    sp500 = yf.download(
        "^GSPC", start=df_index_pd["date"].iloc[0], end=df_index_pd["date"].iloc[-1])
    sp500["Return"] = sp500["Close"].pct_change()
    sp500["Cumulative Return"] = (1 + sp500["Return"]).cumprod()

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

    # === Summary Metrics ===
    st.subheader("📊 Summary Metrics")
    col0, col1, col2, col3 = st.columns(4)
    col1.metric("Cumulative Return (1M)", f"{cumulative_return:.2f}%")
    col2.metric("Annualized Volatility", f"{volatility:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    col7, col4, col5, col6 = st.columns(4)
    col4.metric("Max Drawdown", f"{max_drawdown/1e9:.2f} bn")
    col5.metric("CAGR", f"{cagr:.2f}%")

    col6.metric("Avg Monthly Turnover", f"{avg_turnover:.0f} stocks")

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
    df_composition_changes_pd["date"] = df_composition_changes_pd["date"].dt.date
    st.dataframe(df_composition_changes_pd[[
                 "date", "top_100_symbols", "additions", "removals"]], hide_index=True)

    # === Treemap Visualization ===
    st.subheader("🧱 Stock Weights Heatmap")
    selected_comp = df_composition_changes_pd[pd.to_datetime(
        df_composition_changes_pd["date"]) == selected_date_ts]
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
        st.warning(
            "No valid data available for treemap. Please check the selected date.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <style>
      /* Ensure the sidebar uses full viewport height */
      .css-1d391kg, .sidebar .sidebar-content { 
          display: flex; 
          flex-direction: column; 
          height: 100vh; 
      }
      /* Let the main content grow, forcing the footer to the bottom */
      .css-1d391kg > div:nth-child(2), .sidebar .sidebar-content > div:first-child { 
          flex: 1;
      }
      .sidebar-footer {
          text-align: center;
          padding: 10px;
          font-size: 0.9rem;
          color: #888;
      }
    </style>
""", unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-footer">Built with 💓 by Chandra Sekhar Mullu</div>', unsafe_allow_html=True)
