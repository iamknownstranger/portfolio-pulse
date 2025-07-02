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

# Updated to ensure calculations use real data from DuckDB or yfinance
def fetch_data(con, start_date, end_date, symbols_filter=None):
    # Build symbol filter if symbols are specified.
    symbol_clause = ""
    if symbols_filter:
        # Format each symbol with quotes and comma-separated
        symbols_str = ", ".join([f"'{s}'" for s in symbols_filter])
        symbol_clause = f" AND symbol IN ({symbols_str})"
    query = f"""
            SELECT * FROM market_cap_data
            WHERE date BETWEEN '{start_date}' AND '{end_date}' {symbol_clause}
            ORDER BY date, market_cap DESC
        """
    return pl.DataFrame(con.execute(query).fetchall(), schema=["symbol", "date", "market_cap"])

# Ensure fallback to yfinance if DuckDB data is unavailable
def fetch_yfinance_data(symbols, start_date, end_date):
    try:
        df_yf = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)["Close"]
        df_yf.index = df_yf.index.date
        return df_yf.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Error fetching data from yfinance: {e}")
        return pd.DataFrame()

# === App Layout ===
st.set_page_config(page_title="Index Insights", page_icon="📈", layout="wide")
st.title("📊 Index Insights")
symbols, start_date, end_date, period, benchmark_symbol, benchmark_name = render_sidebar()


# Move DB connection and max_date definition before filters
max_date = con.execute("SELECT MAX(date) FROM market_cap_data").fetchone()[0]
INCEPTION_DATE = con.execute("SELECT min(date) FROM market_cap_data").fetchone()[0]

data = fetch_data(con, start_date, end_date, symbols_filter=symbols)

# === Calculations ===

df_index = compute_equal_weighted_index(data).sort("date")
df_composition_changes = detect_composition_changes(data)
df_index_pd = df_index.to_pandas().sort_values("date")
df_index_pd["daily_pct_change"] = df_index_pd["index_value"].pct_change()

month_mask = (df_index_pd["date"] >= (pd.to_datetime(end_date) - timedelta(days=365*3))) & (df_index_pd["date"] <= pd.to_datetime(end_date))
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
# --- Ensure date alignment for merging ---
df_index_pd["date"] = pd.to_datetime(df_index_pd["date"]).dt.date
sp500_daily = sp500_daily.reset_index()
sp500_daily["Date"] = pd.to_datetime(sp500_daily["Date"]).dt.date
merged = pd.merge(
    df_index_pd[["date", "daily_pct_change"]],
    sp500_daily.rename(columns={'Date': 'date', 'daily_return': 'daily_return'}),
    on="date", how="inner"
)
if not merged.empty:
    beta = merged["daily_pct_change"].cov(merged["daily_return"]) / merged["daily_return"].var()
    corr_sp500 = merged["daily_pct_change"].corr(merged["daily_return"])
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
    end_date = df_composition_changes_pd.iloc[selected_row_index]["date"]
else:
    end_date = df_composition_changes_pd["date"].max()

selected_comp = df_composition_changes_pd[
    df_composition_changes_pd["date"] == end_date]

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
                text=f"<b>Index Market Cap Composition for {end_date}</b>",
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

# === Portfolio vs Index Composition Comparison ===
st.subheader("📊 Portfolio vs Index Composition Comparison")
# Get latest top 100 symbols from composition changes (using the latest date)
latest_date = df_composition_changes_pd["date"].max()
latest_row = df_composition_changes_pd[df_composition_changes_pd["date"] == latest_date]
if not latest_row.empty:
    top100_symbols = latest_row["top_100_symbols"].iloc[0]
    portfolio_set = set(symbols)
    index_set = set(top100_symbols)
    common_count = len(portfolio_set.intersection(index_set))
    coverage_ratio = (common_count / len(portfolio_set) * 100) if portfolio_set else 0
    st.metric("Portfolio Coverage in Index", f"{coverage_ratio:.2f}%")
    st.write(f"Your portfolio of {len(portfolio_set)} stocks overlaps with {common_count} stocks from the top 100 index.")
else:
    st.warning("No composition data available for portfolio comparison.")

# === Portfolio vs Index Performance & Analytics ===
st.subheader("📈 Portfolio vs Index Performance")

# --- Fetch portfolio price data ---
@st.cache_data(ttl=86400)
def get_portfolio_data(tickers, start, end):
    try:
        df_yf = yf.download(tickers, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
        df_yf.index = pd.to_datetime(df_yf.index)
        return df_yf.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Error fetching portfolio data: {e}")
        return pd.DataFrame()

portfolio_df = get_portfolio_data(symbols, start_date, end_date)

if not portfolio_df.empty:
    # --- Portfolio daily/cumulative returns ---
    portfolio_daily = portfolio_df.pct_change().dropna().mean(axis=1)
    portfolio_cum = (1 + portfolio_daily).cumprod()
    # --- Index daily/cumulative returns ---
    index_daily = df_index_pd.set_index("date")["daily_pct_change"].dropna()
    index_cum = (1 + index_daily).cumprod()
    # --- Align for plotting ---
    perf_df = pd.concat([
        portfolio_cum.rename("Portfolio"),
        index_cum.rename("Top 100 Index")
    ], axis=1, join='inner')
    st.line_chart(perf_df, use_container_width=True)
    # --- Portfolio metrics ---
    portfolio_cum_return = (portfolio_cum.iloc[-1] - 1) * 100 if not portfolio_cum.empty else 0.0
    portfolio_vol = portfolio_daily.std() * np.sqrt(252) * 100
    portfolio_max_dd = ((portfolio_cum.cummax() - portfolio_cum) / portfolio_cum.cummax() * 100).max()
    downside_std_pf = portfolio_daily[portfolio_daily < 0].std() * np.sqrt(252)
    portfolio_sortino = (portfolio_daily.mean() * 252) / downside_std_pf if downside_std_pf != 0 else np.nan
    # CAGR
    if not portfolio_cum.empty:
        days_held_pf = (portfolio_cum.index[-1] - portfolio_cum.index[0]).days
        portfolio_cagr = ((portfolio_cum.iloc[-1]) ** (365/days_held_pf) - 1) * 100 if days_held_pf else 0
    else:
        portfolio_cagr = 0
    # Beta/correlation vs S&P500
    sp500_data = yf.download("^GSPC", start=portfolio_df.index[0], end=portfolio_df.index[-1])
    sp500_data["daily_return"] = sp500_data["Close"].pct_change()
    sp500_daily = sp500_data["daily_return"].dropna()
    # --- Fix: Ensure both DataFrames have 'date' column for merging ---
    pf_df = portfolio_daily.reset_index().rename(columns={portfolio_daily.index.name or 'index': 'date', 0: 'portfolio_daily'})
    sp500_df = sp500_daily.reset_index().rename(columns={sp500_daily.index.name or 'index': 'date', 0: 'daily_return'})
    merged_pf = pd.merge(
        pf_df,
        sp500_df,
        on="date", how="inner"
    )
    if not merged_pf.empty:
        print(merged_pf)
        beta_pf = merged_pf["portfolio_daily"].cov(merged_pf["daily_return"]) / merged_pf["daily_return"].var()
        corr_pf = merged_pf["portfolio_daily"].corr(merged_pf["daily_return"])
    else:
        beta_pf, corr_pf = np.nan, np.nan
    # --- Display metrics side by side ---
    st.subheader("📊 Portfolio vs Index Metrics")
    cols = st.columns(7)
    cols[0].metric("Portfolio Cum. Return", f"{portfolio_cum_return:.2f}%")
    cols[1].metric("Index Cum. Return", f"{cumulative_return:.2f}%")
    cols[2].metric("Portfolio Volatility", f"{portfolio_vol:.2f}%")
    cols[3].metric("Index Volatility", f"{volatility:.2f}%")
    cols[4].metric("Portfolio Sortino", f"{portfolio_sortino:.2f}")
    cols[5].metric("Index Sortino", f"{sortino_ratio:.2f}")
    cols[6].metric("Portfolio Beta vs S&P500", f"{beta_pf:.2f}")
    st.metric("Portfolio Correlation with S&P500", f"{corr_pf:.2f}")
    # --- Risk/Return analytics ---
    st.subheader("Portfolio Correlation Matrix")
    corr_matrix = portfolio_df.corr(method='pearson')
    st.dataframe(corr_matrix)
    import plotly.express as px
    corr_heatmap = px.imshow(corr_matrix, title='Correlation between Portfolio Stocks')
    st.plotly_chart(corr_heatmap, use_container_width=True)
    st.subheader("Portfolio Daily Returns Distribution")
    returns_hist = px.histogram(portfolio_daily, nbins=50, title="Portfolio Daily Returns Distribution")
    st.plotly_chart(returns_hist, use_container_width=True)
else:
    st.warning("No valid portfolio data available for comparison. Please check your stock selection.")

# === Dynamic Benchmark Selection ===
# Helper to fetch benchmark data based on symbol
@st.cache_data(ttl=86400)
def get_benchmark_data(symbol, start, end):
    if symbol == "TOP100US":
        # Top 100 US by market cap (equal-weighted)
        df = pd.read_csv("data/largest-companies-in-the-usa-by-market-cap.csv")
        df = df.sort_values("marketcap", ascending=False).head(100)
        tickers = df["Symbol"].tolist()
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
            st.warning(f"Failed to fetch benchmark data: {e}")
            return pd.Series(dtype=float)

benchmark_series = get_benchmark_data(benchmark_symbol, start_date, end_date)

# --- Portfolio vs Benchmark Performance & Analytics ---
st.subheader("📈 Portfolio vs Benchmark Performance")
portfolio_df = get_portfolio_data(symbols, start_date, end_date)

if not portfolio_df.empty and not benchmark_series.empty:
    # --- Portfolio daily/cumulative returns ---
    portfolio_daily = portfolio_df.pct_change().dropna().mean(axis=1)
    portfolio_cum = (1 + portfolio_daily).cumprod()
    # --- Benchmark daily/cumulative returns ---
    benchmark_daily = benchmark_series.pct_change().dropna()
    benchmark_cum = (1 + benchmark_daily).cumprod()
    benchmark_cum.name = benchmark_name  # Fix: set the name attribute for use in pd.concat
    # --- Align for plotting ---
    perf_df = pd.concat([
        portfolio_cum.rename("Portfolio"),
        benchmark_cum
    ], axis=1, join='inner')
    st.line_chart(perf_df, use_container_width=True)
    # --- Portfolio metrics ---
    portfolio_cum_return = (portfolio_cum.iloc[-1] - 1) * 100 if not portfolio_cum.empty else 0.0
    portfolio_vol = portfolio_daily.std() * np.sqrt(252) * 100
    portfolio_max_dd = ((portfolio_cum.cummax() - portfolio_cum) / portfolio_cum.cummax() * 100).max()
    downside_std_pf = portfolio_daily[portfolio_daily < 0].std() * np.sqrt(252)
    portfolio_sortino = (portfolio_daily.mean() * 252) / downside_std_pf if downside_std_pf != 0 else np.nan
    # CAGR
    if not portfolio_cum.empty:
        days_held_pf = (portfolio_cum.index[-1] - portfolio_cum.index[0]).days
        portfolio_cagr = ((portfolio_cum.iloc[-1]) ** (365/days_held_pf) - 1) * 100 if days_held_pf else 0
    else:
        portfolio_cagr = 0
    # Benchmark metrics
    benchmark_cum_return = (benchmark_cum.iloc[-1] - 1) * 100 if not benchmark_cum.empty else 0.0
    benchmark_vol = benchmark_daily.std() * np.sqrt(252) * 100
    if isinstance(downside_std_bm, (float, int, np.floating, np.integer)):
        notnull = not pd.isnull(downside_std_bm)
    else:
        notnull = pd.notnull(downside_std_bm).all()
    benchmark_sortino = (benchmark_daily.mean() * 252) / downside_std_bm if notnull and downside_std_bm != 0 else np.nan
    # Beta/correlation vs benchmark (dynamic)
    pf_df = portfolio_daily.reset_index().rename(columns={portfolio_daily.index.name or 'index': 'date', 0: 'portfolio_daily'})
    bm_df = benchmark_daily.reset_index().rename(columns={benchmark_daily.index.name or 'index': 'date', 0: 'benchmark_daily'})
    merged_pf = pd.merge(
        pf_df,
        bm_df,
        on="date", how="inner"
    )
    if not merged_pf.empty:
        beta_pf = merged_pf["portfolio_daily"].cov(merged_pf["benchmark_daily"]) / merged_pf["benchmark_daily"].var()
        corr_pf = merged_pf["portfolio_daily"].corr(merged_pf["benchmark_daily"])
    else:
        beta_pf, corr_pf = np.nan, np.nan
    # --- Display metrics side by side ---
    st.subheader("📊 Portfolio vs Benchmark Metrics")
    cols = st.columns(7)
    cols[0].metric("Portfolio Cum. Return", f"{portfolio_cum_return:.2f}%")
    cols[1].metric(f"{benchmark_name} Cum. Return", f"{benchmark_cum_return:.2f}%")
    cols[2].metric("Portfolio Volatility", f"{portfolio_vol:.2f}%")
    cols[3].metric(f"{benchmark_name} Volatility", f"{benchmark_vol:.2f}%")
    cols[4].metric("Portfolio Sortino", f"{portfolio_sortino:.2f}")
    cols[5].metric(f"{benchmark_name} Sortino", f"{benchmark_sortino:.2f}")
    cols[6].metric("Portfolio Beta vs Benchmark", f"{beta_pf:.2f}")
    st.metric("Portfolio Correlation with Benchmark", f"{corr_pf:.2f}")
    # --- Risk/Return analytics ---
    st.subheader("Portfolio Correlation Matrix")
    corr_matrix = portfolio_df.corr(method='pearson')
    st.dataframe(corr_matrix)
    import plotly.express as px
    corr_heatmap = px.imshow(corr_matrix, title='Correlation between Portfolio Stocks')
    st.plotly_chart(corr_heatmap, use_container_width=True)
    st.subheader("Portfolio Daily Returns Distribution")
    returns_hist = px.histogram(portfolio_daily, nbins=50, title="Portfolio Daily Returns Distribution")
    st.plotly_chart(returns_hist, use_container_width=True)
else:
    st.warning("No valid portfolio or benchmark data available for comparison. Please check your stock selection and benchmark.")
