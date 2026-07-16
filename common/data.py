"""Shared market-data access helpers.

Centralizes the DuckDB-first / yfinance-fallback fetching logic that was
previously copy-pasted (with bugs) across every page.
"""

import time

import duckdb
import pandas as pd
import streamlit as st
import yfinance as yf

MARKET_DATA_DB = "data/market_data.db"
TOP100_CSV = "data/largest-companies-in-the-usa-by-market-cap.csv"


def _fetch_from_duckdb(tickers, start, end, db_path=MARKET_DATA_DB):
    """Fetch close prices from the local DuckDB database.

    Uses parameterized queries — ticker symbols and dates come from user
    input and must never be interpolated into SQL.
    """
    if not tickers:
        return pd.DataFrame()
    placeholders = ", ".join(["?"] * len(tickers))
    query = f"""
        SELECT date, symbol, close_price FROM market_data
        WHERE symbol IN ({placeholders})
        AND date BETWEEN ? AND ?
        ORDER BY date;
    """
    with duckdb.connect(db_path, read_only=True) as con:
        rows = con.execute(query, [*tickers, str(start), str(end)]).fetchall()
    df = pd.DataFrame(rows, columns=["date", "symbol", "close_price"])
    if df.empty:
        return df
    return df.pivot(index="date", columns="symbol", values="close_price")


def _fetch_from_yfinance(tickers, start, end, max_retries=3, retry_delay=5):
    """Fetch close prices from Yahoo Finance with rate-limit retries."""
    for attempt in range(max_retries):
        try:
            df_yf = yf.download(
                tickers, start=start, end=end,
                auto_adjust=False, multi_level_index=False,
            )["Close"]
            if isinstance(df_yf, pd.Series):
                df_yf = df_yf.to_frame(name=tickers[0] if isinstance(tickers, (list, tuple)) else tickers)
            df_yf.index = pd.to_datetime(df_yf.index).date
            return df_yf.dropna(axis=1, how="all")
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                if attempt < max_retries - 1:
                    st.warning(f"Yahoo Finance rate limited. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    st.error("Too many requests to Yahoo Finance. Please try again later.")
                    return pd.DataFrame()
            else:
                st.error(f"Error fetching data from yfinance: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=86400)
def get_price_data(tickers, start, end):
    """Historical close prices for tickers, DuckDB first, yfinance fallback.

    Returns a DataFrame indexed by date with one column per symbol.
    """
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()
    try:
        df = _fetch_from_duckdb(tickers, start, end)
        if not df.empty:
            return df
    except duckdb.CatalogException:
        # Local database has no market_data table yet — expected, use yfinance.
        pass
    except Exception as e:
        st.warning(f"DuckDB data fetch failed: {e}. Falling back to yfinance.")
    return _fetch_from_yfinance(tickers, start, end)


@st.cache_data(ttl=86400)
def get_top100us_index(start, end):
    """Equal-weighted index of the top 100 US companies by market cap."""
    df_companies = pd.read_csv(TOP100_CSV)
    df_companies["marketcap"] = pd.to_numeric(df_companies["marketcap"], errors="coerce")
    tickers = (
        df_companies.sort_values("marketcap", ascending=False)
        .head(100)["Symbol"]
        .tolist()
    )
    prices = _fetch_from_yfinance(tickers, start, end)
    if prices.empty:
        return pd.Series(dtype=float)
    index_val = prices.mean(axis=1)
    index_val.index = pd.to_datetime(index_val.index)
    return index_val.dropna()


@st.cache_data(ttl=86400)
def get_benchmark_data(symbol, start, end):
    """Historical close series for a benchmark index symbol.

    Supports the synthetic "TOP100US" equal-weighted benchmark as well as
    regular Yahoo Finance index symbols.
    """
    if symbol == "TOP100US":
        return get_top100us_index(start, end)
    prices = _fetch_from_yfinance([symbol], start, end)
    if prices.empty:
        return pd.Series(dtype=float)
    series = prices.iloc[:, 0].dropna()
    series.index = pd.to_datetime(series.index)
    return series


@st.cache_data
def load_companies_csv(path="data/companiesmarketcap.com - Largest American companies by market capitalization.csv"):
    """Market-cap CSV with numeric columns coerced."""
    df = pd.read_csv(path)
    df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
    if "price (INR)" in df.columns:
        df["price (INR)"] = pd.to_numeric(df["price (INR)"], errors="coerce")
    return df
