import duckdb
import polars as pl
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Index Insights",
    page_icon="📈",
)


# Function to compute the equal-weighted index
def compute_equal_weighted_index(df):
    df_sorted = df.sort(["date", "market_cap"], descending=[True, True])
    df_top100 = df_sorted.group_by("date").agg(
        pl.col("market_cap").head(100).alias("top_100_market_cap"),
        pl.col("symbol").head(100).alias("top_100_symbols")
    ).sort("date")
    # Calculate index as average market cap of top100 (using count in case <100 stocks)
    df_index = df_top100.with_columns(
        (pl.col("top_100_market_cap").list.sum() / pl.col("top_100_market_cap").list.len()).alias("index_value")
    )
    return df_index

# Function to detect composition changes (with the correct symbols)
def detect_composition_changes(df):
    df_sorted = df.sort("market_cap", descending=True)
    df_top100 = df_sorted.group_by("date").agg(
        pl.col("symbol").head(100).alias("top_100_symbols"),
        pl.col("market_cap").head(100).alias("top_100_market_cap")
    ).sort("date")
    # Shift the symbols and market cap list from the previous day
    df_top100 = df_top100.with_columns(
        pl.col("top_100_symbols").shift(1).alias("prev_top_100_symbols"),
        pl.col("top_100_market_cap").shift(1).alias("prev_top_100_market_cap")
    )
    # Flag days where the top100 symbols differ from the previous day
    df_top100 = df_top100.with_columns(
        pl.struct(["top_100_symbols", "prev_top_100_symbols"]).map_elements(
            lambda s: s["top_100_symbols"] != s["prev_top_100_symbols"]
        ).alias("changed")
    )

    return df_top100