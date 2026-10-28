from common.presentation import page_header
import duckdb
import pandas as pd
import streamlit as st

from common.data import get_benchmark_data, get_price_data
from common.indices import archive_bounds, read_membership
from common.portfolio import analyze_portfolio, compare_with_benchmark
from common.portfolio_ui import growth_chart, headline_metrics, load_portfolio, model_methodology
from common.sidebar import portfolio_filters

page_header('Index Insights',
            'Explore equal-weight performance and the historical universe behind it.', 'Equities / Indices')
symbols, start, end, period, benchmark_symbol, benchmark_name = portfolio_filters()


@st.cache_data(ttl=86400)
def membership(as_of, size):
    return read_membership(as_of, size)


universe = st.radio("Index universe", ["Your selected assets", "Archived market-cap universe"], horizontal=True)
if universe == "Your selected assets":
    portfolio = load_portfolio(symbols, start, end)
    index = analyze_portfolio(portfolio.prices)
    st.caption("The index assigns an equal weight to each selected asset. Compare it with your custom model allocation.")
    headline_metrics(index)
    growth_chart({"Equal-weight index": index.wealth, "Your model": portfolio.wealth})
else:
    try:
        first, last = archive_bounds()
        if first is None:
            raise ValueError("The archive is empty.")
    except (duckdb.Error, OSError, ValueError):
        st.info("The historical market-cap archive is unavailable. Select your own assets to build an index.")
        st.stop()
    st.caption(f"Membership archive: {first:%d %b %Y} – {last:%d %b %Y}. It is a historical reference, not a live ranking.")
    cols = st.columns(2)
    as_of = cols[0].date_input("Membership as of", value=min(max(pd.Timestamp(start).date(), first), last),
                              min_value=first, max_value=last)
    size = cols[1].select_slider("Number of constituents", [10, 25, 50, 100], value=25)
    members, snapshot, removed = membership(as_of.isoformat(), size)
    if members.empty:
        st.info("No membership snapshot is available for that date.")
        st.stop()
    st.caption(f"Using the snapshot from {snapshot:%d %b %Y}. Membership is held fixed across the performance window.")
    if snapshot > pd.Timestamp(start).date():
        st.warning("Membership was selected after the performance start. This introduces look-ahead and survivorship bias.")
    with st.expander("Membership and changes from the previous archived snapshot", expanded=True):
        st.dataframe(members[["Asset", "Change"]], hide_index=True, width="stretch")
        st.caption("Removed: " + (", ".join(removed) if removed else "None"))
        st.download_button("Download membership", members[["Asset", "Change"]].to_csv(index=False).encode(),
                           "index_membership.csv", "text/csv", on_click="ignore")
    index_symbols = members["Asset"].tolist()
    prices = get_price_data(index_symbols, start, end, adjusted=True)
    available = [s for s in index_symbols if s in prices and prices[s].dropna().size >= 3]
    missing = sorted(set(index_symbols) - set(available))
    if missing:
        st.warning("No complete index can be calculated because constituents are unavailable: " + ", ".join(missing) +
                   ". Try a different snapshot, smaller universe, or longer period.")
        st.stop()
    try:
        index = analyze_portfolio(prices.reindex(columns=index_symbols))
    except ValueError as exc:
        st.info(str(exc))
        st.stop()
    st.caption(f"Price window: {index.prices.index[0]:%d %b %Y} – {index.prices.index[-1]:%d %b %Y}. "
               f"{index.excluded_rows} incomplete price rows excluded.")
    headline_metrics(index)
    growth_chart({"Equal-weight index": index.wealth})

benchmark = get_benchmark_data(benchmark_symbol, start, end)
if not benchmark.empty:
    try:
        aligned, _, benchmark_wealth = compare_with_benchmark(index.prices, benchmark)
    except ValueError:
        st.info("Not enough common benchmark dates for a comparison.")
    else:
        growth_chart({"Equal-weight index": aligned.wealth, benchmark_name: benchmark_wealth}, title="Index vs benchmark · growth of $10,000")
        st.caption("Benchmark and index are rebased on the same common dates before returns are calculated.")
else:
    st.info("Benchmark prices are unavailable for comparison.")
model_methodology()
