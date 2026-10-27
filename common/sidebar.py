"""Persistent, explicitly applied portfolio filters shared across equity pages."""

from datetime import date, timedelta

import streamlit as st

from common.dates import resolve_start_date
from common.portfolio import DEFAULT_SYMBOLS, normalize_symbols

BENCHMARKS = {
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI",
    "S&P 500 ETF (SPY)": "SPY", "NASDAQ 100 ETF (QQQ)": "QQQ",
    "Top 100 US snapshot (equal weight)": "TOP100US",
    "S&P 500 Technology": "^SP500-45", "S&P 500 Healthcare": "^SP500-35",
    "S&P 500 Financials": "^SP500-40", "S&P 500 Industrials": "^SP500-20",
    "S&P 500 Consumer Discretionary": "^SP500-25",
}


def portfolio_settings():
    if "portfolio_settings" not in st.session_state:
        st.session_state["portfolio_settings"] = {
            "symbols": DEFAULT_SYMBOLS.copy(),
            "start": date.today() - timedelta(days=3 * 365), "end": date.today(),
            "period": "ITD", "benchmark": "S&P 500", "risk_free_rate": 0.0,
        }
    return st.session_state["portfolio_settings"]


def render_sidebar():
    settings = portfolio_settings()
    st.sidebar.title("Your portfolio")
    st.sidebar.caption("Build a model. Explore performance. Understand the risk.")
    # Persistent settings are independent of widget keys: Streamlit removes
    # page-owned widget state when navigating to another page.
    defaults = {
        "_pf_symbols": ", ".join(settings["symbols"]), "_pf_start": settings["start"],
        "_pf_end": settings["end"], "_pf_period": settings["period"],
        "_pf_benchmark": settings["benchmark"], "_pf_rf": settings["risk_free_rate"] * 100,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.sidebar.form("portfolio_form", enter_to_submit=False):
        raw_symbols = st.text_area("Tickers", key="_pf_symbols", height=90,
                                   help="Paste symbols separated by commas or spaces. Up to 30 assets, including ETFs.")
        start = st.date_input("Start date", key="_pf_start", max_value=date.today())
        end = st.date_input("End date", key="_pf_end", max_value=date.today())
        period = st.radio("Period", ["WTD", "MTD", "YTD", "ITD"], key="_pf_period", horizontal=True,
                          help="ITD uses your dates; other periods derive the start from the end date.")
        benchmark = st.selectbox("Benchmark", list(BENCHMARKS), key="_pf_benchmark")
        risk_free = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=30.0,
                                    step=0.25, key="_pf_rf")
        submitted = st.form_submit_button("Update analysis", type="primary", width="stretch")
    if submitted:
        try:
            symbols = normalize_symbols(raw_symbols)
            effective_start = resolve_start_date(period, start, end)
            if effective_start >= end:
                raise ValueError("Choose an end date after the start date. Short periods may need a longer window.")
        except ValueError as exc:
            st.sidebar.error(str(exc))
            st.stop()
        if set(symbols) != set(settings["symbols"]):
            st.session_state.pop("portfolio_weights", None)
        settings = {"symbols": symbols, "start": start, "end": end, "period": period,
                    "benchmark": benchmark, "risk_free_rate": risk_free / 100}
        st.session_state["portfolio_settings"] = settings
    if st.sidebar.button("Refresh market data", width="stretch",
                         help="Clear cached market prices and reload the applied selection."):
        from common.data import get_benchmark_data, get_ohlcv_data, get_price_data, get_top100us_index

        for fetcher in (get_price_data, get_benchmark_data, get_top100us_index, get_ohlcv_data):
            fetcher.clear()
    effective_start = resolve_start_date(settings["period"], settings["start"], settings["end"])
    st.sidebar.caption(f"{len(settings['symbols'])} assets · {effective_start:%d %b %Y} – {settings['end']:%d %b %Y}")
    custom = st.session_state.get("portfolio_weights")
    st.sidebar.caption("Allocation: custom model" if custom else "Allocation: equal weight")
    st.sidebar.page_link("views/holdings_exposure.py", label="Edit model allocations", icon="📋")
    st.sidebar.divider()
    st.sidebar.caption("Research models · Daily rebalancing · No trading costs\n\nBuilt by Chandra Sekhar Mullu")
    return portfolio_filters()


def portfolio_filters():
    """Applied filters for views; shared widgets are rendered by the app router."""
    settings = portfolio_settings()
    start = resolve_start_date(settings["period"], settings["start"], settings["end"])
    return (settings["symbols"], start.isoformat(), settings["end"].isoformat(),
            settings["period"], BENCHMARKS[settings["benchmark"]], settings["benchmark"])
