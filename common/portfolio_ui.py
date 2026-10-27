"""Small presentation helpers for the portfolio workspace."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import data
from common.portfolio import analyze_portfolio, normalized_weights
from common.sidebar import portfolio_settings

ACCENT = "#35D6B4"


def format_value(value, percent=False, dollars=False):
    if not np.isfinite(value):
        return "—"
    if dollars:
        return f"${value:,.2f}"
    return f"{value:.2%}" if percent else f"{value:.2f}"


def model_weights(symbols):
    saved = st.session_state.get("portfolio_weights")
    return normalized_weights(symbols, saved if saved is not None and set(saved) == set(symbols) else None)


def load_portfolio(symbols, start, end):
    with st.spinner("Loading adjusted market prices…"):
        prices = data.get_price_data(symbols, start, end, adjusted=True)
    missing = [s for s in symbols if s not in prices.columns or prices[s].dropna().empty]
    if missing:
        st.warning("Price data is unavailable for: " + ", ".join(missing) +
                   ". Check the tickers or try again. Update the selection to continue.")
        st.stop()
    try:
        portfolio = analyze_portfolio(prices.reindex(columns=symbols), model_weights(symbols))
    except ValueError as exc:
        st.info(str(exc))
        st.stop()
    st.caption(
        f"{len(portfolio.weights)} assets · {portfolio.prices.index[0]:%d %b %Y} – "
        f"{portfolio.prices.index[-1]:%d %b %Y} · {len(portfolio.returns):,} return observations · "
        "Split/dividend-adjusted closes"
    )
    if portfolio.excluded_rows:
        st.info(f"Using common price dates for every asset; {portfolio.excluded_rows} incomplete rows were excluded. "
                "Returns span consecutive available common dates, which may be more than one trading day.")
    if (pd.Timestamp(end) - portfolio.prices.index[-1]).days > 5:
        st.warning("The latest available price is more than five calendar days before your selected end date.")
    return portfolio


def headline_metrics(portfolio):
    stats = portfolio.stats(portfolio_settings()["risk_free_rate"])
    cols = st.columns(4)
    for col, label, key, percent, help_text in [
        (cols[0], "Total return", "total_return", True, "Compounded model return over the available period."),
        (cols[1], "Annual volatility", "volatility", True, "Sample return volatility, annualized using 252 observations."),
        (cols[2], "Max drawdown", "max_drawdown", True, "Largest decline from a peak, including the initial investment."),
        (cols[3], "Sharpe ratio", "sharpe", False, "Annualized excess return per unit of risk, using your risk-free rate."),
    ]:
        col.metric(label, format_value(stats[key], percent), help=help_text)
    return stats


def growth_chart(series, title="Growth of $10,000", initial_value=10000):
    fig = go.Figure()
    colors = [ACCENT, "#8FA8DD", "#F3B76C"]
    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=values.index, y=values * initial_value, name=name,
                                 mode="lines", line={"color": colors[i % len(colors)], "width": 2.5}))
    fig.update_layout(title=title, height=410, xaxis_title=None, yaxis_title="Model value ($)",
                      yaxis_tickprefix="$", hovermode="x unified", legend={"orientation": "h", "y": 1.12},
                      margin={"l": 10, "r": 10, "t": 85, "b": 10})
    st.plotly_chart(fig, width="stretch")


def model_methodology():
    with st.expander("How this model works"):
        st.write("The portfolio is a hypothetical, fully invested allocation rebalanced to its model weights "
                 "at each observed close. It uses adjusted Yahoo Finance prices on dates shared by all assets. "
                 "Returns exclude fees, taxes, trading costs and cash flows. Changing an allocation applies it "
                 "to the entire historical window; it does not record a trade.")
        st.write("Volatility and risk-adjusted ratios use 252 return observations per year. Total return covers "
                 "the actual observed period; CAGR uses elapsed calendar time. The S&P 500 and other index "
                 "benchmarks are price indices, so they exclude reinvested dividends even when the portfolio "
                 "includes them. Use a total-return ETF such as SPY in a model when that distinction matters.")
        st.caption("Market data is cached for up to one hour. Allocations and filters persist across pages within this session.")
