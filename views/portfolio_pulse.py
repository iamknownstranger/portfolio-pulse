"""The portfolio workspace: outcomes first, diagnostics and exports on demand."""

from common.presentation import page_header

import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import get_benchmark_data
from common.portfolio import compare_with_benchmark
from common.portfolio_ui import format_value, growth_chart, headline_metrics, load_portfolio, model_methodology
from common.sidebar import portfolio_filters

page_header('Portfolio Pulse',
            'Your allocation, performance and risk. One clear view of the bigger picture.', 'Equities / Overview')
symbols, start, end, period, benchmark_symbol, benchmark_name = portfolio_filters()
portfolio = load_portfolio(symbols, start, end)
stats = headline_metrics(portfolio)

performance, assets, relationships, data = st.tabs(["Overview", "Asset performance", "Diversification", "Data & exports"])
with performance:
    benchmark = get_benchmark_data(benchmark_symbol, start, end)
    if not benchmark.empty:
        try:
            aligned, _, benchmark_wealth = compare_with_benchmark(portfolio.prices, benchmark, portfolio.weights)
            growth_chart({"Portfolio": aligned.wealth, benchmark_name: benchmark_wealth})
            excess = (aligned.wealth.iloc[-1] - benchmark_wealth.iloc[-1]) * 100
            st.caption(f"Compared on common dates: {aligned.prices.index[0]:%d %b %Y} – "
                       f"{aligned.prices.index[-1]:%d %b %Y}. Portfolio return difference: {excess:+.2f} percentage points.")
        except ValueError:
            st.info("The benchmark has too little overlapping history. Showing portfolio performance.")
            growth_chart({"Portfolio": portfolio.wealth})
    else:
        st.info("Benchmark prices are unavailable. Portfolio analytics are still available.")
        growth_chart({"Portfolio": portfolio.wealth})
    cols = st.columns(3)
    cols[0].metric("Model value from $10,000", format_value(10000 * portfolio.wealth.iloc[-1], dollars=True))
    cols[1].metric("Annualized growth (CAGR)", format_value(stats["cagr"], True),
                   help="Geometric annualization of the observed period, not a forecast. Short windows can magnify this value.")
    cols[2].metric("Worst observed day", format_value(stats["worst_day"], True))
    st.subheader("Your next move")
    for i, (col, page, label, description, icon) in enumerate(zip(st.columns(3),
            ["views/risk_wall.py", "views/rebalancing_optimization.py", "views/holdings_exposure.py"],
            ["Understand your risk", "Build a rebalance plan", "Set your allocations"],
            ["Drawdowns, loss estimates and stress scenarios.", "Compare constrained targets and export dollar trades.",
             "Edit the model weights used across your dashboards."],
            [":material/shield:", ":material/balance:", ":material/pie_chart:"])):
        with col.container(border=True, key=f"next_action_{i}"):
            st.page_link(page, label=label, icon=icon)
            st.caption(description)
with assets:
    normalized = portfolio.prices.div(portfolio.prices.iloc[0]).mul(100)
    fig = px.line(normalized, labels={"value": "Growth of 100", "index": "Date", "variable": "Asset"})
    fig.update_layout(title="Every asset, the same starting point", hovermode="x unified", height=410)
    st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame({"Model weight": portfolio.weights,
                          "Total return": portfolio.prices.iloc[-1] / portfolio.prices.iloc[0] - 1,
                          "Annual volatility": portfolio.asset_returns.std() * (252 ** 0.5),
                          "Return contribution": portfolio.contributions()})
    st.dataframe(table.style.format("{:.2%}"), width="stretch")
    st.caption("Contributions include daily rebalancing and add to the portfolio's total return.")
with relationships:
    st.subheader("Do your assets move together?")
    st.caption("Correlation uses returns. Values near +1 indicate similar moves; lower values can offer diversification.")
    fig = px.imshow(portfolio.asset_returns.corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    text_auto=".2f", aspect="auto")
    st.plotly_chart(fig, width="stretch")
    effective = 1 / portfolio.weights.pow(2).sum()
    st.metric("Effective number of positions", f"{effective:.1f}", help="1 / sum of squared model weights; measures allocation concentration, not correlations.")
with data:
    st.subheader("Adjusted closing prices")
    st.dataframe(portfolio.prices, width="stretch")
    export_cols = st.columns(3)
    for col, label, frame, filename in [
        (export_cols[0], "Download adjusted prices", portfolio.prices, "portfolio_prices.csv"),
        (export_cols[1], "Download daily returns", portfolio.asset_returns, "portfolio_returns.csv"),
        (export_cols[2], "Download model performance", portfolio.wealth.to_frame(), "portfolio_growth.csv"),
    ]:
        col.download_button(label, frame.to_csv(index_label="Date").encode(), filename, "text/csv", on_click="ignore")
model_methodology()
