import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.optimization import optimize_allocation
from common.portfolio import rebalance_plan
from common.portfolio_ui import format_value, load_portfolio, model_methodology
from common.sidebar import portfolio_settings, portfolio_filters

st.title("Rebalancing & Optimization")
st.caption("Turn your model allocation into a constrained target and an exportable dollar plan.")
symbols, start, end, *_ = portfolio_filters()
portfolio = load_portfolio(symbols, start, end)
rf = portfolio_settings()["risk_free_rate"]

cols = st.columns([2, 1, 1])
method = cols[0].selectbox("Target objective", ["Maximum Sharpe", "Minimum volatility", "Equal weight"])
minimum_cap = math.ceil(100 / len(symbols))
cap = cols[1].number_input("Max allocation per asset (%)", min_value=minimum_cap, max_value=100,
                           value=max(40, minimum_cap), step=5,
                           help="Long-only, fully invested. Every target weight respects this cap.")
value = cols[2].number_input("Illustrative portfolio value ($)", min_value=1.0, value=10000.0, step=1000.0)


@st.cache_data(show_spinner=False)
def estimate(prices, method, cap, rf):
    return optimize_allocation(prices, method, cap, rf)


try:
    with st.spinner("Estimating a feasible allocation…"):
        result = estimate(portfolio.prices, method, cap / 100, rf)
except ValueError as exc:
    st.info(str(exc))
    st.stop()
if result.notice:
    st.info(result.notice)
if len(portfolio.prices) < 60:
    st.warning("This estimate uses fewer than 60 common price observations. Short samples can produce unstable targets.")
st.subheader(f"Target allocation · {result.method}")
plan = rebalance_plan(portfolio.weights.to_dict(), result.weights.to_dict(), value)
turnover = plan["Change"].abs().sum() / 2
cols = st.columns(3)
cols[0].metric("One-way turnover", f"{turnover:.2%}", help="Half the sum of absolute weight changes.")
cols[1].metric("Buy value", format_value(plan["Trade value ($)"].clip(lower=0).sum(), dollars=True))
cols[2].metric("Sell value", format_value(-plan["Trade value ($)"].clip(upper=0).sum(), dollars=True))

trades, estimates = st.tabs(["Allocation & trade plan", "Risk / return estimates"])
with trades:
    fig = go.Figure()
    fig.add_bar(x=plan.index, y=plan["Current weight"] * 100, name="Current model")
    fig.add_bar(x=plan.index, y=plan["Target weight"] * 100, name="Target", marker_color="#35D6B4")
    fig.update_layout(barmode="group", yaxis_title="Allocation (%)", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(plan.style.format({"Current weight": "{:.2%}", "Target weight": "{:.2%}",
                                    "Change": "{:+.2%}", "Trade value ($)": "${:+,.2f}"}), width="stretch")
    st.download_button("Download rebalance plan", plan.to_csv().encode(), "rebalance_plan.csv", "text/csv", type="primary", on_click="ignore")
    st.caption("Trade values are illustrative changes from your fixed model weights, not from live brokerage "
               "positions or a historical buy-and-hold allocation. Fractional dollar trades exclude costs, taxes "
               "and cash flows. The plan does not place orders.")
with estimates:
    current_metrics = result.performance(portfolio.weights, rf)
    target_metrics = result.performance(result.weights, rf)
    table = pd.DataFrame({"Current model": current_metrics, "Target": target_metrics},
                          index=["Expected annual return", "Annual volatility", "Sharpe ratio"])
    st.dataframe(table.style.format(subset=pd.IndexSlice[table.index[:2], :], formatter="{:.2%}")
                 .format(subset=pd.IndexSlice[["Sharpe ratio"], :], formatter="{:.2f}"), width="stretch")
    fig = go.Figure()
    if not result.frontier.empty:
        fig.add_scatter(x=result.frontier["Volatility"], y=result.frontier["Return"], mode="lines", name="Efficient frontier")
    for name, (mean, vol, _), color in [("Current model", current_metrics, "#8FA8DD"), ("Target", target_metrics, "#35D6B4")]:
        fig.add_scatter(x=[vol], y=[mean], mode="markers", name=name, marker={"size": 13, "color": color})
    fig.update_layout(xaxis_title="Annual volatility", yaxis_title="Expected annual return",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.caption("Both allocations use the same arithmetic historical return and sample covariance estimates, "
               "the same risk-free rate, and the same observations. The frontier shows feasible portfolios "
               "under your cap. These are in-sample estimates, not out-of-sample performance or forecasts.")
model_methodology()
