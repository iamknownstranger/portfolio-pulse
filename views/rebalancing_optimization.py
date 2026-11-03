from common.presentation import page_header
import math
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.optimization import optimize_allocation
from common.backtesting import BacktestConfig, walk_forward_allocation
from common.portfolio import rebalance_plan
from common.portfolio_ui import format_value, growth_chart, load_portfolio, model_methodology
from common.sidebar import portfolio_settings, portfolio_filters

page_header('Rebalancing & Optimization',
            'Turn your allocation into a constrained target and an illustrative dollar plan.', 'Equities / Allocation')
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

trades, estimates, forward = st.tabs(["Allocation & trade plan", "Risk / return estimates", "Forward validation"])
with trades:
    fig = go.Figure()
    fig.add_bar(x=plan.index, y=plan["Current weight"] * 100, name="Current model", marker_color="#8FA8F5")
    fig.add_bar(x=plan.index, y=plan["Target weight"] * 100, name="Target", marker_color="#35D6B4")
    fig.update_layout(barmode="group", yaxis_title="Allocation (%)", legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(plan.style.format({"Current weight": "{:.2%}", "Target weight": "{:.2%}",
                                    "Change": "{:+.2%}", "Trade value ($)": "${:+,.2f}"}), width="stretch")
    st.download_button("Download rebalance plan", plan.to_csv().encode(), "rebalance_plan.csv", "text/csv", type="primary", on_click="ignore")
    st.caption("Trade values are illustrative changes from your fixed model weights, not from live brokerage "
               "positions or a historical buy-and-hold allocation. Fractional dollar trades exclude costs, taxes "
               "and cash flows. The plan does not place orders.")
    cost_bps = st.number_input("Estimated trading cost (bps per side)", min_value=0.0, max_value=1000.0,
                               value=6.0, step=1.0, key="rebalance_cost_bps")
    cost = plan["Trade value ($)"].abs().sum() * cost_bps / 10000
    st.metric("Estimated rebalance cost", format_value(cost, dollars=True),
              help="Sum of absolute buy and sell notional × cost in basis points / 10,000.")
    st.caption("This estimate excludes costs from target sizing. Backtest sizes positions after funding costs.")
    if st.button("Backtest this target", icon=":material/science:"):
        st.session_state["backtest_candidate"] = {"weights": result.weights.to_dict(), "label": result.method,
            "estimated_start": portfolio.prices.index[0].date().isoformat(),
            "estimated_end": portfolio.prices.index[-1].date().isoformat()}
        st.session_state["backtest_requested_source"] = "Optimizer target"
        st.switch_page("views/backtest.py")
    st.caption("The target uses the full selected estimation window. Testing it over that same history is in-sample; "
               "inspect costs and stability before treating an attractive allocation as evidence.")
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
with forward:
    st.subheader("Re-estimate on the past. Evaluate on the next window.")
    st.caption("Each fold estimates fresh weights using only its trailing training prices. That allocation executes "
               "at the following close, then stays fixed until the next scheduled rebalance. The simulation starts "
               "after the initial training window, with cash and trading costs funded inside the account.")
    cols = st.columns(2)
    training = cols[0].selectbox("Training window (return observations)", [63, 126, 252, 504], index=1, key="wf_train")
    testing = cols[1].selectbox("Forward window (return observations)", [21, 63, 126], index=1, key="wf_test")
    signature = hashlib.sha256((portfolio.prices.to_csv() + repr((method, cap, rf, value, cost_bps, training, testing))).encode()).hexdigest()
    if st.button("Run walk-forward validation", type="primary"):
        try:
            with st.spinner("Estimating and evaluating chronological folds…"):
                validation = walk_forward_allocation(portfolio.prices, method, cap / 100, rf, training, testing,
                    BacktestConfig(initial_capital=value, commission_bps=0, slippage_bps=cost_bps))
            st.session_state["walkforward_active"] = {"signature": signature, "validation": validation}
        except ValueError as exc:
            st.info(str(exc))
    saved_validation = st.session_state.get("walkforward_active")
    if saved_validation and saved_validation["signature"] == signature:
        validation = saved_validation["validation"]
        wf_stats = validation.result.stats(rf)
        cols = st.columns(3)
        cols[0].metric("Forward net return", format_value(wf_stats["total_return"], True))
        cols[1].metric("Forward max drawdown", format_value(wf_stats["max_drawdown"], True))
        cols[2].metric("Forward execution costs", format_value(wf_stats["total_cost"], dollars=True))
        growth_chart({"Walk-forward allocation": validation.result.equity / value,
                      "Equal weight · same schedule": validation.equal_weight.equity / value},
                     title="Forward performance after the first training window", initial_value=value)
        st.dataframe(validation.folds, hide_index=True, width="stretch", column_config={
            **{name: st.column_config.DateColumn(format="YYYY-MM-DD") for name in
               ["Train start", "Decision / train end", "First execution", "Test end"]},
            "Return": st.column_config.NumberColumn(format="percent"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Fold returns use consecutive, non-overlapping return intervals; the last fold may be shorter. "
                   "The equal-weight reference uses the same universe, execution dates, capital and costs. "
                   "Changing settings after reviewing these results introduces selection bias.")
        st.subheader("Allocation chosen at each decision close")
        st.dataframe(validation.weights.style.format("{:.2%}"), width="stretch")
        cols = st.columns(2)
        cols[0].download_button("Download forward folds", validation.folds.to_csv(index=False).encode(),
                                "walk_forward_folds.csv", "text/csv", on_click="ignore")
        cols[1].download_button("Download forward allocations", validation.weights.to_csv(index_label="Decision date").encode(),
                                "walk_forward_allocations.csv", "text/csv", on_click="ignore")
    else:
        st.info("Run validation to evaluate the selected objective, cap and cost assumption on later observations.")
model_methodology()
