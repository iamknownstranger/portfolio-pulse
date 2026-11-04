"""Backtesting workspace for daily stock/ETF strategies and model allocations."""

from dataclasses import asdict, replace
import hashlib
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import get_benchmark_data
from common.backtesting import (BacktestConfig, FREQUENCIES, STRATEGIES, run_backtest, run_manifest)
from common.portfolio import compare_with_benchmark, drawdowns
from common.price_history import clean_prices
from common.portfolio_ui import format_value, growth_chart, load_portfolio
from common.presentation import page_header
from common.research_ui import render_drawdown_episodes, render_signal_research, render_validation
from common.sidebar import portfolio_filters, portfolio_settings

page_header("Backtest", "Turn an allocation or a trading rule into evidence. Compare the outcome, inspect the costs, and test its stability.",
            "Equities / Strategy research")
symbols, start, end, _, benchmark_symbol, benchmark_name = portfolio_filters()
portfolio = load_portfolio(symbols, start, end)
rf = portfolio_settings()["risk_free_rate"]

with st.container(border=True):
    st.markdown("**01  Build a hypothesis  →  02  Run the simulation  →  03  Inspect the evidence**")
    st.caption("Daily adjusted closes · Long-only stocks & ETFs · Fractional model units · Signals execute at the next common close")

candidate = st.session_state.get("backtest_candidate")
valid_candidate = candidate and set(candidate["weights"]) == set(symbols)
sources = ["Current model"] + (["Optimizer target"] if valid_candidate else [])
requested = st.session_state.pop("backtest_requested_source", None)
if requested in sources:
    st.session_state["backtest_source"] = requested
if st.session_state.get("backtest_source") not in sources:
    st.session_state["backtest_source"] = sources[0]
if requested in sources or st.session_state.get("_bt_source") not in sources:
    st.session_state["_bt_source"] = st.session_state["backtest_source"]
source = st.selectbox("Allocation source", sources, key="_bt_source")
st.session_state["backtest_source"] = source
weights = pd.Series(candidate["weights"]).reindex(symbols) if source == "Optimizer target" else portfolio.weights
if source == "Optimizer target":
    st.info(f"Testing the copied {candidate['label']} target, estimated over "
            f"{candidate.get('estimated_start', 'its source start')} – {candidate.get('estimated_end', 'its source end')}. "
            "Backtest observations inside that estimation window are in-sample. A later-window split does not remove that overlap.")

with st.expander("Strategy & execution settings", expanded=False):
    defaults = BacktestConfig(**st.session_state.get("backtest_config", {}))
    with st.form("backtest_form", enter_to_submit=False):
        cols = st.columns(3)
        strategy = cols[0].selectbox("Strategy template", STRATEGIES, index=STRATEGIES.index(defaults.strategy), key="_bt_strategy")
        frequency = cols[1].selectbox("Rebalance decisions", FREQUENCIES, index=FREQUENCIES.index(defaults.frequency), key="_bt_frequency")
        capital = cols[2].number_input("Starting capital ($)", min_value=1.0, max_value=1e12, value=defaults.initial_capital, step=1000.0)
        cols = st.columns(3)
        lookback = cols[0].number_input("Strategy lookback (observations)", min_value=2, max_value=504, value=defaults.lookback, key="_bt_lookback")
        top_n = cols[1].number_input("Momentum holdings", min_value=1, max_value=len(symbols), value=min(defaults.top_n, len(symbols)), key="_bt_top_n")
        cash_rate = cols[2].number_input("Annual cash yield (%)", min_value=0.0, max_value=100.0, value=defaults.cash_rate * 100, step=.25)
        cols = st.columns(2)
        commission = cols[0].number_input("Commission (bps per side)", min_value=0.0, max_value=1000.0, value=defaults.commission_bps, step=1.0)
        slippage = cols[1].number_input("Slippage (bps per side)", min_value=0.0, max_value=1000.0, value=defaults.slippage_bps, step=1.0)
        st.caption("1 bp = 0.01% of traded notional, charged on buys and sells. Periodic rebalance and buy & hold use "
                   "your source weights. Trend applies those weights only above each asset's trailing moving average, "
                   "rebalanced daily; inactive sleeves stay in cash. Momentum equally weights up to N positive trailing-return "
                   "leaders at the selected frequency. Buy & hold enters once. Trend ignores the frequency; only trend and momentum use the lookback.")
        st.caption("Momentum uses the entire selected ticker universe, including zero-weight tickers, and invests 100% "
                   "among qualifying names. It ignores source allocation weights; when no ticker qualifies, it holds cash.")
        submitted = st.form_submit_button("Run backtest", type="primary", width="stretch")

config = BacktestConfig(strategy, frequency, int(lookback), int(top_n), capital, commission, slippage, cash_rate / 100)
prices = portfolio.prices
benchmark_prices = None
benchmark = get_benchmark_data(benchmark_symbol, start, end)
if not benchmark.empty:
    try:
        benchmark = clean_prices(benchmark.rename("Benchmark").to_frame()).iloc[:, 0]
        aligned, _, _ = compare_with_benchmark(prices, benchmark, weights)
        prices = aligned.prices
        benchmark_prices = benchmark.reindex(prices.index).to_frame("Benchmark")
    except ValueError:
        st.info("Too little overlapping benchmark history. Running the strategy without a benchmark comparison.")
else:
    st.info("Benchmark prices are unavailable. The strategy can still be tested.")

context = {"requested_start": start, "requested_end": end, "source": source,
           "source_label": candidate["label"] if source == "Optimizer target" else "Current model",
           "source_estimation_start": candidate.get("estimated_start") if source == "Optimizer target" else None,
           "source_estimation_end": candidate.get("estimated_end") if source == "Optimizer target" else None}
fingerprint_prices = prices.copy()
if benchmark_prices is not None:
    fingerprint_prices["__benchmark__"] = benchmark_prices.iloc[:, 0]
context_id = hashlib.sha256((fingerprint_prices.to_csv() + json.dumps(
    {"weights": weights.to_dict(), "benchmark": benchmark_symbol, "rf": rf, **context}, sort_keys=True)).encode()).hexdigest()
active = st.session_state.get("backtest_active")
if submitted or active is None or active["context_id"] != context_id:
    try:
        with st.spinner("Simulating holdings, cash and execution costs…"):
            result = run_backtest(prices, weights, config)
            gross = run_backtest(prices, weights, replace(config, commission_bps=0, slippage_bps=0))
            reference = (run_backtest(benchmark_prices, config=replace(config, strategy="Buy & hold"))
                         if benchmark_prices is not None else None)
        manifest = run_manifest(result, fingerprint_prices, weights, benchmark_symbol if reference else None, rf, context)
        active = {"context_id": context_id, "result": result, "gross": gross, "reference": reference,
                  "manifest": manifest, "prices": fingerprint_prices.copy(), "benchmark_name": benchmark_name}
        st.session_state["backtest_active"] = active
        st.session_state["backtest_config"] = asdict(config)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
result, gross, reference = active["result"], active["gross"], active["reference"]
manifest = active["manifest"]
stats = result.stats(rf)

st.subheader(f"{result.config.strategy} · run {manifest['run_id']}")
st.caption(f"{result.equity.index[0]:%d %b %Y} – {result.equity.index[-1]:%d %b %Y} · "
           f"{len(result.returns):,} return observations · {context['source_label']} · "
           f"{result.config.commission_bps + result.config.slippage_bps:g} bps per side")
with st.container(key="headline_metrics"):
    cols = st.columns(4)
    for col, label, key, pct in zip(cols, ["Net total return", "CAGR", "Sharpe ratio", "Max drawdown"],
                                  ["total_return", "cagr", "sharpe", "max_drawdown"], [True, True, False, True]):
        col.metric(label, format_value(stats[key], pct))
if result.config.strategy in {"Trend filter", "Momentum rotation"} and not (result.allocations.drop(columns="Cash").sum(axis=1) > 1e-10).any():
    st.warning("No invested positions: the signal may still be warming up or no asset met its entry rule. Extend the dates or adjust the lookback.")
if len(result.returns) < 252:
    st.caption("Less than one year of observations. Annualized metrics can be unstable over short windows.")
if len(prices) < len(portfolio.prices):
    st.info(f"Benchmark alignment excluded {len(portfolio.prices) - len(prices)} additional portfolio price dates. "
            "Signals and returns use the remaining common calendar, without forward-filling.")

outcome, validation, signals, ledger, saved = st.tabs(["Results", "Validation", "Signal research", "Trade ledger", "Saved runs & exports"])
with outcome:
    curves = {"Strategy · net": result.equity / result.config.initial_capital,
              "Strategy · zero costs": gross.equity / result.config.initial_capital}
    if reference is not None:
        curves[f"{active['benchmark_name']} · buy & hold"] = reference.equity / result.config.initial_capital
    growth_chart(curves, title=f"Growth of ${result.config.initial_capital:,.0f}", initial_value=result.config.initial_capital)
    cols = st.columns(4)
    cols[0].metric("Final model value", format_value(stats["final_value"], dollars=True))
    cols[1].metric("Commissions + slippage", format_value(stats["total_cost"], dollars=True))
    cols[2].metric("Asset executions", f"{stats['executions']:,}", help="One buy or sell in one asset, not a closed round-trip trade.")
    cols[3].metric("Average invested allocation", format_value(stats["average_exposure"], True),
                   help="Arithmetic mean of invested allocations at observed closes, including initial cash. Not weighted by elapsed calendar time.")
    st.caption(f"Zero-cost versus net terminal-value difference: {format_value(gross.equity.iloc[-1] - result.equity.iloc[-1], dollars=True)}. "
               "This includes the compounding effect of costs; it differs from the cash fees paid.")
    if reference is not None:
        difference = (result.equity.iloc[-1] - reference.equity.iloc[-1]) / result.config.initial_capital * 100
        st.caption(f"Return difference versus benchmark: {difference:+.2f} percentage points. Benchmark decides at the first "
                   "common close and enters at the second, independently of strategy warm-up. It pays the same per-side "
                   "entry cost rates and holds. Index benchmarks exclude reinvested dividends.")
    fig = px.area(x=result.equity.index, y=drawdowns(result.equity / result.equity.iloc[0]),
                  labels={"x": "Date", "y": "Drawdown"}, color_discrete_sequence=["#F38F9D"])
    fig.update_layout(yaxis_tickformat=".1%", height=280, title="Losses from the running peak")
    st.plotly_chart(fig, width="stretch")
    fig = px.area(result.allocations, title="Where the capital was allocated")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Allocation", xaxis_title=None, height=300, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    render_drawdown_episodes(result.equity, "bt")
with validation:
    render_validation(result.equity, rf, "bt")
    if source == "Optimizer target":
        st.info("The recipe records the optimizer's source window. Any overlap with it remains in-sample; "
                "a chronological return split does not remove that overlap. Rebalancing's Forward validation re-estimates each fold on earlier data.")
with signals:
    render_signal_research(prices, "bt_signal")
with ledger:
    st.subheader("Every execution, with its decision timestamp")
    st.caption("Signed units are positive for buys and negative for sells. Reference prices are adjusted closes. "
               "Slippage is booked as a separate cash cost. Units are synthetic, not actual broker share counts. "
               "Open positions remain marked at the end; there is no automatic final sale.")
    if result.ledger.empty:
        st.info("No asset executions for these rules and dates.")
    else:
        st.dataframe(result.ledger, hide_index=True, width="stretch")
    st.download_button("Download execution ledger", result.ledger.to_csv(index=False).encode(), "backtest_executions.csv", "text/csv", on_click="ignore")
    st.subheader("Final model allocations")
    st.dataframe(result.allocations.iloc[-1].rename("Weight").to_frame().style.format("{:.2%}"), width="stretch")
with saved:
    st.caption("Save up to five runs for this session. Export the recipe and adjusted prices to keep the inputs; market providers can revise historical prices.")
    run_name = st.text_input("Run name", value=f"{result.config.strategy} · {manifest['run_id'][:6]}", max_chars=80)
    if st.button("Save run to session", icon=":material/bookmark:"):
        collection = dict(st.session_state.get("backtest_saved", {}))
        if manifest["run_id"] not in collection and len(collection) >= 5:
            st.warning("Five runs are already saved. Clear the saved comparison to start a new set.")
        else:
            collection[manifest["run_id"]] = {"name": run_name.strip() or result.config.strategy,
                "equity": result.equity.copy(), "stats": stats, "manifest": manifest,
                "ledger": result.ledger.copy(), "prices": active["prices"].copy(),
                "equity_export": pd.concat([result.equity, gross.equity.rename("Zero costs"),
                                             result.costs], axis=1).copy()}
            st.session_state["backtest_saved"] = collection
            st.success("Run saved for this session.")
    collection = st.session_state.get("backtest_saved", {})
    if collection:
        rows, curves = [], {}
        for run_id, run in collection.items():
            rows.append({"Run": run["name"], "ID": run_id, "Start": run["equity"].index[0].date(),
                         "End": run["equity"].index[-1].date(), "Return": run["stats"]["total_return"],
                         "Sharpe": run["stats"]["sharpe"], "Max drawdown": run["stats"]["max_drawdown"]})
            curves[f"{run['name']} · {run_id[:6]}"] = run["equity"] / run["equity"].iloc[0]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Return": st.column_config.NumberColumn(format="percent"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f")})
        growth_chart(curves, "Saved runs · each starts at 100", initial_value=100)
        st.caption("Compare dates, universes, benchmark choices and assumptions before comparing outcomes. "
                   "Different windows are shown on their original dates and are not ranked.")
        st.download_button("Download saved run recipes", json.dumps([run["manifest"] for run in collection.values()], indent=2),
                           "saved_backtest_recipes.json", "application/json", on_click="ignore")
        export_id = st.selectbox("Saved run to export", list(collection),
            format_func=lambda run_id: f"{collection[run_id]['name']} · {run_id}", key="_bt_export_saved")
        export_run = collection[export_id]
        cols = st.columns(3)
        cols[0].download_button("Download saved input prices", export_run["prices"].to_csv(index_label="Date").encode(),
            f"backtest_{export_id}_prices.csv", "text/csv", on_click="ignore")
        cols[1].download_button("Download saved equity & costs", export_run["equity_export"].to_csv(index_label="Date").encode(),
            f"backtest_{export_id}_equity.csv", "text/csv", on_click="ignore")
        cols[2].download_button("Download saved executions", export_run["ledger"].to_csv(index=False).encode(),
            f"backtest_{export_id}_executions.csv", "text/csv", on_click="ignore")
        if st.button("Clear saved comparison"):
            st.session_state.pop("backtest_saved", None)
            st.rerun()
    cols = st.columns(3)
    cols[0].download_button("Download run recipe", json.dumps(manifest, indent=2), "backtest_recipe.json", "application/json", on_click="ignore")
    cols[1].download_button("Download equity & costs", pd.concat([result.equity, gross.equity.rename("Zero costs"), result.costs], axis=1).to_csv(index_label="Date").encode(),
                            "backtest_equity.csv", "text/csv", on_click="ignore")
    cols[2].download_button("Download run prices", active["prices"].to_csv(index_label="Date").encode(), "backtest_prices.csv", "text/csv", on_click="ignore")

with st.expander("Assumptions & limits"):
    st.write("All runs start in cash. A decision at close t trades at the next available common close t+1, after "
             "existing holdings earn that interval's return. Weekly, monthly and quarterly decisions use the first observed "
             "close of a new period. Trend waits for a full moving-average window; momentum waits for its lookback and "
             "then the next scheduled decision. There is no pre-window warm-up history.")
    st.write("Trading costs are user-selected linear estimates, not measured bid/ask quotes or a calibrated fill model. "
             "The account is self-financing, long-only and unlevered. Cash earns the selected effective annual yield over "
             "elapsed calendar days. No taxes, cash flows, liquidity limits, market impact or intraday fills are modeled. "
             "Open positions are valued at the final adjusted close without a liquidation fee.")
    st.write("Yahoo adjusted closes incorporate splits and dividends. Today's selected universe can have survivorship bias; "
             "this is not point-in-time constituent research. Missing observations use the common complete calendar, "
             "without forward-filling. Return intervals can span multiple trading days. Risk ratios use 252 observations "
             "per year and your portfolio risk-free rate; CAGR uses elapsed calendar time. Historical options quotes, "
             "margin simulation, trained models and live execution are outside this daily equity engine.")
    st.caption("Research only. Historical results and diagnostic correlations do not predict future returns. "
               "Run IDs fingerprint the settings and exported price inputs; saved runs last only in this session.")
