import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import get_benchmark_data
from common.portfolio import compare_with_benchmark, performance_stats, relative_stats
from common.portfolio_ui import format_value, growth_chart, headline_metrics, load_portfolio, model_methodology
from common.sidebar import portfolio_settings, portfolio_filters

st.title("Performance Analytics")
st.caption("Find what drove your returns and compare the model on a common benchmark calendar.")
symbols, start, end, period, benchmark_symbol, benchmark_name = portfolio_filters()
portfolio = load_portfolio(symbols, start, end)
headline_metrics(portfolio)
rf = portfolio_settings()["risk_free_rate"]

comparison, contributions, monthly = st.tabs(["Benchmark comparison", "Return attribution", "Monthly returns"])
with comparison:
    benchmark = get_benchmark_data(benchmark_symbol, start, end)
    if benchmark.empty:
        st.info("Benchmark data is unavailable. Return attribution and monthly returns are still available.")
    else:
        try:
            aligned, benchmark_returns, benchmark_wealth = compare_with_benchmark(portfolio.prices, benchmark, portfolio.weights)
        except ValueError:
            st.info("At least three overlapping portfolio and benchmark price dates are needed.")
        else:
            growth_chart({"Portfolio": aligned.wealth, benchmark_name: benchmark_wealth})
            st.caption(f"Comparison window: {aligned.prices.index[0]:%d %b %Y} – {aligned.prices.index[-1]:%d %b %Y}. "
                       "Both series start at the same value on the same date.")
            relative = relative_stats(aligned.returns, benchmark_returns, rf)
            cols = st.columns(4)
            for col, label, key, pct, help_text in [
                (cols[0], "Annual alpha", "alpha", True, "CAPM alpha using your risk-free rate and the aligned observation window."),
                (cols[1], "Beta", "beta", False, "Sensitivity to benchmark returns; unavailable for a constant benchmark."),
                (cols[2], "Tracking error", "tracking_error", True, "Annualized volatility of portfolio minus benchmark returns."),
                (cols[3], "Information ratio", "information_ratio", False, "Annualized active return per unit of tracking error."),
            ]:
                col.metric(label, format_value(relative[key], pct), help=help_text)
            p_stats = aligned.stats(rf)
            b_stats = performance_stats(benchmark_returns, rf, start=aligned.prices.index[0])
            metrics = {"Total return": "total_return", "CAGR": "cagr", "Annual volatility": "volatility",
                       "Max drawdown": "max_drawdown", "Sharpe ratio": "sharpe", "Sortino ratio": "sortino"}
            table = pd.DataFrame({name: {label: format_value(stats[key], "ratio" not in label)
                                        for label, key in metrics.items()}
                                  for name, stats in [("Portfolio", p_stats), (benchmark_name, b_stats)]})
            st.dataframe(table, width="stretch")
with contributions:
    contribution = portfolio.contributions().sort_values() * 100
    table = pd.DataFrame({"Asset": contribution.index, "Contribution (pp)": contribution.values})
    fig = px.bar(table, x="Contribution (pp)", y="Asset", orientation="h", color="Contribution (pp)",
                 color_continuous_scale="RdYlGn", title="Each asset's contribution to total portfolio return")
    st.plotly_chart(fig, width="stretch")
    st.caption("Daily weighted returns are linked through portfolio growth. Contributions sum to total return, "
               "including the effect of daily rebalancing. Values are percentage points.")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download return attribution", table.to_csv(index=False).encode(), "return_attribution.csv",
                       "text/csv", on_click="ignore")
with monthly:
    monthly_returns = (1 + portfolio.returns).resample("ME").prod(min_count=1) - 1
    if len(monthly_returns) < 2:
        st.info("Extend the date range to compare multiple months.")
    else:
        frame = pd.DataFrame({"Year": monthly_returns.index.year, "Month": monthly_returns.index.month,
                              "Return (%)": monthly_returns.values * 100})
        matrix = frame.pivot(index="Year", columns="Month", values="Return (%)").reindex(columns=range(1, 13))
        matrix.columns = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%b")
        fig = px.imshow(matrix, text_auto=".1f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        aspect="auto", labels={"color": "Return (%)"})
        st.plotly_chart(fig, width="stretch")
        st.caption("The first and last month may contain only part of a month. Missing months have no return observations.")
model_methodology()
