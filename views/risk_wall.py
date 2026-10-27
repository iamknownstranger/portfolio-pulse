import numpy as np
import plotly.express as px
import streamlit as st

from common.portfolio import drawdowns
from common.portfolio_ui import format_value, headline_metrics, load_portfolio, model_methodology
from common.sidebar import portfolio_filters

st.title("Risk Wall")
st.caption("Understand the losses behind the return, and what another shock could mean.")
symbols, start, end, *_ = portfolio_filters()
portfolio = load_portfolio(symbols, start, end)
stats = headline_metrics(portfolio)

st.subheader("One-day loss estimates")
st.caption("All three estimates below use a 95% confidence level and a one-observation horizon. Historical losses can exceed them.")
cols = st.columns(3)
for col, label, key, help_text in [
    (cols[0], "Historical VaR · 95%", "var_95", "5th percentile of observed returns, expressed as a non-negative loss."),
    (cols[1], "Expected shortfall · 95%", "cvar_95", "Average loss in the worst 5% of observations."),
    (cols[2], "EWMA VaR · 95%", "ewma_var_95", "Normal approximation with daily EWMA second moment, decay 0.94; no annualization."),
]:
    col.metric(label, format_value(stats[key], True), help=help_text)

losses, volatility, correlation = st.tabs(["Drawdowns & stress", "Volatility", "Correlations"])
with losses:
    dd = drawdowns(portfolio.wealth) * 100
    fig = px.area(x=dd.index, y=dd, labels={"x": "Date", "y": "Drawdown (%)"},
                  title="Distance from the running peak", color_discrete_sequence=["#F58C93"])
    st.plotly_chart(fig, width="stretch")
    st.subheader("Stress your model value")
    value = st.number_input("Illustrative portfolio value ($)", min_value=1.0, value=10000.0, step=1000.0)
    cols = st.columns(3)
    peak_ratio = portfolio.wealth.max() / portfolio.wealth.iloc[-1]
    for col, shock in zip(cols, [-0.05, -0.10, -0.20]):
        with col.container(border=True):
            st.metric(f"{shock:.0%} shock", format_value(value * (1 + shock), dollars=True),
                      format_value(value * shock, dollars=True))
            st.caption(f"Resulting drawdown from historical peak: {(1 + shock) / peak_ratio - 1:.2%}")
    st.caption("Illustrative simultaneous shocks to portfolio value. These are scenarios, not probability estimates.")
with volatility:
    window = st.select_slider("Rolling window (observations)", [10, 21, 63], value=21)
    rolling = portfolio.returns.rolling(window).std() * np.sqrt(252) * 100
    if rolling.dropna().empty:
        st.info(f"At least {window} return observations are needed for this chart.")
    else:
        fig = px.line(x=rolling.index, y=rolling, labels={"x": "Date", "y": "Annual volatility (%)"})
        st.plotly_chart(fig, width="stretch")
    fig = px.histogram(x=portfolio.returns * 100, nbins=45, labels={"x": "Portfolio return (%)"},
                       title="Observed return distribution")
    st.plotly_chart(fig, width="stretch")
with correlation:
    st.caption("Correlations are calculated from returns rather than price levels.")
    fig = px.imshow(portfolio.asset_returns.corr(), zmin=-1, zmax=1, text_auto=".2f",
                    color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig, width="stretch")
model_methodology()
