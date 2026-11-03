"""Research diagnostics shared by Backtest, Performance Analytics and Risk Wall."""

import numpy as np
import plotly.express as px
import streamlit as st

from common.backtesting import (drawdown_episodes, rolling_windows, signal_deciles,
                                signal_observations, validation_windows)
from common.portfolio_ui import format_value


def render_drawdown_episodes(wealth, key):
    st.subheader("Drawdown episodes")
    episodes = drawdown_episodes(wealth).head(10)
    if episodes.empty:
        st.info("No drawdown episodes in this window.")
        return
    st.dataframe(episodes, hide_index=True, width="stretch", column_config={
        "Peak": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Trough": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Recovery": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Drawdown": st.column_config.NumberColumn(format="percent"),
    })
    st.caption("Ten deepest peak-to-recovery episodes. A blank recovery is still underwater at the last close. "
               "Duration is calendar days from the peak, including unrecovered episodes.")
    st.download_button("Download drawdown episodes", drawdown_episodes(wealth).to_csv(index=False).encode(),
                       "drawdown_episodes.csv", "text/csv", key=f"{key}_episodes", on_click="ignore")


def render_validation(equity, risk_free_rate=0.0, key="validation"):
    st.subheader("Does the result hold up later?")
    st.caption("The same fixed rules continue across the split, with positions and signal history retained. "
               "No model is trained and no parameters are chosen by this check.")
    fraction = st.slider("Initial window (%)", 50, 90, 70, step=5, key=f"{key}_split") / 100
    table = validation_windows(equity, risk_free_rate, fraction)
    if table.empty:
        st.info("At least 21 price observations are needed for a chronological split.")
        return
    st.dataframe(table, width="stretch", column_config={
        "Start": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "End": st.column_config.DateColumn(format="YYYY-MM-DD"),
        "Return": st.column_config.NumberColumn(format="percent"),
        "CAGR": st.column_config.NumberColumn(format="percent"),
        "Max drawdown": st.column_config.NumberColumn(format="percent"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
    })
    st.warning("The later window is a useful holdout only while you keep its results out of your design decisions. "
               "Repeatedly tuning on it makes it part of your research sample. This split is not walk-forward model training.")
    windows = rolling_windows(equity, risk_free_rate)
    if not windows.empty:
        st.subheader("Stability across 63-observation windows")
        fig = px.bar(windows, x="End", y="Return", color="Return", color_continuous_scale="RdYlGn",
                     color_continuous_midpoint=0, hover_data=["Start", "Observations"])
        fig.update_layout(yaxis_tickformat=".1%", xaxis_title=None, height=300, showlegend=False)
        st.plotly_chart(fig, width="stretch", key=f"{key}_windows")
        st.caption("Non-overlapping return windows; the last may be shorter. Drawdown resets at each window's baseline.")
        st.download_button("Download window diagnostics", windows.to_csv(index=False).encode(),
                           "validation_windows.csv", "text/csv", key=f"{key}_download", on_click="ignore")


def render_signal_research(prices, key="signal"):
    st.subheader("Interrogate a signal before you trust it")
    st.caption("Explore one asset with independent research controls; these do not change your backtest strategy settings.")
    cols = st.columns(4)
    asset = cols[0].selectbox("Research asset", list(prices.columns), key=f"{key}_asset")
    kind = cols[1].selectbox("Signal", ["Momentum", "Distance from moving average"], key=f"{key}_kind")
    lookback = cols[2].number_input("Lookback (observations)", min_value=2, max_value=504, value=63, key=f"{key}_lookback")
    horizon = cols[3].selectbox("Forward horizon (observations)", [1, 5, 21], index=1, key=f"{key}_horizon")
    observations = signal_observations(prices, asset, lookback, horizon, kind)
    st.caption("At decision close t, measure the signal using only trailing data. The outcome begins at execution "
               "close t+1 and ends after the selected horizon. These asset returns exclude trading costs.")
    if len(observations) < 20:
        st.info("At least 20 complete signal/outcome pairs are needed. Extend the dates or shorten the lookback/horizon.")
        return
    corr = (observations["Signal"].corr(observations["Forward return"])
            if observations["Signal"].std() > 1e-12 and observations["Forward return"].std() > 1e-12 else np.nan)
    cols = st.columns(3)
    cols[0].metric("Signal / forward-return correlation", format_value(corr))
    cols[1].metric("Signal / outcome pairs", f"{len(observations):,}")
    cols[2].metric("Mean forward return", format_value(observations["Forward return"].mean(), True))
    left, right = st.columns(2)
    with left:
        fig = px.line(observations, y="Signal", title=f"{asset} · signal history")
        fig.update_layout(yaxis_tickformat=".1%", height=300)
        st.plotly_chart(fig, width="stretch", key=f"{key}_history")
        fig = px.scatter(observations, x="Signal", y="Forward return", opacity=.45,
                         title="Signal versus later returns")
        fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%", height=300)
        st.plotly_chart(fig, width="stretch", key=f"{key}_scatter")
    with right:
        fig = px.histogram(observations, x="Signal", nbins=35, title="Signal distribution")
        fig.update_layout(xaxis_tickformat=".1%", height=300)
        st.plotly_chart(fig, width="stretch", key=f"{key}_distribution")
        deciles = signal_deciles(observations)
        if deciles.empty:
            st.info("A constant signal cannot be separated into deciles.")
        else:
            fig = px.bar(deciles.reset_index(), x="Bucket", y="Mean forward return", hover_data=["Observations"],
                         title="Forward returns by signal decile")
            fig.update_layout(yaxis_tickformat=".1%", height=300, xaxis_dtick=1)
            st.plotly_chart(fig, width="stretch", key=f"{key}_deciles")
            st.caption("Tied signal values may produce fewer than ten buckets. Hover for each bucket's sample count.")
    st.caption("Descriptive research on the selected historical universe. Adjacent forward outcomes overlap, "
               "so sample counts are not independent trials. Correlation and deciles do not establish significance or a tradable edge.")
    st.download_button("Download signal observations", observations.to_csv(index_label="Decision date").encode(),
                       "signal_research.csv", "text/csv", key=f"{key}_download", on_click="ignore")
