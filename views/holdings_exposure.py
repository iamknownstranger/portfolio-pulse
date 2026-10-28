from common.presentation import page_header
import pandas as pd
import plotly.express as px
import streamlit as st

from common.data import load_companies_csv
from common.portfolio import normalized_weights
from common.portfolio_ui import load_portfolio, model_methodology, model_weights
from common.sidebar import portfolio_filters

page_header('Holdings & Exposure',
            'Set your model weights and see how your portfolio is concentrated.', 'Equities / Holdings')
symbols, start, end, *_ = portfolio_filters()
weights = model_weights(symbols)
st.subheader("Your model allocation")
st.write("Start with equal weights or enter your own percentages. These weights apply to historical analytics "
         "across the portfolio pages. They represent a model allocation, not imported brokerage holdings.")
with st.form("allocation_form"):
    editor = st.data_editor(pd.DataFrame({"Asset": symbols, "Weight (%)": weights.reindex(symbols).values * 100}),
                            disabled=["Asset"], hide_index=True, width="stretch",
                            key=f"allocation_editor_{'_'.join(symbols)}_{st.session_state.get('allocation_revision', 0)}",
                            column_config={"Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0,
                                                                                       step=0.1, format="%.2f")})
    cols = st.columns(2)
    save = cols[0].form_submit_button("Apply allocations", type="primary", width="stretch")
    reset = cols[1].form_submit_button("Reset to equal weights", width="stretch")
if reset:
    st.session_state.pop("portfolio_weights", None)
    # A new widget identity discards unsaved edits when resetting.
    st.session_state["allocation_revision"] = st.session_state.get("allocation_revision", 0) + 1
    st.rerun()
if save:
    try:
        weights = normalized_weights(symbols, pd.Series(editor["Weight (%)"].to_numpy() / 100, index=editor["Asset"]))
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["portfolio_weights"] = weights.to_dict()
        st.success("Model allocations updated across your portfolio dashboards.")
weights = model_weights(symbols)
cols = st.columns(3)
cols[0].metric("Assets selected", len(symbols))
cols[1].metric("Largest allocation", f"{weights.max():.2%}")
cols[2].metric("Effective positions", f"{1 / weights.pow(2).sum():.1f}", help="1 / sum of squared weights; lower values mean greater concentration.")

allocation, performance = st.tabs(["Allocation & geography", "Asset history"])
with allocation:
    names = pd.DataFrame({"Asset": symbols, "Weight": weights.reindex(symbols).values})
    try:
        reference = load_companies_csv().copy()
        reference["Asset"] = reference["Symbol"].str.replace(r"\.([AB])$", r"-\1", regex=True)
        reference = reference.drop_duplicates("Asset").rename(columns={"country": "Country"})
        names = names.merge(reference[["Asset", "Name", "Country"]], how="left", on="Asset")
    except (OSError, ValueError, KeyError):
        names["Name"], names["Country"] = None, None
    names["Country"] = names["Country"].fillna("Unknown / outside reference")
    names["Name"] = names["Name"].fillna(names["Asset"])
    cols = st.columns(2)
    cols[0].plotly_chart(px.pie(names, names="Asset", values="Weight", hole=0.6, title="Model allocation"), width="stretch")
    geo = names.groupby("Country", as_index=False)["Weight"].sum()
    cols[1].plotly_chart(px.bar(geo, x="Weight", y="Country", orientation="h", title="Domicile exposure by weight"), width="stretch")
    st.caption("Country and company names come from a bundled historical reference. Country indicates domicile, "
               "not revenue exposure; unrecognized symbols remain visible as unknown.")
    st.dataframe(names[["Asset", "Name", "Country", "Weight"]].style.format({"Weight": "{:.2%}"}), hide_index=True, width="stretch")
    st.download_button("Download model allocation", names.to_csv(index=False).encode(), "model_allocation.csv", "text/csv", on_click="ignore")
with performance:
    portfolio = load_portfolio(symbols, start, end)
    normalized = portfolio.prices / portfolio.prices.iloc[0] * 100
    st.plotly_chart(px.line(normalized, title="Asset growth from 100", labels={"value": "Growth of 100", "index": "Date"}), width="stretch")
model_methodology()
