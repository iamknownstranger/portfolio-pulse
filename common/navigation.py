"""App router: asset-class navigation for the multipage app.

Pages are grouped by asset class (GitHub issue #5): equities split into
linear (cash portfolios) and non-linear (options), plus fixed income and
commodities sections.
"""

import streamlit as st

from common.presentation import apply_design_system, workspace_brand

# (script path, title, icon, url_path) per section. Kept as plain data so
# tests can validate the registry without a Streamlit runtime.
PAGE_SECTIONS = {
    "📈 Equities — Linear": [
        ("views/portfolio_pulse.py", "Portfolio Pulse", ":material/dashboard:", "portfolio"),
        ("views/index_insights.py", "Index Insights", ":material/query_stats:", "index-insights"),
        ("views/risk_wall.py", "Risk Wall", ":material/shield:", "risk-wall"),
        ("views/performance_analytics.py", "Performance Analytics", ":material/monitoring:", "performance"),
        ("views/rebalancing_optimization.py", "Rebalancing & Optimization", ":material/balance:", "rebalancing"),
        ("views/holdings_exposure.py", "Holdings & Exposure", ":material/pie_chart:", "holdings"),
    ],
    "🎯 Equities — Non-Linear": [
        ("views/option_chain.py", "Option Chain", ":material/functions:", "option-chain"),
    ],
    "🏦 Fixed Income": [
        ("views/rates_macro.py", "Rates & Macro", ":material/account_balance:", "rates-macro"),
    ],
    "🛢️ Commodities": [
        ("views/commodities.py", "Commodities", ":material/oil_barrel:", "commodities"),
    ],
    "🪙 Digital Assets": [
        ("views/crypto.py", "Crypto Metrics", ":material/currency_bitcoin:", "crypto"),
    ],
    "🧭 Thematic & Flows": [
        ("views/smart_money.py", "Smart Money Radar", ":material/radar:", "smart-money"),
    ],
}

DEFAULT_PAGE = "views/portfolio_pulse.py"


def run():
    """Entry point: configure the app and dispatch to the selected page."""
    st.set_page_config(page_title="Portfolio Pulse", page_icon="💼", layout="wide",
                       initial_sidebar_state="auto")
    apply_design_system()
    pages = {}
    for section, entries in PAGE_SECTIONS.items():
        for path, title, icon, url_path in entries:
            if len(entries) == 1:
                _, _, title = section.partition(" ")
            pages[path] = st.Page(path, title=title, icon=icon, url_path=url_path,
                                  default=(path == DEFAULT_PAGE))

    # Native navigation always places ungrouped links before named sections.
    # Register routes normally, then render sidebar links in registry order.
    page = st.navigation(list(pages.values()), position="hidden")
    with st.sidebar:
        workspace_brand()
        with st.container(key="workspace_navigation"):
            for section, entries in PAGE_SECTIONS.items():
                if len(entries) == 1:
                    item = pages[entries[0][0]]
                    st.page_link(item, label=item.title, icon=item.icon, width="stretch")
                else:
                    with st.expander(section.partition(" ")[2], expanded=True, type="compact",
                                     icon=":material/candlestick_chart:"):
                        for path, *_ in entries:
                            item = pages[path]
                            st.page_link(item, label=item.title, icon=item.icon, width="stretch")
        st.divider()
    if page.title in {entry[1] for entry in PAGE_SECTIONS["📈 Equities — Linear"]}:
        # Render shared widgets in the entrypoint so their identity remains
        # stable across pages (page-owned widgets are cleaned up on navigation).
        from common.sidebar import render_sidebar

        render_sidebar()
    page.run()
