"""App router: asset-class navigation for the multipage app.

Pages are grouped by asset class (GitHub issue #5): equities split into
linear (cash portfolios) and non-linear (options), plus fixed income and
commodities sections.
"""

import streamlit as st

# (script path, title, icon, url_path) per section. Kept as plain data so
# tests can validate the registry without a Streamlit runtime.
PAGE_SECTIONS = {
    "📈 Equities — Linear": [
        ("views/portfolio_pulse.py", "Portfolio Pulse", "💼", "portfolio"),
        ("views/index_insights.py", "Index Insights", "📈", "index-insights"),
        ("views/risk_wall.py", "Risk Wall", "⚠️", "risk-wall"),
        ("views/performance_analytics.py", "Performance Analytics", "📊", "performance"),
        ("views/rebalancing_optimization.py", "Rebalancing & Optimization", "🔄", "rebalancing"),
        ("views/holdings_exposure.py", "Holdings & Exposure", "📋", "holdings"),
    ],
    "🎯 Equities — Non-Linear": [
        ("views/option_chain.py", "Option Chain", "📝", "option-chain"),
    ],
    "🏦 Fixed Income": [
        ("views/rates_macro.py", "Rates & Macro", "🏦", "rates-macro"),
    ],
    "🛢️ Commodities": [
        ("views/commodities.py", "Commodities", "🛢️", "commodities"),
    ],
    "🪙 Digital Assets": [
        ("views/crypto.py", "Crypto Metrics", "🪙", "crypto"),
    ],
    "🧭 Thematic & Flows": [
        ("views/smart_money.py", "Smart Money Radar", "🧭", "smart-money"),
    ],
}

DEFAULT_PAGE = "views/portfolio_pulse.py"


def run():
    """Entry point: configure the app and dispatch to the selected page."""
    st.set_page_config(page_title="Portfolio Pulse", page_icon="💼", layout="wide")
    nav = {}
    for section, entries in PAGE_SECTIONS.items():
        if len(entries) == 1:
            # A one-page category is itself the destination: show one link
            # with the parent's name and icon, without a redundant child row.
            icon, _, title = section.partition(" ")
            path, _, _, url_path = entries[0]
            nav.setdefault("", []).append(
                st.Page(path, title=title, icon=icon, url_path=url_path)
            )
        else:
            nav[section] = [
                st.Page(path, title=title, icon=icon, url_path=url_path,
                        default=(path == DEFAULT_PAGE))
                for path, title, icon, url_path in entries
            ]
    page = st.navigation(nav)
    if page.title in {entry[1] for entry in PAGE_SECTIONS["📈 Equities — Linear"]}:
        # Render shared widgets in the entrypoint so their identity remains
        # stable across pages (page-owned widgets are cleaned up on navigation).
        from common.sidebar import render_sidebar

        render_sidebar()
    page.run()
