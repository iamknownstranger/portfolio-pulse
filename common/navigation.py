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
}

DEFAULT_PAGE = "views/portfolio_pulse.py"


def run():
    """Entry point: configure the app and dispatch to the selected page."""
    st.set_page_config(page_title="Portfolio Pulse", page_icon="💼", layout="wide")
    nav = {
        section: [
            st.Page(path, title=title, icon=icon, url_path=url_path,
                    default=(path == DEFAULT_PAGE))
            for path, title, icon, url_path in entries
        ]
        for section, entries in PAGE_SECTIONS.items()
    }
    st.navigation(nav).run()
