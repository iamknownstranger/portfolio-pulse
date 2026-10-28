"""Shared visual language for the research workspace, independent of analytics."""

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

STYLES = Path(__file__).resolve().parent.parent / "assets" / "workspace.css"
COLORS = ["#35D6B4", "#8FA8F5", "#E8BA78", "#BD9EF3", "#F38F9D", "#68C6E2"]


def apply_design_system():
    st.html(STYLES)
    template = go.layout.Template(pio.templates["plotly_dark"])
    template.layout.update(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORS, font={"family": "Inter, system-ui, sans-serif", "size": 12, "color": "#9BAABD"},
        title={"font": {"size": 15, "color": "#E9EFF7"}, "x": 0.02},
        hoverlabel={"bgcolor": "#1A2838", "bordercolor": "#314256", "font_size": 13},
        xaxis={"gridcolor": "rgba(148,163,184,0.06)", "zeroline": False, "showline": False},
        yaxis={"gridcolor": "rgba(148,163,184,0.10)", "zeroline": False, "showline": False},
        legend={"bgcolor": "rgba(0,0,0,0)", "font_size": 12},
        margin={"l": 24, "r": 24, "t": 64, "b": 32},
    )
    pio.templates["portfolio_workspace"] = template
    pio.templates.default = "portfolio_workspace"


def workspace_brand():
    with st.container(key="workspace_brand"):
        st.html('''<div class="pp-brand" data-portfolio-ui="studio-20261028">
          <div class="pp-brand-mark" aria-hidden="true">PP</div>
          <div><strong>Portfolio Pulse</strong><span>INVESTMENT WORKSPACE</span></div>
        </div>''')


def page_header(title, description, section="Equities / Research"):
    with st.container(key="page_header"):
        st.html(f'<div class="pp-eyebrow">{escape(section)}</div>')
        st.title(title)
        st.caption(description)


def analysis_context(portfolio):
    equal_weight = all(abs(w - 1 / len(portfolio.weights)) < 1e-6 for w in portfolio.weights)
    labels = [
        f"{len(portfolio.weights)} assets",
        f"{portfolio.prices.index[0]:%d %b %Y} — {portfolio.prices.index[-1]:%d %b %Y}",
        f"{len(portfolio.returns):,} return observations",
        "Equal weight" if equal_weight else "Custom allocation",
        "Adjusted prices",
    ]
    st.html('<div class="pp-context">' + ''.join(f'<span>{escape(label)}</span>' for label in labels) + '</div>')
