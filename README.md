# Portfolio Pulse

A Streamlit investment research workspace for understanding portfolio performance, managing model allocations, and exploring markets across asset classes.

## Start locally

Use Python 3.11 or later:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`1_💼_Portfolio_Pulse.py` is an equivalent entrypoint for existing Streamlit Community Cloud deployments. The `.devcontainer/` setup is also available for VS Code and Codespaces.

## Portfolio workflow

1. Start with the deterministic AAPL, MSFT, GOOGL, AMZN and NVDA model. Open **Portfolio settings** in the sidebar to paste up to 30 tickers or adjust the model's date range and benchmark.
2. Select dates, a benchmark and an annual risk-free rate. Click **Update analysis** to apply the filters. They persist when switching portfolio pages.
3. Use **Holdings & Exposure** to enter model weights totalling 100%, or reset to equal weights. The same model drives portfolio performance, risk and rebalancing.
4. Review growth, drawdowns, return attribution, monthly performance and correlations.
5. In **Rebalancing & Optimization**, choose maximum Sharpe, minimum volatility or equal weight, set a per-asset cap, and export an illustrative dollar rebalance plan.
6. Download adjusted prices, returns, model growth, allocation or attribution CSVs. Use **Refresh market data** to reload cached prices.

The portfolio is a hypothetical fully invested model rebalanced to fixed weights at every observed common close. It excludes costs, taxes and cash flows. Custom weights are not imported brokerage positions, and no trades are executed. Settings persist within the current session.

## Dashboards

| Group | Tools |
| --- | --- |
| Equities — Linear | Portfolio Pulse, Index Insights, Risk Wall, Performance Analytics, Rebalancing & Optimization, Holdings & Exposure |
| Equities — Non-Linear | NSE index option chains, Yahoo fallback, Greeks, Max Pain, open-interest and implied-volatility charts |
| Fixed Income | Treasury yield curve, slope/inversion, DXY and VIX gauges |
| Commodities | Futures performance, rolling volatility and return correlations |
| Digital Assets | Crypto risk/return metrics with 365-day annualization, drawdowns, ETH/BTC strength and traditional-asset correlations |
| Thematic & Flows | Smart Money Radar: thematic accumulation scores, lifecycle stages, rotation, stage changes and SEC filing confirmation |

The overview leads with total return, annual volatility, maximum drawdown and Sharpe. Tabs provide asset details, return correlations and exports. Portfolio analytics use a shared, UI-independent calculation engine; benchmark prices align before returns are calculated.

Equities — Linear is the first sidebar section, expanded by default, and Portfolio Pulse is the landing page. Categories with a single dashboard remain direct links under the category name.

The shared workspace theme includes a branded sidebar, consistent research headers, metric cards, compact tabs, and coordinated chart colors. On phones, the sidebar starts collapsed and portfolio KPIs use a two-column grid. Presentation lives in `common/presentation.py` and `assets/workspace.css`, separate from the calculation engine.

## Market data and assumptions

Portfolio analytics use Yahoo split/dividend-adjusted closes. Index symbols such as the S&P 500 (`^GSPC`) are price indices and exclude reinvested dividends; SPY and QQQ ETF benchmark choices are available for adjusted-price comparisons. Dates are inclusive. Prices are cached for up to one hour.

The raw-close DuckDB store is used by other data consumers only when all requested symbols and date bounds are covered. It has no market-data table in the supplied checkout. The market-cap membership archive ends on **19 March 2025**. Index Insights labels archive dates, holds selected membership fixed, and calculates returns from asset prices rather than changes in market caps. The Top 100 US benchmark uses a historical CSV universe, not a live point-in-time ranking.

Risk metrics use 252 observations per year for equities. Drawdowns include the initial investment; VaR, expected shortfall and EWMA VaR share a one-observation horizon. Model weights remain fixed when price dates are missing; common complete dates are used without forward-filling, with excluded rows disclosed. Consecutive observations can then span multiple trading days. Optimization estimates are in-sample, not forecasts.

### Smart Money Radar

The radar combines two evidence layers:

- **Price/volume accumulation proxies:** Chaikin Money Flow, dollar-volume thrust, relative-strength acceleration and breadth. These are not measured institutional fund flows.
- **SEC EDGAR filings:** Form 4 open-market purchases (`P` only) and 13F institutional position changes. 13F filings can lag quarter end by 45 days.

For SEC fair access, set a monitored contact address in the request User-Agent:

```sh
export SEC_CONTACT_EMAIL="you@example.com"
```

## Development and tests

```sh
pip install -r requirements-dev.txt
pytest -q
python -m py_compile streamlit_app.py common/*.py views/*.py
python scripts/check_streamlit_startup.py
```

Tests include portfolio math, optimizer constraints, provider adapters, archived membership, options, thematic scoring and SEC parsers. Streamlit integration tests exercise routing, persistent filters, allocations and provider failure states using deterministic data. CI also starts a real server and verifies its health, HTML, JavaScript and favicon responses with gzip enabled.

Keep the Starlette compatibility pin in `requirements.txt`: Streamlit 1.58's gzip middleware is incompatible with Starlette 1.x. An import check or healthy server response alone does not catch its static-file HTTP 500 error.

See [the product and architecture review](docs/PRODUCT_REVIEW.md) for the codebase map, corrected behaviors, model contract and prioritized next improvements.
