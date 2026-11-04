# Streamlit Cloud import regression

Reported after merge commit `944a6b9`: Rebalancing failed while importing `common.backtesting`, at the import of `normalize_observation_dates` from `common.portfolio`.

The merged source contains that export. A running interpreter can still hold an older `common.portfolio` namespace while newly executed page source imports the backtester for the first time. The reported traceback is consistent with that state; remote process logs were not available to independently inspect its module cache.

## Reproduction and change

Five new regression cases retain a portfolio namespace without the newer date-helper export, then freshly import backtesting or run Backtest/Rebalancing through each app entrypoint. All five failed before the fix with `ImportError: cannot import name 'normalize_observation_dates' from 'common.portfolio'`, matching the reported failing import.

Shared calendar normalization and numeric price cleaning now live in `common/price_history.py`, below both calculation engines. Backtesting and its benchmark view import them directly from that module. `common.portfolio` keeps the existing exported names for current consumers. The backtester therefore relies only on longstanding portfolio exports when a worker retains the earlier portfolio namespace.

The regression also verifies missing timestamps remain rejected under this cached-portfolio condition, preserving the accounting checks from the prior review. Calculation rules, engine version, requirements and saved-run formats are unchanged.

## Validation

- Python 3.12: **297 tests passed**, including the five previously failing cases.
- Python 3.13.5, matching the deployed major/minor version: **297 tests passed**.
- Python 3.13 source compilation and the real Streamlit health, HTML, JavaScript and favicon check passed with gzip enabled.
- CI now runs the full suite on both Python 3.11 and 3.13. The deployment job imports the backtesting and price-history modules as well as external dependencies.

The earlier checks used fresh module imports and missed the cached-namespace update path. These tests model that path explicitly; they do not claim an end-to-end Streamlit Cloud deployment was performed.

## Operational recovery

Streamlit Cloud's Reboot app action reloads imported modules from the currently checked-out source and should clear this cached-import condition immediately. The code fix is delivered through a pull request; production adoption requires merging it into the app's configured branch. If an existing worker remains stale afterward, reboot it once.
