"""Portfolio math shared by the equity dashboards; no UI or network calls.

The model is long-only, fully invested and rebalanced to fixed weights at each
observed close. Prices are restricted to a common calendar before calculating
returns. No forward-filling, changing constituent weights or cash flows are
implied by missing observations.
"""

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def normalize_symbols(value):
    """Parse pasted symbols, normalize Yahoo class-share notation and deduplicate."""
    tokens = re.split(r"[,;\s]+", value.strip()) if isinstance(value, str) else value
    tokens = [str(t).strip().upper() for t in tokens if str(t).strip()]
    symbols = list(dict.fromkeys(re.sub(r"^([A-Z]+)\.([AB])$", r"\1-\2", t) for t in tokens))
    if not symbols:
        raise ValueError("Add at least one ticker to your portfolio.")
    if len(symbols) > 30:
        raise ValueError("Use up to 30 tickers for a readable portfolio analysis.")
    invalid = [s for s in symbols if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,24}", s)]
    if invalid:
        raise ValueError("Check these ticker symbols: " + ", ".join(invalid))
    return symbols


def normalized_weights(symbols, weights=None):
    """Validate a complete, fully invested weight vector; defaults to equal weight."""
    if not len(symbols):
        raise ValueError("At least one asset is required.")
    if weights is None:
        return pd.Series(1 / len(symbols), index=symbols, dtype=float)
    w = pd.Series(weights, dtype=float).reindex(symbols)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Every asset needs a finite, non-negative allocation.")
    if not np.isclose(w.sum(), 1, atol=1e-4, rtol=0):
        raise ValueError(f"Allocations must total 100% (currently {w.sum():.2%}).")
    return w / w.sum()


def clean_prices(prices):
    """Positive numeric prices on unique, sorted, timezone-free dates."""
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.where(np.isfinite(frame) & (frame > 0))


def wealth_index(returns):
    """Growth from return observations; drawdowns must also consider initial wealth 1."""
    return (1 + returns).cumprod()


def drawdowns(wealth):
    return wealth / wealth.cummax().clip(lower=1) - 1


def performance_stats(returns, risk_free_rate=0.0, periods=TRADING_DAYS, start=None):
    """Daily-return statistics, fractional units, sample volatility and daily VaR.

    CAGR uses elapsed calendar time when the initial price date is supplied.
    Sortino uses the RMS of negative excess returns across *all* observations.
    """
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    keys = ["total_return", "cagr", "annual_return", "volatility", "sharpe", "sortino",
            "max_drawdown", "var_95", "cvar_95", "ewma_var_95", "worst_day"]
    if r.empty:
        return dict.fromkeys(keys, np.nan)
    wealth = wealth_index(r)
    annual_return = r.mean() * periods
    vol = r.std() * np.sqrt(periods)
    rf_daily = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = r - rf_daily
    downside = np.sqrt(np.minimum(excess, 0).pow(2).mean()) * np.sqrt(periods)
    quantile = r.quantile(0.05)
    decay = 0.94 ** np.arange(len(r) - 1, -1, -1)
    daily_ewma = np.sqrt(np.average(r.to_numpy() ** 2, weights=decay))
    years = len(r) / periods
    if start is not None and isinstance(r.index, pd.DatetimeIndex):
        years = (r.index[-1] - pd.Timestamp(start)).days / 365.25
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    return {
        "total_return": wealth.iloc[-1] - 1,
        "cagr": cagr,
        "annual_return": annual_return,
        "volatility": vol,
        "sharpe": excess.mean() * periods / vol if vol > 1e-12 else np.nan,
        "sortino": excess.mean() * periods / downside if downside > 1e-12 else np.nan,
        "max_drawdown": drawdowns(wealth).min(),
        "var_95": max(0.0, -quantile),
        "cvar_95": max(0.0, -r[r <= quantile].mean()),
        "ewma_var_95": 1.6448536269514722 * daily_ewma,
        "worst_day": r.min(),
    }


@dataclass
class Portfolio:
    prices: pd.DataFrame
    weights: pd.Series
    asset_returns: pd.DataFrame
    returns: pd.Series
    wealth: pd.Series
    excluded_rows: int

    def stats(self, risk_free_rate=0.0):
        return performance_stats(self.returns, risk_free_rate, start=self.prices.index[0])

    def contributions(self):
        """Linked contributions sum exactly to this daily-rebalanced model's return."""
        prior_wealth = self.wealth.shift(1).reindex(self.returns.index)
        return self.asset_returns.mul(self.weights).mul(prior_wealth, axis=0).sum()


def analyze_portfolio(prices, weights=None):
    if prices.empty or not len(prices.columns):
        raise ValueError("No price data is available. Try another date range or ticker.")
    clean = clean_prices(prices)
    absent = clean.columns[clean.notna().sum() < 3].tolist()
    if absent:
        raise ValueError("At least three prices are needed for: " + ", ".join(absent))
    common = clean.dropna(how="any")
    if len(common) < 3:
        raise ValueError("At least three common price dates are needed. Extend the date range.")
    w = normalized_weights(common.columns, weights)
    asset_returns = common.pct_change(fill_method=None).iloc[1:]
    portfolio_returns = asset_returns.dot(w).rename("Portfolio")
    wealth = pd.concat([
        pd.Series([1.0], index=common.index[:1]), wealth_index(portfolio_returns),
    ]).rename("Portfolio")
    return Portfolio(common, w, asset_returns, portfolio_returns, wealth, len(clean) - len(common))


def compare_with_benchmark(prices, benchmark, weights=None):
    """Align price observations *before* returns so both paths have the same baseline."""
    clean = clean_prices(prices)
    bench = clean_prices(benchmark.rename("benchmark").to_frame()).iloc[:, 0]
    dates = clean.dropna().index.intersection(bench.dropna().index)
    portfolio = analyze_portfolio(clean.loc[dates], weights)
    benchmark_prices = bench.loc[portfolio.prices.index]
    benchmark_returns = benchmark_prices.pct_change(fill_method=None).iloc[1:]
    benchmark_wealth = benchmark_prices / benchmark_prices.iloc[0]
    return portfolio, benchmark_returns, benchmark_wealth


def relative_stats(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    aligned = pd.concat([portfolio_returns.rename("p"), benchmark_returns.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        return dict.fromkeys(["beta", "alpha", "correlation", "tracking_error", "information_ratio"], np.nan)
    p, b = aligned["p"], aligned["b"]
    variance = b.var()
    beta = p.cov(b) / variance if variance > 1e-16 else np.nan
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    alpha = ((p.mean() - rf_daily) - beta * (b.mean() - rf_daily)) * TRADING_DAYS
    active = p - b
    tracking_error = active.std() * np.sqrt(TRADING_DAYS)
    return {
        "beta": beta, "alpha": alpha,
        "correlation": p.corr(b) if p.std() > 1e-12 and b.std() > 1e-12 else np.nan,
        "tracking_error": tracking_error,
        "information_ratio": active.mean() * TRADING_DAYS / tracking_error if tracking_error > 1e-12 else np.nan,
    }


def rebalance_plan(current, target, portfolio_value):
    """Dollar changes for a fully invested model; no assumed share counts or execution."""
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive.")
    symbols = list(current)
    if set(symbols) != set(target):
        raise ValueError("Current and target allocations must contain the same assets.")
    current = normalized_weights(symbols, current)
    target = normalized_weights(symbols, target)
    plan = pd.DataFrame({"Current weight": current, "Target weight": target})
    plan["Change"] = target - current
    plan["Trade value ($)"] = plan["Change"] * portfolio_value
    plan["Action"] = np.where(plan["Change"].abs() < 1e-6, "Hold", np.where(plan["Change"] > 0, "Buy", "Sell"))
    return plan.rename_axis("Asset")
