"""Constrained allocation estimates and an efficient frontier on a common model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

from common.portfolio import normalized_weights


@dataclass
class Optimization:
    weights: pd.Series
    expected_returns: pd.Series
    covariance: pd.DataFrame
    method: str
    notice: str | None
    frontier: pd.DataFrame

    def performance(self, weights, risk_free_rate=0.0):
        w = normalized_weights(self.expected_returns.index, weights)
        mean = float(w.dot(self.expected_returns))
        vol = float(np.sqrt(max(0, w @ self.covariance @ w)))
        return mean, vol, (mean - risk_free_rate) / vol if vol > 1e-12 else np.nan


def optimize_allocation(prices, method="Maximum Sharpe", max_weight=1.0, risk_free_rate=0.0):
    if len(prices) < 20:
        raise ValueError("Optimization needs at least 20 common price observations. Extend the date range.")
    n = len(prices.columns)
    if not n or not np.isfinite(max_weight) or not 0 < max_weight <= 1 or n * max_weight < 1 - 1e-8:
        raise ValueError("The allocation cap must allow all weights to sum to 100%.")
    if method not in {"Maximum Sharpe", "Minimum volatility", "Equal weight"}:
        raise ValueError("Unknown optimization method.")
    mu = expected_returns.mean_historical_return(prices, compounding=False)
    cov = risk_models.sample_cov(prices)
    if not np.isfinite(mu).all() or not np.isfinite(cov).all().all():
        raise ValueError("There is insufficient valid return history to estimate this allocation.")
    notice = None

    def engine():
        # Always create a fresh instance; a failed max_sharpe mutates its solver.
        return EfficientFrontier(mu, cov, weight_bounds=(0, max_weight))

    if method == "Equal weight" or n == 1:
        weights = normalized_weights(prices.columns)
    else:
        ef = engine()
        try:
            if method == "Maximum Sharpe":
                ef.max_sharpe(risk_free_rate=risk_free_rate)
            else:
                ef.min_volatility()
        except Exception as exc:
            if method != "Maximum Sharpe":
                raise ValueError("The optimizer could not find a feasible allocation. Try a higher cap.") from exc
            method = "Minimum volatility"
            notice = "Maximum Sharpe was unavailable for these estimates; showing a fresh minimum-volatility solution."
            ef = engine()
            try:
                ef.min_volatility()
            except Exception as fallback_exc:
                raise ValueError("The optimizer could not find a feasible allocation. Try a higher cap.") from fallback_exc
        weights = pd.Series(ef.weights, index=mu.index).clip(lower=0)
        weights /= weights.sum()
    points = []
    if n > 1 and n * max_weight > 1 + 1e-6:
        minimum = engine()
        try:
            minimum.min_volatility()
        except Exception:
            # The frontier is supplementary; an unavailable plot must not
            # discard a valid equal-weight or already-solved target.
            return Optimization(weights, mu, cov, method, notice, pd.DataFrame())
        low = minimum.portfolio_performance(risk_free_rate=risk_free_rate)[0]
        remaining, high = 1.0, 0.0
        for value in mu.sort_values(ascending=False):
            allocation = min(max_weight, remaining)
            high += allocation * value
            remaining -= allocation
        for target in np.linspace(low, high, 18):
            try:
                ef = engine()
                ef.efficient_return(float(target))
                mean, vol, _ = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                points.append({"Return": mean, "Volatility": vol})
            except Exception:
                # Exact endpoints can be numerically infeasible. An omitted
                # point never changes the allocation selected above.
                continue
    return Optimization(weights, mu, cov, method, notice, pd.DataFrame(points))
