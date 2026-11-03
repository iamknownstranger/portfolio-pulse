"""Deterministic daily equity research. No network, UI, fitting or order execution.

Signals observed at close t become target weights at close t+1. Holdings drift
between rebalances; cash earns a calendar-time effective rate. Costs apply to
both sides of asset turnover and are funded before target holdings are sized.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from common.portfolio import normalized_weights, performance_stats
from common.price_history import clean_prices, normalize_observation_dates

ENGINE_VERSION = "1.1"
STRATEGIES = ("Buy & hold", "Periodic rebalance", "Trend filter", "Momentum rotation")
FREQUENCIES = ("Daily", "Weekly", "Monthly", "Quarterly")
LEDGER_COLUMNS = ["Signal date", "Execution date", "Asset", "Side", "Units", "Reference price",
                  "Notional", "Commission", "Slippage", "Reason"]


@dataclass(frozen=True)
class BacktestConfig:
    strategy: str = "Periodic rebalance"
    frequency: str = "Monthly"
    lookback: int = 63
    top_n: int = 2
    initial_capital: float = 10000.0
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    cash_rate: float = 0.0

    def validate(self):
        if self.strategy not in STRATEGIES or self.frequency not in FREQUENCIES:
            raise ValueError("Choose a supported strategy and rebalance frequency.")
        if not isinstance(self.lookback, int) or not 2 <= self.lookback <= 504:
            raise ValueError("Signal lookback must be 2–504 observations.")
        if not isinstance(self.top_n, int) or self.top_n < 1:
            raise ValueError("Momentum rotation needs at least one holding.")
        if not np.isfinite(self.initial_capital) or not 0 < self.initial_capital <= 1e12:
            raise ValueError("Starting capital must be positive and at most $1 trillion.")
        for cost in (self.commission_bps, self.slippage_bps):
            if not np.isfinite(cost) or not 0 <= cost <= 1000:
                raise ValueError("Each trading cost must be between 0 and 1,000 basis points.")
        if not np.isfinite(self.cash_rate) or not 0 <= self.cash_rate <= 1:
            raise ValueError("The annual cash rate must be between 0% and 100%.")


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series
    returns: pd.Series
    allocations: pd.DataFrame
    ledger: pd.DataFrame
    costs: pd.DataFrame
    signals: pd.DataFrame
    targets: pd.DataFrame
    excluded_rows: int

    def stats(self, risk_free_rate=0.0):
        stats = performance_stats(self.returns, risk_free_rate, start=self.equity.index[0])
        stats.update(final_value=float(self.equity.iloc[-1]),
                     total_cost=float(self.costs.sum().sum()), executions=len(self.ledger),
                     average_exposure=float(1 - self.allocations["Cash"].mean()))
        return stats


def _prepared_prices(prices):
    if prices.empty or not prices.columns.is_unique:
        raise ValueError("Supply positive prices with unique asset names.")
    clean = clean_prices(prices)
    common = clean.dropna(how="any")
    if len(common) < 3:
        raise ValueError("Backtesting needs at least three common price dates.")
    if "Cash" in common.columns:
        raise ValueError("Cash is reserved for the cash allocation.")
    return common, len(clean) - len(common)


def decision_dates(index, frequency):
    """Decide at the first observed close of a new period, using no future dates."""
    if frequency == "Daily":
        return pd.Series(True, index=index)
    period = index.to_period({"Weekly": "W-FRI", "Monthly": "M", "Quarterly": "Q"}[frequency])
    return pd.Series(np.r_[True, period[1:] != period[:-1]], index=index)


def strategy_targets(prices, weights, config):
    """Decision-time targets; all-NaN rows mean hold, all-zero rows mean cash."""
    targets = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    schedule = decision_dates(prices.index, config.frequency)
    if config.strategy == "Buy & hold":
        targets.iloc[0] = weights
    elif config.strategy == "Periodic rebalance":
        targets.loc[schedule] = np.tile(weights.to_numpy(), (int(schedule.sum()), 1))
    elif config.strategy == "Trend filter":
        # Full lookback observations are required. Missing warm-up signals stay cash.
        signals = prices.div(prices.rolling(config.lookback).mean()) - 1
        targets = signals.gt(0).mul(weights, axis=1).astype(float)
    else:
        signals = prices.pct_change(config.lookback, fill_method=None)
        eligible = signals.gt(0) & signals.rank(axis=1, ascending=False, method="first").le(config.top_n)
        count = eligible.sum(axis=1).replace(0, np.nan)
        selected = eligible.div(count, axis=0).fillna(0.0)
        targets.loc[schedule] = selected.loc[schedule]
    return targets, signals


def _funded_rebalance(marked, equity, target, cost_rate):
    """Solve post-cost equity + turnover fee = pre-trade equity (no borrowing)."""
    full_value = target * equity
    if np.all(np.abs(full_value - marked) <= 4 * np.spacing(equity)):
        # Keep holdings when only floating-point roundoff separates them from target.
        return marked.copy()
    if cost_rate == 0:
        return full_value
    low, high = 0.0, equity
    for _ in range(48):
        net = (low + high) / 2
        total = net + np.abs(target * net - marked).sum() * cost_rate
        if total > equity:
            high = net
        else:
            low = net
    return target * ((low + high) / 2)


def simulate_targets(prices, targets, config, signals=None):
    """Execute close-t decisions at the next common close, including initial entry.

    The last decision cannot execute. Open holdings remain marked at the final
    close; no terminal liquidation fee is fabricated. Units use adjusted prices
    and are synthetic fractional units, not split-adjusted brokerage quantities.
    """
    config.validate()
    prices, excluded = _prepared_prices(prices)
    if not targets.columns.is_unique or set(targets.columns) != set(prices.columns):
        raise ValueError("Targets must include exactly the price assets, without duplicates.")
    targets = normalize_observation_dates(targets)
    targets = targets.reindex(index=prices.index, columns=prices.columns).astype(float)
    rows = targets.notna().any(axis=1)
    active = targets.loc[rows]
    if (active.isna().any().any() or not np.isfinite(active).all().all()
            or (active < 0).any().any() or (active.sum(axis=1) > 1 + 1e-10).any()):
        raise ValueError("Targets must be complete, non-negative and sum to at most 100%.")
    # A tolerated floating-point overshoot must not create equity or borrowed cash.
    targets.loc[rows] = active.div(active.sum(axis=1).clip(lower=1.0), axis=0)
    units = np.zeros(len(prices.columns))
    cash = config.initial_capital
    equity_values, allocations, ledger, costs = [], [], [], []
    values = prices.to_numpy()
    target_values = targets.to_numpy()
    total_rate = (config.commission_bps + config.slippage_bps) / 10000
    for i, timestamp in enumerate(prices.index):
        if i:
            days = (timestamp - prices.index[i - 1]).days
            cash *= (1 + config.cash_rate) ** (days / 365.25)
        marked = units * values[i]
        equity = float(cash + marked.sum())
        commission = slippage = 0.0
        if i and np.isfinite(target_values[i - 1]).all():
            desired = _funded_rebalance(marked, equity, target_values[i - 1], total_rate)
            trades = desired - marked
            # Every holding change and fee must be reconstructible from the ledger.
            for j in np.flatnonzero(trades):
                notional = abs(float(trades[j]))
                fee = notional * config.commission_bps / 10000
                slip = notional * config.slippage_bps / 10000
                ledger.append([prices.index[i - 1], timestamp, prices.columns[j],
                               "Buy" if trades[j] > 0 else "Sell", float(trades[j] / values[i, j]),
                               values[i, j], notional, fee, slip, config.strategy])
            commission = float(np.abs(trades).sum() * config.commission_bps / 10000)
            slippage = float(np.abs(trades).sum() * config.slippage_bps / 10000)
            units = desired / values[i]
            cash = max(0.0, equity - desired.sum() - commission - slippage)
            marked = desired
            equity = float(cash + marked.sum())
        equity_values.append(equity)
        allocations.append([*(marked / equity), cash / equity])
        costs.append([commission, slippage])
    equity = pd.Series(equity_values, index=prices.index, name="Strategy")
    return BacktestResult(config, equity, equity.pct_change(fill_method=None).iloc[1:],
                          pd.DataFrame(allocations, index=prices.index, columns=[*prices.columns, "Cash"]),
                          pd.DataFrame(ledger, columns=LEDGER_COLUMNS),
                          pd.DataFrame(costs, index=prices.index, columns=["Commission", "Slippage"]),
                          signals if signals is not None else pd.DataFrame(index=prices.index), targets, excluded)


def run_backtest(prices, weights=None, config=None):
    config = config or BacktestConfig()
    config.validate()
    common, excluded = _prepared_prices(prices)
    w = normalized_weights(common.columns, weights)
    targets, signals = strategy_targets(common, w, config)
    result = simulate_targets(common, targets, config, signals)
    result.excluded_rows = excluded
    return result


def validation_windows(equity, risk_free_rate=0.0, train_fraction=0.7):
    """Chronological diagnostics of fixed rules, retaining live position state.

    No fitting or parameter selection is done. Test returns start strictly after
    the split; the split close is the test's initial wealth baseline.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("Choose an initial window fraction between zero and one.")
    if len(equity) < 21:
        return pd.DataFrame()
    split = max(2, min(len(equity) - 3, int((len(equity) - 1) * train_fraction)))
    windows = [("Initial window", equity.iloc[:split + 1]), ("Later holdout", equity.iloc[split:])]
    rows = []
    for label, path in windows:
        returns = path.pct_change(fill_method=None).iloc[1:]
        stats = performance_stats(returns, risk_free_rate, start=path.index[0])
        rows.append({"Window": label, "Start": path.index[0], "End": path.index[-1],
                     "Observations": len(returns), "Return": stats["total_return"], "CAGR": stats["cagr"],
                     "Sharpe": stats["sharpe"], "Max drawdown": stats["max_drawdown"]})
    return pd.DataFrame(rows).set_index("Window")


def rolling_windows(equity, risk_free_rate=0.0, window=63):
    """Non-overlapping return windows, each with its own initial wealth baseline."""
    rows = []
    for start in range(0, len(equity) - 1, window):
        path = equity.iloc[start:start + window + 1]
        if len(path) < 2:
            continue
        stats = performance_stats(path.pct_change(fill_method=None).iloc[1:], risk_free_rate, start=path.index[0])
        rows.append({"Start": path.index[0], "End": path.index[-1], "Observations": len(path) - 1,
                     "Return": stats["total_return"], "Sharpe": stats["sharpe"],
                     "Max drawdown": stats["max_drawdown"]})
    return pd.DataFrame(rows)


def signal_observations(prices, asset, lookback=63, horizon=5, kind="Momentum"):
    """Signal at t versus return from execution close t+1 to close t+1+h.

    Consecutive outcomes overlap; correlations are descriptive, not independent
    trials or statistical significance. Missing warm-up and tail rows are dropped.
    """
    if lookback < 2 or horizon < 1:
        raise ValueError("Use a lookback of at least two and a positive forward horizon.")
    common, _ = _prepared_prices(prices)
    price = common[asset]
    if kind not in {"Momentum", "Distance from moving average"}:
        raise ValueError("Choose a supported research signal.")
    signal = (price.pct_change(lookback, fill_method=None) if kind == "Momentum"
              else price / price.rolling(lookback).mean() - 1)
    forward = price.shift(-(horizon + 1)) / price.shift(-1) - 1
    return pd.DataFrame({"Signal": signal, "Forward return": forward}).replace([np.inf, -np.inf], np.nan).dropna()


def signal_deciles(observations):
    if len(observations) < 20 or observations["Signal"].nunique() < 2:
        return pd.DataFrame()
    frame = observations.copy()
    frame["Bucket"] = pd.qcut(frame["Signal"], 10, labels=False, duplicates="drop") + 1
    return frame.groupby("Bucket", observed=True)["Forward return"].agg(["mean", "count"]).rename(
        columns={"mean": "Mean forward return", "count": "Observations"})


def drawdown_episodes(wealth):
    """Peak, trough and recovery, including a loss immediately after investment."""
    if wealth.empty:
        return pd.DataFrame()
    peak, peak_date = float(wealth.iloc[0]), wealth.index[0]
    episode, rows = None, []
    for timestamp, value in wealth.items():
        if value >= peak:
            if episode is not None:
                episode["Recovery"] = timestamp
                episode["Days underwater"] = (timestamp - episode["Peak"]).days
                rows.append(episode)
                episode = None
            peak, peak_date = float(value), timestamp
        else:
            depth = value / peak - 1
            if episode is None:
                episode = {"Peak": peak_date, "Trough": timestamp, "Recovery": pd.NaT,
                           "Drawdown": depth, "Days underwater": 0}
            if depth < episode["Drawdown"]:
                episode.update(Trough=timestamp, Drawdown=depth)
    if episode is not None:
        episode["Days underwater"] = (wealth.index[-1] - episode["Peak"]).days
        rows.append(episode)
    return pd.DataFrame(rows).sort_values("Drawdown").reset_index(drop=True) if rows else pd.DataFrame()


def run_manifest(result, prices, weights, benchmark, risk_free_rate=0.0, context=None):
    """Portable settings plus data fingerprint; prices remain a separate export."""
    prices = prices.reindex(index=result.equity.index)
    digest = hashlib.sha256(prices.to_csv(index_label="Date").encode()).hexdigest()
    manifest = {"engine_version": ENGINE_VERSION, "config": asdict(result.config),
                "weights": {str(k): float(v) for k, v in weights.items()},
                "benchmark": benchmark, "risk_free_rate": risk_free_rate,
                "observed_start": result.equity.index[0].isoformat(),
                "observed_end": result.equity.index[-1].isoformat(), "price_sha256": digest,
                "context": context or {},
                "execution": "Decision at close t; execution at next common close t+1",
                "data": "Yahoo split/dividend-adjusted closes; fixed present-day universe",
                "costs": "Commission and slippage in bps per side on absolute traded notional",
                "terminal_positions": "Marked at final close; no forced liquidation"}
    manifest["run_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return manifest


@dataclass
class WalkForwardResult:
    result: BacktestResult
    equal_weight: BacktestResult
    folds: pd.DataFrame
    weights: pd.DataFrame


def walk_forward_allocation(prices, method="Minimum volatility", max_weight=1.0, risk_free_rate=0.0,
                            training_observations=126, test_observations=63, config=None):
    """Refit on trailing history only; execute frozen targets one close later.

    Each fold trains on `training_observations` returns through its decision
    close. It is evaluated on subsequent returns before the next decision.
    The comparison equal-weights at the identical decision and execution dates.
    """
    from common.optimization import optimize_allocation

    common, excluded = _prepared_prices(prices)
    if not isinstance(training_observations, int) or training_observations < 20:
        raise ValueError("Walk-forward training needs at least 20 return observations.")
    if not isinstance(test_observations, int) or test_observations < 1:
        raise ValueError("A forward window needs at least one return observation.")
    if len(common) < training_observations + 3:
        raise ValueError(f"At least {training_observations + 3} common prices are needed for this training window.")
    config = config or BacktestConfig()
    config.validate()
    evaluation_prices = common.iloc[training_observations:]
    targets = pd.DataFrame(np.nan, index=evaluation_prices.index, columns=common.columns)
    reference_targets = targets.copy()
    folds, weight_rows = [], []
    for decision in range(training_observations, len(common) - 1, test_observations):
        train = common.iloc[decision - training_observations:decision + 1]
        allocation = optimize_allocation(train, method, max_weight, risk_free_rate, include_frontier=False)
        timestamp = common.index[decision]
        targets.loc[timestamp] = allocation.weights
        reference_targets.loc[timestamp] = 1 / len(common.columns)
        end = min(decision + test_observations, len(common) - 1)
        folds.append({"Train start": train.index[0], "Decision / train end": timestamp,
                      "First execution": common.index[decision + 1], "Test end": common.index[end],
                      "Test observations": end - decision, "Actual objective": allocation.method,
                      "Notice": allocation.notice or ""})
        weight_rows.append(allocation.weights.rename(timestamp))
    result = simulate_targets(evaluation_prices, targets, config)
    reference = simulate_targets(evaluation_prices, reference_targets, config)
    result.excluded_rows = reference.excluded_rows = excluded
    for fold in folds:
        path = result.equity.loc[fold["Decision / train end"]:fold["Test end"]]
        stats = performance_stats(path.pct_change(fill_method=None).iloc[1:], risk_free_rate, start=path.index[0])
        fold.update(Return=stats["total_return"], Sharpe=stats["sharpe"], **{"Max drawdown": stats["max_drawdown"]})
    return WalkForwardResult(result, reference, pd.DataFrame(folds), pd.DataFrame(weight_rows))
