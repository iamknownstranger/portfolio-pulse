import numpy as np
import pandas as pd
import pytest
from pypfopt.exceptions import OptimizationError

from common import optimization


@pytest.fixture
def history():
    rng = np.random.default_rng(7)
    returns = rng.normal([.001, .0005, .0003], [.01, .008, .006], (160, 3))
    return pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)), columns=['A', 'B', 'C'])


@pytest.mark.parametrize('method', ['Maximum Sharpe', 'Minimum volatility', 'Equal weight'])
def test_targets_are_fully_invested_long_only_and_respect_cap(history, method):
    result = optimization.optimize_allocation(history, method, .4)
    assert result.weights.sum() == pytest.approx(1)
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= .4 + 1e-6).all()
    assert np.isfinite(result.performance(result.weights)).all()
    if not result.frontier.empty:
        assert (result.frontier['Return'].diff().dropna() >= -1e-8).all()
        assert (result.frontier['Volatility'].diff().dropna() >= -1e-5).all()


@pytest.mark.parametrize('failure', [ValueError, OptimizationError])
def test_failed_max_sharpe_uses_a_fresh_solver(history, monkeypatch, failure):
    instances = []
    original = optimization.EfficientFrontier

    class FailingSharpe(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.failed = False
            instances.append(self)

        def max_sharpe(self, **kwargs):
            self.failed = True
            raise failure('maximum-Sharpe solution unavailable')

        def min_volatility(self):
            assert not self.failed, 'a mutated max-Sharpe solver was reused'
            return super().min_volatility()

    monkeypatch.setattr(optimization, 'EfficientFrontier', FailingSharpe)
    result = optimization.optimize_allocation(history)
    assert result.method == 'Minimum volatility'
    assert result.notice
    assert len(instances) > 1


def test_infeasible_constraints_and_short_history_are_actionable(history):
    with pytest.raises(ValueError, match='cap'):
        optimization.optimize_allocation(history, max_weight=.2)
    with pytest.raises(ValueError, match='20'):
        optimization.optimize_allocation(history.iloc[:10])


def test_single_asset_is_a_valid_fully_invested_model(history):
    result = optimization.optimize_allocation(history[['A']])
    assert result.weights.to_dict() == {'A': 1}
    assert result.frontier.empty
