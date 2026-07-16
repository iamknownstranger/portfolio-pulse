import numpy as np
import pandas as pd
import pytest

from common.options import build_final_df, calculate_greeks, calculate_max_pain


class TestCalculateGreeks:
    def test_call_delta_between_zero_and_one(self):
        greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.07, iv=0.2, option_type="call")
        assert 0 < greeks["Delta"] < 1

    def test_put_delta_between_minus_one_and_zero(self):
        greeks = calculate_greeks(S=100, K=100, T=0.25, r=0.07, iv=0.2, option_type="put")
        assert -1 < greeks["Delta"] < 0

    def test_put_call_delta_parity(self):
        # call delta - put delta = 1 for identical parameters
        call = calculate_greeks(S=100, K=105, T=0.5, r=0.07, iv=0.3, option_type="call")
        put = calculate_greeks(S=100, K=105, T=0.5, r=0.07, iv=0.3, option_type="put")
        assert call["Delta"] - put["Delta"] == pytest.approx(1.0)

    def test_gamma_and_vega_shared_between_call_and_put(self):
        call = calculate_greeks(S=100, K=95, T=0.5, r=0.07, iv=0.25, option_type="call")
        put = calculate_greeks(S=100, K=95, T=0.5, r=0.07, iv=0.25, option_type="put")
        assert call["Gamma"] == pytest.approx(put["Gamma"])
        assert call["Vega"] == pytest.approx(put["Vega"])

    def test_deep_itm_call_delta_approaches_one(self):
        greeks = calculate_greeks(S=200, K=100, T=0.1, r=0.07, iv=0.2, option_type="call")
        assert greeks["Delta"] > 0.99

    def test_theta_is_negative_for_atm_options(self):
        call = calculate_greeks(S=100, K=100, T=0.25, r=0.07, iv=0.2, option_type="call")
        assert call["Theta"] < 0

    @pytest.mark.parametrize("S,K,T,iv", [
        (0, 100, 0.25, 0.2),   # zero spot
        (100, 100, 0, 0.2),    # expired
        (100, 100, -0.1, 0.2), # past expiry
        (100, 100, 0.25, 0),   # zero volatility
    ])
    def test_degenerate_inputs_return_zero_greeks(self, S, K, T, iv):
        greeks = calculate_greeks(S=S, K=K, T=T, r=0.07, iv=iv)
        assert greeks == {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}


class TestCalculateMaxPain:
    def test_max_pain_on_symmetric_chain(self):
        # OI concentrated so that the middle strike minimizes writer payout
        df = pd.DataFrame({
            "CE_OI": [100, 100, 100],
            "PE_OI": [100, 100, 100],
        }, index=[90, 100, 110])
        assert calculate_max_pain(df) == 100

    def test_max_pain_pulled_to_heavy_put_strike(self):
        # Huge put OI at the lowest strike pulls max pain down to it:
        # settling at 90 keeps those puts worthless, settling higher
        # pays them out heavily.
        df = pd.DataFrame({
            "CE_OI": [10, 10, 10],
            "PE_OI": [10000, 10, 10],
        }, index=[90, 100, 110])
        assert calculate_max_pain(df) == 90

    def test_missing_columns_returns_zero(self):
        assert calculate_max_pain(pd.DataFrame({"other": [1]})) == 0

    def test_empty_chain_returns_zero(self):
        df = pd.DataFrame({"CE_OI": [], "PE_OI": []})
        assert calculate_max_pain(df) == 0


class TestBuildFinalDf:
    def test_empty_list_returns_empty_df(self):
        assert build_final_df([], "01-Jan-2026").empty

    def test_combines_calls_and_puts_by_strike(self):
        options = [
            {"Type": "CE", "Strike": 100, "LTP": 5.0, "IV": 20.0, "OI": 10,
             "Chg_OI": 1, "Volume": 3, "Delta": 0.5, "Gamma": 0.01, "Theta": -0.1, "Vega": 0.2},
            {"Type": "PE", "Strike": 100, "LTP": 4.0, "IV": 22.0, "OI": 12,
             "Chg_OI": 2, "Volume": 5, "Delta": -0.5, "Gamma": 0.01, "Theta": -0.1, "Vega": 0.2},
            {"Type": "CE", "Strike": 110, "LTP": 1.0, "IV": 25.0, "OI": 7,
             "Chg_OI": 0, "Volume": 1, "Delta": 0.2, "Gamma": 0.02, "Theta": -0.05, "Vega": 0.1},
        ]
        df = build_final_df(options, "01-Jan-2026")
        assert list(df.index) == [100, 110]
        assert df.loc[100, "CE_LTP"] == 5.0
        assert df.loc[100, "PE_LTP"] == 4.0
        # Strike 110 has no put — filled with 0, not NaN
        assert df.loc[110, "PE_LTP"] == 0
        assert not df.isna().any().any()
