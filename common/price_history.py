"""Shared price/calendar cleaning, independent of cached portfolio modules.

Keep these inputs below the portfolio and backtesting engines so a newly loaded
backtest does not depend on new exports in an already-imported portfolio module.
"""

import numpy as np
import pandas as pd


def normalize_observation_dates(frame):
    """Use unique UTC calendar labels; reject missing dates and keep the last duplicate."""
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    if frame.index.hasnans:
        raise ValueError("Every observation needs a valid date.")
    return frame.loc[~frame.index.duplicated(keep="last")].sort_index()


def clean_prices(prices):
    """Positive numeric prices on unique, sorted, timezone-free dates."""
    frame = normalize_observation_dates(prices)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.where(np.isfinite(frame) & (frame > 0))
