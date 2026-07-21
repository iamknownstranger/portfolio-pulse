"""Pure option-chain processing and Black-Scholes math.

Extracted from the Option Chain page so the logic can be unit tested
without importing Streamlit.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE = 0.07


def calculate_greeks(S, K, T, r, iv, option_type="call"):
    """Option Greeks (Delta, Gamma, Theta, Vega) via Black-Scholes."""
    if T <= 0 or iv <= 0 or S <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}
    try:
        d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        vega = S * pdf_d1 * np.sqrt(T) / 100
        gamma = pdf_d1 / (S * iv * np.sqrt(T))

        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    except (ValueError, ZeroDivisionError):
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega}


def calculate_max_pain(df):
    """Strike where aggregate option-holder loss is maximized at expiry."""
    if "CE_OI" not in df.columns or "PE_OI" not in df.columns:
        return 0
    strikes = df.index.values
    ce_oi = df["CE_OI"].values
    pe_oi = df["PE_OI"].values
    total_loss = [
        np.sum(np.maximum(strikes - s, 0) * ce_oi) + np.sum(np.maximum(s - strikes, 0) * pe_oi)
        for s in strikes
    ]
    if not total_loss:
        return 0
    return strikes[np.argmin(total_loss)]


def build_final_df(options_list, expiry_date):
    """Combine per-option records into a strike-indexed CE_/PE_ DataFrame."""
    if not options_list:
        return pd.DataFrame()
    df = pd.DataFrame(options_list)
    # Drop Type after splitting: the CE_/PE_ prefixes carry it, and keeping a
    # string column breaks numeric formatting of the combined frame.
    df_calls = df[df["Type"] == "CE"].drop(columns="Type").set_index("Strike").add_prefix("CE_")
    df_puts = df[df["Type"] == "PE"].drop(columns="Type").set_index("Strike").add_prefix("PE_")
    full_df = pd.concat([df_calls, df_puts], axis=1).sort_index()
    full_df.columns.name = expiry_date
    return full_df.fillna(0)


def process_nse_data(data, selected_expiry, current_price):
    """Build an option-chain DataFrame from the NSE JSON format."""
    options = []
    records = data.get("records", {}).get("data", [])
    expiry_dt = datetime.strptime(selected_expiry, "%d-%b-%Y")
    time_to_expiry = (expiry_dt - datetime.now() + timedelta(hours=8)).days / 365.0

    for record in records:
        # Legacy NSE rows carry "expiryDate"; the option-chain-v3 API uses
        # "expiryDates" (same '%d-%b-%Y' format, one expiry per response).
        record_expiry = record.get("expiryDate", record.get("expiryDates"))
        if record_expiry != selected_expiry:
            continue
        for opt_type in ["CE", "PE"]:
            option_data = record.get(opt_type, {})
            if not option_data or "strikePrice" not in option_data:
                continue

            iv = option_data.get("impliedVolatility", 0) / 100
            greeks = calculate_greeks(
                current_price, option_data["strikePrice"], time_to_expiry,
                RISK_FREE_RATE, iv, "call" if opt_type == "CE" else "put",
            )

            options.append({
                "Type": opt_type, "Strike": option_data["strikePrice"],
                "LTP": option_data.get("lastPrice", 0),
                "IV": iv * 100, "OI": option_data.get("openInterest", 0),
                "Chg_OI": option_data.get("changeinOpenInterest", 0),
                "Volume": option_data.get("totalTradedVolume", 0), **greeks,
            })
    return build_final_df(options, selected_expiry)


def process_yfinance_data(data, selected_expiry, current_price):
    """Build an option-chain DataFrame from the yfinance option_chain format."""
    options = []
    expiry_dt = datetime.strptime(selected_expiry, "%Y-%m-%d")
    time_to_expiry = (expiry_dt - datetime.now() + timedelta(hours=8)).days / 365.0

    for df_type, opt_type in [(data.calls, "CE"), (data.puts, "PE")]:
        for _, row in df_type.iterrows():
            iv = row.get("impliedVolatility", 0)
            greeks = calculate_greeks(
                current_price, row["strike"], time_to_expiry,
                RISK_FREE_RATE, iv, "call" if opt_type == "CE" else "put",
            )

            options.append({
                "Type": opt_type, "Strike": row["strike"],
                "LTP": row.get("lastPrice", 0),
                "IV": iv * 100, "OI": row.get("openInterest", 0),
                "Chg_OI": 0,  # yfinance doesn't provide change in OI
                "Volume": row.get("volume", 0), **greeks,
            })
    return build_final_df(options, selected_expiry)
