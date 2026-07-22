import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import time
from datetime import datetime

from common.options import (
    calculate_max_pain,
    process_nse_data,
    process_yfinance_data,
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .success-message {
        background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724;
        padding: 0.75rem 1.25rem; border-radius: 0.25rem; margin-bottom: 1rem;
    }
    .warning-message {
        background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404;
        padding: 0.75rem 1.25rem; border-radius: 0.25rem; margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []
if 'data_source' not in st.session_state:
    st.session_state.data_source = ""

# --- CONSTANTS ---
INDEX_SYMBOLS = {
    "NIFTY": {"nse": "NIFTY", "yf": "^NSEI"},
    "BANKNIFTY": {"nse": "BANKNIFTY", "yf": "^NSEBANK"},
    "FINNIFTY": {"nse": "FINNIFTY", "yf": "NIFTY_FIN_SERVICE.NS"},
    "MIDCPNIFTY": {"nse": "MIDCPNIFTY", "yf": "^CNXMIDCAP"},
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
    # No explicit Accept-Encoding: requests advertises only codings it can
    # decode. Hardcoding "br" made NSE reply brotli, which fails without the
    # optional brotli package installed.
}

# --- DATA FETCHING & PROCESSING ---

def _nse_session():
    """Requests session with the cookies NSE requires for its JSON API."""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=10)  # Initialize session
    return session

@st.cache_data(ttl=300)
def fetch_option_data(index_key, expiry=None):
    """
    Fetches option chain data using a primary (NSE) and secondary (yfinance) source.
    Returns the data, the source it came from, and underlying price.

    NSE's option-chain-v3 API serves one expiry per request ('%d-%b-%Y'
    format); when expiry is None the nearest one is used. The response's
    records.expiryDates still lists every available expiry.
    """
    nse_symbol = INDEX_SYMBOLS[index_key]["nse"]
    yf_symbol = INDEX_SYMBOLS[index_key]["yf"]

    # --- PRIMARY SOURCE: NSE ---
    try:
        session = _nse_session()
        if expiry is None:
            # The old option-chain-indices endpoint (all expiries in one
            # response) was retired; expiries now come from contract-info.
            info = session.get(
                f"https://www.nseindia.com/api/option-chain-contract-info?symbol={nse_symbol}",
                timeout=15,
            )
            info.raise_for_status()
            expiry_dates = info.json().get("expiryDates", [])
            if not expiry_dates:
                raise ValueError("NSE returned no expiry dates")
            expiry = expiry_dates[0]
        response = session.get(
            f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={nse_symbol}&expiry={expiry}",
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("records", {}).get("data"):
            raise ValueError(f"NSE returned no option data for expiry {expiry}")
        st.session_state.data_source = "NSE (Live)"
        return data, "NSE", data.get('records', {}).get('underlyingValue')
    except Exception as e:
        st.warning(f"NSE API failed: {e}. Switching to yfinance as a fallback.")

    # --- FALLBACK SOURCE: yfinance ---
    try:
        ticker = yf.Ticker(yf_symbol)
        expiry_dates = ticker.options
        if not expiry_dates:
            return None, "Error", None

        # Select the nearest expiry date for yfinance data
        chain = ticker.option_chain(expiry_dates[0])
        st.session_state.data_source = "yfinance (Fallback)"
        # Get the latest price from the ticker object itself
        underlying_price = ticker.history(period="1d")['Close'].iloc[-1]
        return chain, "yfinance", underlying_price
    except Exception as e:
        st.error(f"Fallback yfinance API also failed: {e}")
        return None, "Error", None

# --- UI & ANALYSIS COMPONENTS ---

def create_summary_metrics(df, current_price):
    """Displays key summary metrics in metric cards."""
    st.markdown("#### 📈 Market Snapshot")
    cols = st.columns(6)
    
    if df.empty:
        for col in cols:
            col.metric("Data", "N/A")
        return

    total_ce_oi = df['CE_OI'].sum()
    total_pe_oi = df['PE_OI'].sum()
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    max_pain = calculate_max_pain(df)
    key_resistance = df['CE_OI'].idxmax() if not df.empty and not df['CE_OI'].empty else 0
    key_support = df['PE_OI'].idxmax() if not df.empty and not df['PE_OI'].empty else 0

    cols[0].metric("Spot Price", f"₹{current_price:,.2f}")
    cols[1].metric("Max Pain", f"₹{max_pain:,.0f}", help="The strike price where the largest number of option holders would lose money at expiry.")
    cols[2].metric("OI PCR", f"{pcr_oi:.2f}", help="Put-Call Ratio by Open Interest. > 1 is Bullish, < 0.7 is Bearish.")
    cols[3].metric("Key Support (Max Put OI)", f"₹{key_support:,.0f}")
    cols[4].metric("Key Resistance (Max Call OI)", f"₹{key_resistance:,.0f}")
    cols[5].metric("Total OI", f"{total_ce_oi + total_pe_oi:,.0f}")

def plot_oi_and_iv_charts(df, current_price):
    """Plots Open Interest and Implied Volatility charts."""
    if df.empty:
        st.warning("Cannot plot charts as the data frame is empty.")
        return
        
    st.markdown("---")
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=df.index, y=df['CE_OI'], name='Call OI', marker_color='rgba(239, 83, 80, 0.8)'))
    fig_oi.add_trace(go.Bar(x=df.index, y=df['PE_OI'], name='Put OI', marker_color='rgba(38, 166, 154, 0.8)'))
    
    max_pain = calculate_max_pain(df)
    fig_oi.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="orange", annotation_text=f"Spot: {current_price:,.0f}", annotation_position="top left")
    if max_pain > 0:
        fig_oi.add_vline(x=max_pain, line_width=2, line_dash="dot", line_color="blue", annotation_text=f"Max Pain: {max_pain:,.0f}", annotation_position="top right")
    
    fig_oi.update_layout(title="<b>Open Interest Analysis</b>", barmode='group',
                          xaxis_title="Strike Price", yaxis_title="Open Interest",
                          hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_oi, use_container_width=True)
    
    st.markdown("---")
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(x=df.index, y=df['CE_IV'], name='Call IV', mode='lines+markers', line=dict(color='rgba(239, 83, 80, 0.8)')))
    fig_iv.add_trace(go.Scatter(x=df.index, y=df['PE_IV'], name='Put IV', mode='lines+markers', line=dict(color='rgba(38, 166, 154, 0.8)')))
    
    fig_iv.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="orange", annotation_text=f"Spot: {current_price:,.0f}", annotation_position="bottom right")
    
    fig_iv.update_layout(title="<b>Implied Volatility (IV) Smile/Skew</b>",
                          xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
                          hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_iv, use_container_width=True)

def show_trading_insights(df, current_price):
    """Displays actionable trading insights based on processed data."""
    st.markdown("#### 💡 Actionable Trading Insights")
    if df.empty:
        st.warning("No data available to generate insights.")
        return

    pcr = df['PE_OI'].sum() / df['CE_OI'].sum() if df['CE_OI'].sum() > 0 else 0
    max_pain = calculate_max_pain(df)
    key_resistance = df['CE_OI'].idxmax() if not df['CE_OI'].empty else 0
    key_support = df['PE_OI'].idxmax() if not df['PE_OI'].empty else 0

    if pcr > 1.2: sentiment, emoji = "Bullish", "🟢"
    elif pcr < 0.7: sentiment, emoji = "Bearish", "🔴"
    else: sentiment, emoji = "Neutral / Range-bound", "🟡"
    
    atm_strike_row = df.iloc[(df.index.to_series() - current_price).abs().argsort()]
    if not atm_strike_row.empty:
        avg_atm_iv = (atm_strike_row['CE_IV'].iloc[0] + atm_strike_row['PE_IV'].iloc[0]) / 2
        if avg_atm_iv > 25: iv_condition = "High - Option premiums are expensive. Favorable for sellers."
        elif avg_atm_iv < 12: iv_condition = "Low - Option premiums are cheap. Favorable for buyers."
        else: iv_condition = "Moderate - Option premiums are fairly priced."
    else:
        avg_atm_iv = 0
        iv_condition = "N/A"

    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Sentiment:** {emoji} {sentiment} (PCR: {pcr:.2f})")
        st.markdown(f"**IV Environment:** {iv_condition} (ATM IV: {avg_atm_iv:.2f}%)")
    with cols[1]:
        st.markdown(f"**Key Support:** ₹{key_support:,.0f}")
        st.markdown(f"**Key Resistance:** ₹{key_resistance:,.0f}")
        st.markdown(f"**Expiry Bias (Max Pain):** Tends towards ₹{max_pain:,.0f}")

def find_strategic_options(df, current_price):
    """Identifies potentially good options to buy based on a scoring model."""
    st.markdown("#### 🎯 Strategy Dashboard: Best Options to Buy")
    st.info("For positional buyers. Ranks OTM options based on a balance of Delta (direction), Theta (low decay), and IV (cost).")
    if df.empty:
        st.warning("No data available to find strategic options.")
        return

    otm_calls = df[(df.index > current_price) & (df.index < current_price * 1.05) & (df['CE_OI'] > 0)]
    otm_puts = df[(df.index < current_price) & (df.index > current_price * 0.95) & (df['PE_OI'] > 0)]

    def calculate_scores(df_in, opt_type):
        if df_in.empty: return pd.DataFrame()
        df = df_in.copy()
        prefix = f'{opt_type}_'
        delta_max = df[f'{prefix}Delta'].abs().max()
        theta_max = df[f'{prefix}Theta'].abs().max()
        iv_max = df[f'{prefix}IV'].max()
        
        delta_score = df[f'{prefix}Delta'].abs() / delta_max if delta_max > 0 else 0
        theta_score = 1 - (df[f'{prefix}Theta'].abs() / theta_max) if theta_max > 0 else 0
        iv_score = 1 - (df[f'{prefix}IV'] / iv_max) if iv_max > 0 else 0
        
        df['Score'] = 0.3 * delta_score + 0.5 * theta_score + 0.2 * iv_score
        return df.sort_values('Score', ascending=False)

    top_calls = calculate_scores(otm_calls, 'CE')
    top_puts = calculate_scores(otm_puts, 'PE')

    call_display_cols = ['CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'Score']
    put_display_cols = ['PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'Score']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 3 Calls to Consider**")
        if not top_calls.empty:
            st.dataframe(top_calls[call_display_cols].head(3).style.format({
                'CE_LTP': '₹{:,.2f}', 'CE_IV': '{:.2f}%', 'CE_Delta': '{:.2f}',
                'CE_Theta': '₹{:,.2f}', 'Score': '{:.2f}'
            }), use_container_width=True)
        else:
            st.write("No suitable calls found.")
            
    with col2:
        st.markdown("**Top 3 Puts to Consider**")
        if not top_puts.empty:
            st.dataframe(top_puts[put_display_cols].head(3).style.format({
                'PE_LTP': '₹{:,.2f}', 'PE_IV': '{:.2f}%', 'PE_Delta': '{:.2f}',
                'PE_Theta': '₹{:,.2f}', 'Score': '{:.2f}'
            }), use_container_width=True)
        else:
            st.write("No suitable puts found.")

# --- MAIN APP LAYOUT ---
st.title("🎯 Advanced Index Options Tool")
st.markdown("*A data-driven dashboard for positional option traders based on the Max Pain theory.*")

header_cols = st.columns([2, 4, 1.2])
selected_index = header_cols[0].selectbox("Select Index", list(INDEX_SYMBOLS.keys()))
if header_cols[2].button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner(f"Fetching option data for {selected_index}..."):
    raw_data, source, underlying_price = fetch_option_data(selected_index)

if source == "Error" or raw_data is None or underlying_price is None:
    st.error("Could not fetch option data from any source. The APIs may be down or the symbol is unavailable. Please try again later.")
    st.stop()

if source == "NSE":
    st.markdown(f'<div class="success-message">✅ Loaded live data from NSE at {time.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="warning-message">⚠️ Loaded fallback data from yfinance at {time.strftime("%H:%M:%S")} (Live NSE data was unavailable)</div>', unsafe_allow_html=True)

if source == "NSE":
    expiry_dates = raw_data.get('records', {}).get('expiryDates', [])
    expiry_format, display_format = '%d-%b-%Y', '%d-%b-%Y'
else: # yfinance
    expiry_dates = raw_data.options
    expiry_format, display_format = '%Y-%m-%d', '%d-%b-%Y'

if not expiry_dates:
    st.warning("No expiry dates found for this index.")
    st.stop()
    
display_expiries = [datetime.strptime(d, expiry_format).strftime(display_format) for d in expiry_dates]
raw_to_display_map = dict(zip(expiry_dates, display_expiries))

selected_display_expiry = header_cols[1].selectbox("Select Expiry Date", display_expiries)
selected_raw_expiry = [d for d, display_d in raw_to_display_map.items() if display_d == selected_display_expiry][0]

if source == "NSE":
    # option-chain-v3 serves a single expiry per request, so refetch for the
    # user's selection (cached per index+expiry).
    raw_data, source, underlying_price = fetch_option_data(selected_index, selected_raw_expiry)
    if source != "NSE" or raw_data is None or underlying_price is None:
        st.error(f"Could not fetch option data for expiry {selected_display_expiry}. Please try another expiry.")
        st.stop()
    df = process_nse_data(raw_data, selected_raw_expiry, underlying_price)
else: # yfinance
    chain = yf.Ticker(INDEX_SYMBOLS[selected_index]["yf"]).option_chain(selected_raw_expiry)
    df = process_yfinance_data(chain, selected_raw_expiry, underlying_price)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎛️ Strategy & Greeks", "📚 Trading Tips"])

with tab1:
    create_summary_metrics(df, underlying_price)
    plot_oi_and_iv_charts(df, underlying_price)

with tab2:
    show_trading_insights(df, underlying_price)
    st.markdown("---")
    find_strategic_options(df, underlying_price)
    st.markdown("---")
    with st.expander("🔬 View Full Option Chain Data with Greeks", expanded=False):
        st.dataframe(df.style.format("{:,.2f}"), use_container_width=True)
    
with tab3:
    st.markdown("""
    ### 📘 Positional Weekly Options Strategy Guide

    This tool is designed to support a specific strategy: entering a positional trade on a **Tuesday or Wednesday** for the **next week's expiry**, primarily banking on the **Max Pain Theory**.

    **✅ Entry Checklist (Tuesday/Wednesday):**
    1.  **Select Next Week's Expiry:** The tool defaults to this. Confirm it's the date you want.
    2.  **Check Market Sentiment (Dashboard Tab):** Is the PCR bullish (>1.2) or bearish (<0.7)? This sets your initial directional bias.
    3.  **Analyze IV Environment (Strategy Tab):**
        - **Low IV (<12%)**: Favorable for buying options (they are cheaper).
        - **High IV (>25%)**: Options are expensive. Be cautious; consider strategies that benefit from falling IV (e.g., credit spreads).
    4.  **Identify Key Levels (Dashboard Tab):**
        - **Max Pain**: This is the theoretical price where option writers (sellers) are most profitable. The market may gravitate towards this level by expiry.
        - **Key Support/Resistance**: These are strong levels indicated by high Put and Call OI, respectively.
    5.  **Consult the Strategy Dashboard (Strategy Tab):**
        - If you are bullish and the spot price is below the Max Pain level, consider the "Top Calls".
        - If you are bearish and the spot price is above the Max Pain level, consider the "Top Puts".
        - This dashboard helps you find options with a good balance of directional exposure (Delta) and low time decay (Theta).
    6.  **Review the Greeks (Full Data Table):**
        - **Delta:** How much the option price moves for a ₹1 move in the index.
        - **Theta:** How much value the option loses per day. **This is your primary cost as a positional buyer.**
        - **Vega:** Sensitivity to Implied Volatility.

    **🚨 Risk Management:**
    - **Stop-Loss:** Always set a stop-loss on your premium, typically 25-30%.
    - **Time Decay (Theta):** This strategy avoids the last few days of the expiry week when Theta decay is most rapid.
    - **Profit Target:** Aim for a risk-reward ratio of at least 1:2.
    """)