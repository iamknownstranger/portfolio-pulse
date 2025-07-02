import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import time
from datetime import datetime
import numpy as np
from scipy.stats import norm
import random
from retry import retry

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Option Chain Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .success-message {
        background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724;
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
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- CONSTANTS ---
INDEX_SYMBOLS = {
    "NIFTY": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "FINNIFTY": "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "SENSEX": "SENSEX"
}
RISK_FREE_RATE = 0.07  # Assumed risk-free rate for India (7%)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# --- CORE DATA FETCHING & PROCESSING ---

@st.cache_data(ttl=60)
def get_current_price(symbol):
    """Fetches the current price of an index using yfinance."""
    ticker_map = {
        "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN", "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    }
    ticker = ticker_map.get(symbol)
    if not ticker: return None
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1m")
        return data['Close'].iloc[-1] if not data.empty else None
    except Exception:
        return None

@st.cache_data(ttl=300)
@retry(requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
def fetch_nse_option_chain(symbol):
    """Fetches option chain data from NSE's API with robust error handling and proper session priming."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    try:
        session = requests.Session()
        # Set all required headers
        session.headers.update(HEADERS)
        session.headers.update({
            "Referer": "https://www.nseindia.com/option-chain",
            "Accept-Language": "en-US,en;q=0.9",
        })
        # Prime the session and get cookies
        home = session.get("https://www.nseindia.com", timeout=10)
        time.sleep(random.uniform(1, 2))
        # Now request the option chain
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        st.session_state.data_loaded = True
        st.session_state.error_messages = []
        return data.get('records', {}), data.get('filtered', {})
    except requests.exceptions.RequestException as e:
        st.session_state.data_loaded = False
        st.session_state.error_messages.append(f"NSE API Error: {e}")
        return None, None

def get_expiry_types(expiry_dates):
    """Identifies and sorts weekly and monthly expiry dates."""
    weekly, monthly = [], []
    today = pd.Timestamp.now().normalize()
    for d_str in expiry_dates:
        try:
            dt = pd.to_datetime(d_str, format='%d-%b-%Y')
            # Check if it's the last Thursday of its month for monthly expiry
            last_day_of_month = dt + pd.offsets.MonthEnd(0)
            last_thursday = last_day_of_month - pd.DateOffset(days=(last_day_of_month.weekday() - 3) % 7)
            if dt == last_thursday:
                monthly.append(d_str)
            else:
                weekly.append(d_str)
        except ValueError:
            continue
    return sorted(weekly, key=lambda d: datetime.strptime(d, '%d-%b-%Y')), \
           sorted(monthly, key=lambda d: datetime.strptime(d, '%d-%b-%Y'))

def calculate_greeks(S, K, T, r, iv, option_type='call'):
    """Calculates option Greeks using the Black-Scholes model."""
    if T <= 0 or iv <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    vega = S * pdf_d1 * np.sqrt(T) / 100  # Vega per 1% change in IV

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else: # put
        delta = norm.cdf(d1) - 1
        theta = (- (S * pdf_d1 * iv) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    gamma = pdf_d1 / (S * iv * np.sqrt(T))
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def process_option_chain(records, selected_expiry, current_price, expiry_date):
    """Processes raw option chain data into a clean DataFrame and calculates Greeks."""
    options = []
    today = datetime.now()
    expiry_dt = datetime.strptime(expiry_date, '%d-%b-%Y')
    time_to_expiry = (expiry_dt - today + timedelta(hours=8)).days / 365.0 # Add buffer for trading hours
    if time_to_expiry < 0: time_to_expiry = 0

    for record in records:
        if record.get("expiryDate") != selected_expiry:
            continue
        
        strike_price = record.get('strikePrice')
        if not strike_price: continue

        for opt_type in ['CE', 'PE']:
            option_data = record.get(opt_type)
            if not option_data or 'strikePrice' not in option_data: continue
            
            iv = option_data.get('impliedVolatility', 0) / 100
            
            greeks = calculate_greeks(
                S=current_price, K=strike_price, T=time_to_expiry, r=RISK_FREE_RATE,
                iv=iv, option_type='call' if opt_type == 'CE' else 'put'
            )
            
            options.append({
                'Type': opt_type, 'Strike': strike_price,
                'LTP': option_data.get('lastPrice', 0),
                'IV': iv * 100, 'OI': option_data.get('openInterest', 0),
                'Chg_OI': option_data.get('changeinOpenInterest', 0),
                'Volume': option_data.get('totalTradedVolume', 0),
                'Delta': greeks['delta'], 'Theta': greeks['theta'], 'Vega': greeks['vega']
            })

    if not options: return pd.DataFrame()
    df = pd.DataFrame(options)
    
    # Merge CE and PE data side-by-side
    df_calls = df[df['Type'] == 'CE'].set_index('Strike').add_prefix('CE_')
    df_puts = df[df['Type'] == 'PE'].set_index('Strike').add_prefix('PE_')
    
    full_df = pd.concat([df_calls, df_puts], axis=1).sort_index()
    full_df.columns.name = selected_expiry
    return full_df.fillna(0)


# --- UI & ANALYSIS COMPONENTS ---

def create_summary_metrics(df, current_price):
    """Displays key summary metrics in metric cards."""
    st.markdown("#### 📈 Market Snapshot")
    cols = st.columns(5)
    
    total_ce_oi = df['CE_OI'].sum()
    total_pe_oi = df['PE_OI'].sum()
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    max_ce_strike = df['CE_OI'].idxmax() if not df['CE_OI'].empty else 0
    max_pe_strike = df['PE_OI'].idxmax() if not df['PE_OI'].empty else 0
    
    with cols[0]:
        st.metric("Spot Price", f"₹{current_price:,.2f}")
    with cols[1]:
        st.metric("Total Call OI", f"{total_ce_oi:,.0f}")
    with cols[2]:
        st.metric("Total Put OI", f"{total_pe_oi:,.0f}")
    with cols[3]:
        st.metric("OI PCR", f"{pcr_oi:.2f}", help="Put-Call Ratio by Open Interest. > 1 is Bullish, < 0.7 is Bearish.")
    with cols[4]:
        st.metric("Key Levels (R/S)", f"{max_ce_strike:,}/{max_pe_strike:,}", help="Resistance (Max Call OI) / Support (Max Put OI)")

def calculate_max_pain(df):
    """Calculates the Max Pain strike price."""
    strikes = df.index.values
    ce_oi = df['CE_OI'].values
    pe_oi = df['PE_OI'].values
    
    total_loss = []
    for expiry_price in strikes:
        call_loss = np.where(strikes < expiry_price, (expiry_price - strikes) * ce_oi, 0).sum()
        put_loss = np.where(strikes > expiry_price, (strikes - expiry_price) * pe_oi, 0).sum()
        total_loss.append(call_loss + put_loss)
        
    min_loss_idx = np.argmin(total_loss)
    return strikes[min_loss_idx]

def plot_oi_and_iv_charts(df, current_price):
    """Plots Open Interest and Implied Volatility charts."""
    # --- OI Chart ---
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(x=df.index, y=df['CE_OI'], name='Call OI', marker_color='rgba(239, 83, 80, 0.8)'))
    fig_oi.add_trace(go.Bar(x=df.index, y=df['PE_OI'], name='Put OI', marker_color='rgba(38, 166, 154, 0.8)'))
    
    max_pain = calculate_max_pain(df)
    fig_oi.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="orange",
                     annotation_text=f"Spot: {current_price:,.0f}", annotation_position="top left")
    fig_oi.add_vline(x=max_pain, line_width=2, line_dash="dot", line_color="blue",
                     annotation_text=f"Max Pain: {max_pain:,.0f}", annotation_position="top right")

    fig_oi.update_layout(title="<b>Open Interest Analysis</b>", barmode='group',
                          xaxis_title="Strike Price", yaxis_title="Open Interest",
                          hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_oi, use_container_width=True)

    # --- IV Chart ---
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(x=df.index, y=df['CE_IV'], name='Call IV', mode='lines+markers', line=dict(color='rgba(239, 83, 80, 0.8)')))
    fig_iv.add_trace(go.Scatter(x=df.index, y=df['PE_IV'], name='Put IV', mode='lines+markers', line=dict(color='rgba(38, 166, 154, 0.8)')))
    
    fig_iv.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="orange",
                     annotation_text=f"Spot: {current_price:,.0f}", annotation_position="bottom right")

    fig_iv.update_layout(title="<b>Implied Volatility (IV) Smile/Skew</b>",
                          xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
                          hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_iv, use_container_width=True)

def show_trading_insights(df, current_price):
    """Displays actionable trading insights based on processed data."""
    st.markdown("#### 💡 Actionable Trading Insights")
    
    pcr = df['PE_OI'].sum() / df['CE_OI'].sum() if df['CE_OI'].sum() > 0 else 0
    max_pain = calculate_max_pain(df)
    key_resistance = df['CE_OI'].idxmax()
    key_support = df['PE_OI'].idxmax()

    # Sentiment based on PCR
    if pcr > 1.2: sentiment, emoji = "Bullish", "🟢"
    elif pcr < 0.7: sentiment, emoji = "Bearish", "🔴"
    else: sentiment, emoji = "Neutral / Range-bound", "🟡"
    
    # IV Analysis - finding ATM IV
    atm_strike_ce = df.iloc[(df.index - current_price).abs().argsort()]['CE_IV'].iloc[0]
    atm_strike_pe = df.iloc[(df.index - current_price).abs().argsort()]['PE_IV'].iloc[0]
    avg_atm_iv = (atm_strike_ce + atm_strike_pe) / 2
    
    if avg_atm_iv > 25: iv_condition = "High - Option premiums are expensive. Favorable for sellers."
    elif avg_atm_iv < 12: iv_condition = "Low - Option premiums are cheap. Favorable for buyers."
    else: iv_condition = "Moderate - Option premiums are fairly priced."

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

    # Filter for OTM options within a reasonable range
    otm_calls = df[(df.index > current_price) & (df.index < current_price * 1.05) & (df['CE_OI'] > 0)]
    otm_puts = df[(df.index < current_price) & (df.index > current_price * 0.95) & (df['PE_OI'] > 0)]

    # Scoring: Higher delta is good, lower (less negative) theta is good, lower IV is good.
    # We normalize to make them comparable.
    def calculate_scores(df, opt_type):
        if df.empty: return df
        prefix = f'{opt_type}_'
        delta_score = df[f'{prefix}Delta'] / df[f'{prefix}Delta'].max()
        theta_score = 1 - (df[f'{prefix}Theta'].abs() / df[f'{prefix}Theta'].abs().max())
        iv_score = 1 - (df[f'{prefix}IV'] / df[f'{prefix}IV'].max())
        # Weights: Emphasize Theta for positional plays
        df['Score'] = 0.3 * delta_score + 0.5 * theta_score + 0.2 * iv_score
        return df.sort_values('Score', ascending=False)

    top_calls = calculate_scores(otm_calls, 'CE')
    top_puts = calculate_scores(otm_puts, 'PE')
    
    display_cols = ['CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'Score']
    put_display_cols = ['PE_LTP', 'PE_IV', 'PE_Delta', 'PE_Theta', 'Score']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 3 Calls to Consider**")
        st.dataframe(top_calls[display_cols].head(3).style.format({
            'CE_LTP': '₹{:,.2f}', 'CE_IV': '{:.2f}%', 'CE_Delta': '{:.2f}',
            'CE_Theta': '₹{:,.2f}', 'Score': '{:.2f}'
        }).background_gradient(cmap='Greens', subset=['Score']), use_container_width=True)
    with col2:
        st.markdown("**Top 3 Puts to Consider**")
        st.dataframe(top_puts[put_display_cols].head(3).style.format({
            'PE_LTP': '₹{:,.2f}', 'PE_IV': '{:.2f}%', 'PE_Delta': '{:.2f}',
            'PE_Theta': '₹{:,.2f}', 'Score': '{:.2f}'
        }).background_gradient(cmap='Reds', subset=['Score']), use_container_width=True)

# --- MAIN APP LAYOUT ---
st.title("🎯 Advanced Index Options Tool")
st.markdown("*A data-driven dashboard for positional option traders.*")

# --- CONTROLS ---
header_cols = st.columns([2, 2, 4, 1.2])
with header_cols[0]:
    selected_index = st.selectbox("Select Index", list(INDEX_SYMBOLS.keys()))
with header_cols[1]:
    expiry_type = st.radio("Expiry Type", ["Weekly", "Monthly"], index=0, horizontal=True)
with header_cols[3]:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# --- DATA FETCHING & VALIDATION ---
with st.spinner(f"Fetching {selected_index} option chain from NSE..."):
    records, filtered_records = fetch_nse_option_chain(INDEX_SYMBOLS[selected_index])

if not st.session_state.data_loaded or not records:
    for msg in st.session_state.error_messages:
        st.error(f"❌ {msg}")
    st.warning("Could not fetch data. The NSE server might be busy. Please try refreshing in a moment.")
    st.stop()

st.success(f"✅ Data for {selected_index} loaded successfully at {time.strftime('%H:%M:%S')}")

current_price = filtered_records.get('underlyingValue') if filtered_records else get_current_price(selected_index)
if current_price is None:
    st.error("Could not determine the current price. Analysis will be limited.")
    st.stop()

# --- EXPIRY SELECTION ---
expiry_dates = records.get('expiryDates', [])
weekly_expiries, monthly_expiries = get_expiry_types(expiry_dates)
available_expiries = weekly_expiries if expiry_type == "Weekly" else monthly_expiries

if not available_expiries:
    st.warning(f"No {expiry_type.lower()} expiries found for {selected_index}.")
    st.stop()
    
# Logic for default selection based on user's strategy
# Default to next week's expiry if available, otherwise the first available
today = datetime.now()
next_tuesday = today + timedelta(days=(1 - today.weekday() + 7) % 7)
default_selection = available_expiries[0]
for expiry_str in available_expiries:
    expiry_dt = datetime.strptime(expiry_str, '%d-%b-%Y')
    if expiry_dt.date() > next_tuesday.date():
        default_selection = expiry_str
        break

with header_cols[2]:
    selected_expiry = st.selectbox("Select Expiry Date", available_expiries, index=available_expiries.index(default_selection))

# --- DATA PROCESSING & DISPLAY ---
df = process_option_chain(records['data'], selected_expiry, current_price, selected_expiry)

if df.empty:
    st.error(f"No option data processed for {selected_expiry}. Please select another date.")
    st.stop()

# --- ANALYSIS TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎛️ Strategy & Greeks", "📚 Trading Tips"])

with tab1:
    create_summary_metrics(df, current_price)
    st.markdown("---")
    plot_oi_and_iv_charts(df, current_price)
    
with tab2:
    show_trading_insights(df, current_price)
    st.markdown("---")
    find_strategic_options(df, current_price)
    st.markdown("---")
    
    with st.expander("🔬 View Full Option Chain Data with Greeks", expanded=False):
        # Clean up for display
        display_df = df.copy()
        display_df.index.name = "Strike"
        # Rounding for cleaner view
        for col in display_df.columns:
            if 'Delta' in col or 'Theta' in col or 'Vega' in col:
                display_df[col] = display_df[col].round(3)
            if 'IV' in col:
                display_df[col] = display_df[col].round(2)
        st.dataframe(display_df, use_container_width=True)

with tab3:
    st.markdown("""
    ### 📘 Positional Weekly Options Strategy Guide

    This tool is designed to support a specific strategy: entering a positional trade on a **Tuesday or Wednesday** for the **next week's expiry**.

    **✅ Entry Checklist (Tuesday/Wednesday):**
    1.  **Select Next Week's Expiry:** Use the dropdown to choose the correct date.
    2.  **Check Market Sentiment (Dashboard Tab):** Is the PCR bullish (>1.2) or bearish (<0.7)? This sets your initial bias.
    3.  **Analyze IV Environment (Strategy Tab):**
        - **Low IV (<12%)**: Good for buying options (they are cheaper).
        - **High IV (>25%)**: Options are expensive. Be cautious with buying; consider strategies that benefit from falling IV.
    4.  **Identify Key Levels (Dashboard Tab):** Note the major support (max Put OI) and resistance (max Call OI) levels. These are magnets and potential turning points.
    5.  **Consult the Strategy Dashboard (Strategy Tab):**
        - If you are bullish, review the "Top Calls to Consider".
        - If you are bearish, review the "Top Puts to Consider".
        - These suggestions balance direction (Delta) with time decay (Theta), which is critical for positional holds.
    6.  **Review the Greeks (Full Data Table):**
        - **Delta:** How much the option price will move for a ₹1 move in the index. Higher is more directional.
        - **Theta:** How much value the option loses per day. **This is your primary cost as a positional buyer.** Look for lower (less negative) Theta values.
        - **Vega:** Sensitivity to IV. If you expect volatility to rise, a high Vega is good.

    **🚨 Risk Management:**
    - **Stop-Loss:** A crucial rule is to set a stop-loss on your premium, typically 25-30%.
    - **Time Decay (Theta):** Theta accelerates massively in the last 2-3 days before expiry. This strategy avoids holding during that window by targeting the *next* week's expiry.
    - **Profit Target:** Aim for a risk-reward ratio of at least 1:2. If your stop-loss is 10 points, your first target should be at least 20 points.
    """)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("---")    
st.sidebar.markdown("""
    <style>
    /* Ensure the sidebar uses full viewport height */
    .css-1d391kg, .sidebar .sidebar-content { 
        display: flex; 
        flex-direction: column; 
        height: 100vh; 
    }
    /* Let the main content grow, forcing the footer to the bottom */
    .css-1d391kg > div:nth-child(2), .sidebar .sidebar-content > div:first-child { 
        flex: 1;
    }
    .sidebar-footer {
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-footer">Built with 💓 by Chandra Sekhar Mullu</div>', unsafe_allow_html=True)

