import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from retry import retry

st.set_page_config(layout="wide")
st.title("Index Option Chain Analysis")

# --- Constants ---
INDEX_SYMBOLS = {
    "NIFTY": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "FINNIFTY": "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "SENSEX": "SENSEX"
}
OPTION_CHAIN_URL_TEMPLATE = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# --- Functions ---
@st.cache_data(ttl=300)
@retry((requests.exceptions.RequestException, Exception), tries=3, delay=5, backoff=2)
def fetch_option_chain(symbol, max_retries=3, retry_delay=5):
    url = OPTION_CHAIN_URL_TEMPLATE.format(symbol=symbol)
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
    response = session.get(url, headers=HEADERS, timeout=10)
    if response.status_code == 429:
        # Only show warning on first try
        if not hasattr(fetch_option_chain, '_retrying'):
            st.warning(f"Rate limited by NSE (429). Retrying in {retry_delay} seconds...")
            fetch_option_chain._retrying = True
        raise requests.exceptions.RequestException("429 Too Many Requests")
    if response.status_code == 401:
        if not hasattr(fetch_option_chain, '_retrying'):
            st.warning(f"Unauthorized (401) from NSE. Retrying in {retry_delay} seconds...")
            fetch_option_chain._retrying = True
        raise requests.exceptions.RequestException("401 Unauthorized")
    # Clear the retrying flag if successful
    if hasattr(fetch_option_chain, '_retrying'):
        del fetch_option_chain._retrying
    if response.status_code != 200:
        st.error(f"Failed to fetch data from NSE. Status code: {response.status_code}")
        return None, []
    try:
        data = response.json()
    except Exception:
        st.error("Failed to parse option chain data. The NSE website may be blocking requests or is temporarily unavailable.")
        return None, []
    records = data.get('records', {}).get('data', [])
    expiry_dates = data.get('records', {}).get('expiryDates', [])
    return records, expiry_dates
 


def process_option_chain(records, selected_expiry):
    calls, puts = [], []
    for record in records:
        if record.get("expiryDate") == selected_expiry:
            strike = record['strikePrice']
            ce = record.get("CE")
            pe = record.get("PE")
            if ce:
                calls.append({"strike": strike, **ce})
            if pe:
                puts.append({"strike": strike, **pe})
    df_calls = pd.DataFrame(calls).set_index("strike")
    df_puts = pd.DataFrame(puts).set_index("strike")
    df = pd.concat([df_calls.add_prefix("CE_"), df_puts.add_prefix("PE_")], axis=1)
    return df.sort_index()


def plot_oi_bar_chart(df):
    with st.container():
        col1, col2 = st.columns([2.5, 1], vertical_alignment="center")
        with col1:
            st.markdown("#### 📊 Open Interest by Strike Price")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df.index, y=df['CE_openInterest'], name='CE OI', marker_color='red'))
            fig.add_trace(go.Bar(x=df.index, y=df['PE_openInterest'], name='PE OI', marker_color='green'))
            fig.update_layout(title="Open Interest by Strike Price", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("- **Call OI (red)**: May indicate resistance levels.\n"
                        "- **Put OI (green)**: May indicate support levels.\n"
                        "- Rising OI = stronger conviction\n"
                        "- Unwinding OI = exit or reversal zone")


def plot_pcr(df):
    with st.container():
        col1, col2 = st.columns([2.5, 1], vertical_alignment="center")
        with col1:
            st.markdown("#### ⚖️ Put-Call Ratio (PCR)")
            df['PCR'] = df['PE_openInterest'] / df['CE_openInterest']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['PCR'], mode='lines+markers', name='PCR'))
            fig.update_layout(title="Put-Call Ratio by Strike")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("- PCR > 1: Bullish bias\n"
                        "- PCR < 1: Bearish bias\n"
                        "- Extreme values can indicate overbought/oversold sentiment")


def calculate_max_pain(df):
    pain = {}
    for strike in df.index:
        pain[strike] = ((df['CE_openInterest'] * abs(df.index - strike)).sum() +
                        (df['PE_openInterest'] * abs(df.index - strike)).sum())
    min_pain = min(pain, key=pain.get)
    return min_pain


def show_support_resistance(df):
    st.markdown("#### 🛡️ Support & Resistance Zones from OI (Top 5)")
    st.markdown("These help in identifying range, breakout zones, or safe entry levels.")
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        st.markdown("### Resistance (Call OI)")
        top_ce = df['CE_openInterest'].nlargest(5)
        st.dataframe(top_ce.to_frame().style.apply(
            lambda row: ['background-color: #ffb6b6; font-weight: bold']*len(row) if row.name == top_ce.index[0] else ['background-color: #ffe4e1']*len(row),
            axis=1
        ), use_container_width=True)
    with col2:
        st.markdown("### Support (Put OI)")
        top_pe = df['PE_openInterest'].nlargest(5)
        st.dataframe(top_pe.to_frame().style.apply(
            lambda row: ['background-color: #90ee90; font-weight: bold']*len(row) if row.name == top_pe.index[0] else ['background-color: #eaffea']*len(row),
            axis=1
        ), use_container_width=True)


def plot_oi_change_highlight(df):
    st.markdown("#### 🔥 Top 5 OI Changes (Build-up/Unwinding)")
    ce_oi_chg = df['CE_changeinOpenInterest'].sort_values(ascending=False)
    pe_oi_chg = df['PE_changeinOpenInterest'].sort_values(ascending=False)
    # Get top 5 for each
    ce_bu = ce_oi_chg.head(5)
    ce_uw = ce_oi_chg.tail(5)
    pe_bu = pe_oi_chg.head(5)
    pe_uw = pe_oi_chg.tail(5)
    # Find common strike prices
    bu_common = set(ce_bu.index) & set(pe_bu.index)
    uw_common = set(ce_uw.index) & set(pe_uw.index)
    def highlight_row(row, common, color_bu, color_uw):
        if row.name in common:
            return [f'background-color: {color_bu if row[0] >= 0 else color_uw}; font-weight: bold']
        return ['']
    col_bu, col_uw = st.columns(2)
    with col_bu:
        st.markdown("**Top 5 CE OI Build-up:** (Buyers: green, Common: dark green)")
        st.dataframe(
            ce_bu.to_frame().style.apply(
                highlight_row, common=bu_common, color_bu='#228B22', color_uw='#90ee90', axis=1
            ).applymap(lambda v: 'background-color: #90ee90' if v >= 0 else ''),
            use_container_width=True
        )
        st.markdown("**Top 5 PE OI Unwinding:** (Sellers: red, Common: dark red)")
        st.dataframe(
            pe_uw.to_frame().style.apply(
                highlight_row, common=uw_common, color_bu='#8B0000', color_uw='#ffb6b6', axis=1
            ).applymap(lambda v: 'background-color: #ffb6b6' if v < 0 else ''),
            use_container_width=True
        )
    with col_uw:
        st.markdown("**Top 5 PE OI Build-up:** (Buyers: green, Common: dark green)")
        st.dataframe(
            pe_bu.to_frame().style.apply(
                highlight_row, common=bu_common, color_bu='#228B22', color_uw='#90ee90', axis=1
            ).applymap(lambda v: 'background-color: #90ee90' if v >= 0 else ''),
            use_container_width=True
        )
        st.markdown("**Top 5 CE OI Unwinding:** (Sellers: red, Common: dark red)")
        st.dataframe(
            ce_uw.to_frame().style.apply(
                highlight_row, common=uw_common, color_bu='#8B0000', color_uw='#ffb6b6', axis=1
            ).applymap(lambda v: 'background-color: #ffb6b6' if v < 0 else ''),
            use_container_width=True
        )


def plot_oi_heatmap(df):
    st.markdown("#### 🗺️ Option Chain OI & Change Heatmap")
    import plotly.express as px
    heatmap_df = df[[
        'CE_openInterest', 'CE_changeinOpenInterest',
        'PE_openInterest', 'PE_changeinOpenInterest']].copy()
    heatmap_df = heatmap_df.reset_index().melt(id_vars='strike')
    fig = px.imshow(
        heatmap_df.pivot(index='variable', columns='strike', values='value'),
        aspect='auto', color_continuous_scale='Viridis',
        labels=dict(x="Strike", y="Metric", color="Value")
    )
    st.plotly_chart(fig, use_container_width=True)


def highlight_iv_spike(df):
    st.markdown("#### ⚡ IV Crush/Spike Detection")
    ce_iv = df['CE_impliedVolatility']
    pe_iv = df['PE_impliedVolatility']
    ce_iv_spike = ce_iv[ce_iv > ce_iv.mean() + ce_iv.std()]
    pe_iv_spike = pe_iv[pe_iv > pe_iv.mean() + pe_iv.std()]
    ce_iv_crush = ce_iv[ce_iv < ce_iv.mean() - ce_iv.std()]
    pe_iv_crush = pe_iv[pe_iv < pe_iv.mean() - pe_iv.std()]
    st.write(f"**CE IV Spikes:** {ce_iv_spike.index.tolist()}")
    st.write(f"**PE IV Spikes:** {pe_iv_spike.index.tolist()}")
    st.write(f"**CE IV Crush:** {ce_iv_crush.index.tolist()}")
    st.write(f"**PE IV Crush:** {pe_iv_crush.index.tolist()}")


# --- Main Logic ---
st.markdown("This dashboard helps you track how this week's index options are positioned, giving you insights for next week's trades.")

col_sel1, col_sel2, col_sel3 = st.columns([1, 2, 2], vertical_alignment="center")
with col_sel1:
    selected_index = st.selectbox("Select Index", list(INDEX_SYMBOLS.keys()), index=0)
with col_sel2:
    expiry_type = st.radio(
        "Select Expiry Type(s)",
        ["Weekly", "Monthly", "Both (Compare)"],
        index=0,
        horizontal=True
    )

# Fetch option chain for selected index
records, expiry_dates = fetch_option_chain(INDEX_SYMBOLS[selected_index])
if records is None or not expiry_dates:
    st.stop()

# --- Helper to filter expiry dates ---
def get_expiry_types(expiry_dates):
    weekly = []
    monthly = []
    for d in expiry_dates:
        dt = pd.to_datetime(d, dayfirst=True, errors='coerce')
        if pd.isnull(dt):
            continue
        # Monthly expiry: last Thursday of the month
        last_thu = (dt + pd.offsets.MonthEnd(0)).replace(day=1) + pd.offsets.Week(weekday=3, n=4)
        if last_thu.month != dt.month:
            last_thu = last_thu - pd.offsets.Week(1)
        if dt.date() == last_thu.date():
            monthly.append(d)
        else:
            weekly.append(d)
    return weekly, monthly

weekly_expiries, monthly_expiries = get_expiry_types(expiry_dates)

# --- Expiry selection logic ---
if expiry_type == "Weekly":
    default_expiries = weekly_expiries[:1] if weekly_expiries else []
    available_expiries = weekly_expiries
elif expiry_type == "Monthly":
    default_expiries = monthly_expiries[:1] if monthly_expiries else []
    available_expiries = monthly_expiries
else:  # Both
    default_expiries = []
    if weekly_expiries:
        default_expiries.append(weekly_expiries[0])
    if monthly_expiries:
        default_expiries.append(monthly_expiries[0])
    available_expiries = []
    if weekly_expiries:
        available_expiries += weekly_expiries
    if monthly_expiries:
        available_expiries += monthly_expiries

with col_sel3:
    selected_expiries = st.multiselect(
        "Select up to 2 Expiry Dates to Compare",
        available_expiries,
        default=default_expiries,
        max_selections=2
    )
if len(selected_expiries) == 0:
    st.warning("Please select at least one expiry date.")
    st.stop()
if len(selected_expiries) > 2:
    st.warning("Please select at most two expiry dates.")
    st.stop()

# Prepare dataframes for selected expiries
analysis = {}
for expiry in selected_expiries:
    df = process_option_chain(records, expiry)
    analysis[expiry] = df

with st.expander("🧠 How to Use the Option Chain to Predict Index Movement", expanded=False):
    st.markdown("""
1. **OI Buildup:** Heavy CE OI above price = Resistance; Heavy PE OI below = Support
2. **Change in OI:** Fresh positions = trend conviction; Unwinding = reversal/profit-booking
3. **Put/Call Ratio (PCR):** PCR > 1: Bullish; PCR < 1: Bearish
4. **Max Pain Theory:** Price tends to move toward Max Pain by expiry
5. **IV Crush/Spike:** High IV + falling price = fear; IV cooling + rising price = stability
    """)

# Show analysis side by side if two expiries selected
if len(selected_expiries) == 2:
    exp1, exp2 = selected_expiries
    st.subheader(f"📋 Option Chain Data Comparison: {exp1} vs {exp2}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {exp1}")
        st.dataframe(analysis[exp1][[
            'CE_openInterest', 'CE_changeinOpenInterest', 'CE_impliedVolatility',
            'PE_openInterest', 'PE_changeinOpenInterest', 'PE_impliedVolatility']])
        plot_oi_bar_chart(analysis[exp1])
        plot_oi_change_highlight(analysis[exp1])
        plot_pcr(analysis[exp1])
        highlight_iv_spike(analysis[exp1])
        max_pain1 = calculate_max_pain(analysis[exp1])
        st.success(f"📌 Max Pain: {max_pain1}")
        show_support_resistance(analysis[exp1])
    with col2:
        st.markdown(f"#### {exp2}")
        st.dataframe(analysis[exp2][[
            'CE_openInterest', 'CE_changeinOpenInterest', 'CE_impliedVolatility',
            'PE_openInterest', 'PE_changeinOpenInterest', 'PE_impliedVolatility']])
        plot_oi_bar_chart(analysis[exp2])
        plot_oi_change_highlight(analysis[exp2])
        plot_pcr(analysis[exp2])
        highlight_iv_spike(analysis[exp2])
        max_pain2 = calculate_max_pain(analysis[exp2])
        st.success(f"📌 Max Pain: {max_pain2}")
        show_support_resistance(analysis[exp2])
    # Move heatmap to bottom
    st.markdown("#### 🗺️ Option Chain OI & Change Heatmap")
    plot_oi_heatmap(analysis[exp1])
    if len(analysis) > 1:
        plot_oi_heatmap(analysis[exp2])
else:
    expiry = selected_expiries[0]
    df = analysis[expiry]
    st.subheader(f"📋 Option Chain Data: {expiry}")
    col1, col2 = st.columns([2.5, 1], vertical_alignment="center")
    with col1:
        st.dataframe(df[[
            'CE_openInterest', 'CE_changeinOpenInterest', 'CE_impliedVolatility',
            'PE_openInterest', 'PE_changeinOpenInterest', 'PE_impliedVolatility']])
    with col2:
        st.markdown("- **Open Interest** shows where positions are being built\n"
                    "- **Change in OI** reveals bullish/bearish additions\n"
                    "- **IV** hints at expected volatility or premium")
    plot_oi_bar_chart(df)
    plot_oi_change_highlight(df)
    plot_pcr(df)
    highlight_iv_spike(df)
    max_pain = calculate_max_pain(df)
    st.success(f"📌 Max Pain: {max_pain}")
    show_support_resistance(df)
    # Move heatmap to bottom
    st.markdown("#### 🗺️ Option Chain OI & Change Heatmap")
    plot_oi_heatmap(df)

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

