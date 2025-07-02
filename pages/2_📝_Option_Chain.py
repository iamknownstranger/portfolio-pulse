import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Index Option Chain Dashboard (Weekly Analysis)")

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
def fetch_option_chain(symbol):
    url = OPTION_CHAIN_URL_TEMPLATE.format(symbol=symbol)
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
        response = session.get(url, headers=HEADERS, timeout=10)
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
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None, []


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
    st.markdown("#### 🛡️ Support & Resistance Zones from OI")
    st.markdown("These help in identifying range, breakout zones, or safe entry levels.")
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        st.markdown("### Resistance (Call OI)")
        top_ce = df['CE_openInterest'].nlargest(3)
        st.write(top_ce)
    with col2:
        st.markdown("### Support (Put OI)")
        top_pe = df['PE_openInterest'].nlargest(3)
        st.write(top_pe)


def plot_oi_change_highlight(df):
    st.markdown("#### 🔥 Top OI Changes (Build-up/Unwinding)")
    ce_oi_chg = df['CE_changeinOpenInterest'].sort_values(ascending=False)
    pe_oi_chg = df['PE_changeinOpenInterest'].sort_values(ascending=False)
    st.write("**Top 3 CE OI Build-up:**")
    st.dataframe(ce_oi_chg.head(3).to_frame(), use_container_width=True)
    st.write("**Top 3 PE OI Build-up:**")
    st.dataframe(pe_oi_chg.head(3).to_frame(), use_container_width=True)
    st.write("**Top 3 CE OI Unwinding:**")
    st.dataframe(ce_oi_chg.tail(3).to_frame(), use_container_width=True)
    st.write("**Top 3 PE OI Unwinding:**")
    st.dataframe(pe_oi_chg.tail(3).to_frame(), use_container_width=True)


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
# Index selector
st.markdown("This dashboard helps you track how this week's index options are positioned, giving you insights for next week's trades.")

col_idx, col_exp = st.columns([1, 2])
with col_idx:
    selected_index = st.selectbox("Select Index", list(INDEX_SYMBOLS.keys()), index=0)

# Fetch option chain for selected index
records, expiry_dates = fetch_option_chain(INDEX_SYMBOLS[selected_index])
if records is None or not expiry_dates:
    st.stop()

with col_exp:
    selected_expiries = st.multiselect("Select up to 2 Expiry Dates to Compare", expiry_dates, default=expiry_dates[:2])
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
        plot_oi_heatmap(analysis[exp1])
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
        plot_oi_heatmap(analysis[exp2])
        plot_pcr(analysis[exp2])
        highlight_iv_spike(analysis[exp2])
        max_pain2 = calculate_max_pain(analysis[exp2])
        st.success(f"📌 Max Pain: {max_pain2}")
        show_support_resistance(analysis[exp2])
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
    plot_oi_heatmap(df)
    plot_pcr(df)
    highlight_iv_spike(df)
    max_pain = calculate_max_pain(df)
    st.success(f"📌 Max Pain: {max_pain}")
    st.markdown("- Max Pain is the strike at which the most option writers (sellers) would benefit, as most options expire worthless there.\n"
                "- Price tends to gravitate toward this level near expiry.")
    show_support_resistance(df)

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

# Educational tips and external links
with st.expander("🧠 How to Use the Option Chain to Predict Index Movement"):
    st.markdown("""
1. **OI Buildup:** Heavy CE OI above price = Resistance; Heavy PE OI below = Support
2. **Change in OI:** Fresh positions = trend conviction; Unwinding = reversal/profit-booking
3. **Put/Call Ratio (PCR):** PCR > 1: Bullish; PCR < 1: Bearish
4. **Max Pain Theory:** Price tends to move toward Max Pain by expiry
5. **IV Crush/Spike:** High IV + falling price = fear; IV cooling + rising price = stability
    """)
st.sidebar.markdown("---")
st.sidebar.markdown("**External Tools for Deeper Analysis:**\n- [Sensibull](https://sensibull.com)\n- [NiftyTrader](https://www.niftytrader.in/)\n- [Opstra](https://opstra.definedge.com/)\n- [NSE India](https://www.nseindia.com/)\n- [ChartInk](https://chartink.com/)")
