import random
from datetime import date, timedelta
import streamlit as st
from streamlit_tags import st_tags
import pandas as pd


import yfinance as yf

def get_sp500_tickers():

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    tickers = table[0]['Symbol'].tolist()

    # Some tickers may have periods (e.g., BRK.B), which yfinance uses as dashes (e.g., BRK-B)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

# Common list of suggestions (could also be imported from another module)
SP_500 = get_sp500_tickers()

def render_sidebar():
    st.sidebar.title("Portfolio Filters")
    st.sidebar.info("Provide stock tickers and select a date range to analyze your portfolio performance.")
    
    # Ensure default symbols exist in session_state
    if 'default_symbols' not in st.session_state:
        st.session_state.default_symbols = random.sample(SP_500, 7)
        
    with st.sidebar.form(key='portfolio_form'):
        symbols = st_tags(
            label='Enter the name or symbol of the stock in your portfolio',
            text='Press enter to add more',
            value=st.session_state.default_symbols,
            suggestions=SP_500
        )
        start_date_value = date.today() - timedelta(days=3*365)
        end_date_value = date.today()
        start_date = st.date_input('Start date', value=start_date_value)
        end_date = st.date_input('End date', value=end_date_value)
        period = st.radio(
            "Period", ["WTD", "MTD", "YTD", "ITD"], horizontal=True, index=3)
        # --- Enhanced benchmark index selector ---
        benchmark_options = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Top 100 US Market Cap": "TOP100US",
            "S&P 500 Technology": "^SP500-45",
            "S&P 500 Healthcare": "^SP500-35",
            "S&P 500 Financials": "^SP500-40",
            "S&P 500 Industrials": "^SP500-20",
            "S&P 500 Consumer Discretionary": "^SP500-25"
        }
        benchmark_name = st.selectbox("Benchmark Index", list(benchmark_options.keys()), index=0)
        st.form_submit_button("Analyze")

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
    # Return benchmark symbol as well
    return symbols, start_date.isoformat(), end_date.isoformat(), period, benchmark_options[benchmark_name], benchmark_name

