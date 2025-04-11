import random
from datetime import date, timedelta
import streamlit as st
from streamlit_tags import st_tags
import pandas as pd

# Common list of suggestions (could also be imported from another module)
SP_500 = ["MMM", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADBE", "AAP", "AMD", "AES",
            "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALXN", "ALGN", "ALLE",
            "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP",
            "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS",
            "ANTM", "AON", "AOS", "APA", "AAPL", "AMAT", "APTV", "ADM", "ANET", "AJG",
            "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "BKR", "BLL", "BAC",
            "BK", "BAX", "BDX", "BRK.B", "BBY", "BIO", "BIIB", "BLK", "BA", "BKNG",
            "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BF.B", "CHRW", "COG", "CDNS",
            "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE",
            "CBRE", "CDW", "CE", "CNC", "CNP", "CERN", "CF", "SCHW", "CHTR", "CVX",
            "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CTXS",
            "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP",
            "ED", "STZ", "COO", "CPRT", "GLW", "CTVA", "COST", "CCI", "CSX", "CMI",
            "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", "DVN", "DXCM",
            "FANG", "DLR", "DFS", "DISCA", "DISCK", "DISH", "DG", "DLTR", "D", "DPZ",
            "DOV", "DOW", "DTE", "DUK", "DRE", "DD", "DXC", "EMN", "ETN", "EBAY",
            "ECL", "EIX", "EW", "EA", "EMR", "ENPH", "ETR", "EOG", "EFX", "EQIX",
            "EQR", "ESS", "EL", "ETSY", "EVRG", "ES", "RE", "EXC", "EXPE", "EXPD",
            "EXR", "XOM", "FFIV", "FB", "FAST", "FRT", "FDX", "FIS", "FITB", "FE",
            "FRC", "FISV", "FLT", "FMC", "F", "FTNT", "FTV", "FBHS", "FOXA", "FOX",
            "BEN", "FCX", "GPS", "GRMN", "IT", "GD", "GE", "GIS", "GM", "GPC",
            "GILD", "GL", "GPN", "GS", "GWW", "HAL", "HBI", "HIG", "HAS", "HCA",
            "PEAK", "HSIC", "HSY", "HES", "HPE", "HLT", "HFC", "HOLX", "HD", "HON",
            "HRL", "HST", "HWM", "HPQ", "HUM", "HBAN", "HII", "IEX", "IDXX", "INFO",
            "ITW", "ILMN", "INCY", "IR", "INTC", "ICE", "IBM", "IP", "IPG", "IFF",
            "INTU", "ISRG", "IVZ", "IPGP", "IQV", "IRM", "JBHT", "JKHY", "J", "SJM",
            "JNJ", "JCI", "JPM", "JNPR", "KSU", "K", "KEY", "KEYS", "KMB", "KIM",
            "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LEG",
            "LDOS", "LEN", "LLY", "LNC", "LIN", "LYV", "LKQ", "LMT", "L", "LOW",
            "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS",
            "MA", "MKC", "MXIM", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM",
            "MCHP", "MU", "MSFT", "MAA", "MHK", "TAP", "MDLZ", "MPWR", "MNST", "MCO",
            "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA",
            "NWS", "NEE", "NLSN", "NKE", "NI", "NSC", "NTRS", "NOC", "NLOK", "NCLH",
            "NOV", "NRG", "NUE", "NVDA", "NVR", "ORLY", "OXY", "ODFL", "OMC", "OKE",
            "ORCL", "OTIS", "PCAR", "PKG", "PH", "PAYX", "PAYC", "PYPL", "PENN", "PNR",
            "PBCT", "PEP", "PKI", "PRGO", "PFE", "PM", "PSX", "PNW", "PXD", "PNC",
            "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PTC", "PEG",
            "PSA", "PHM", "PVH", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX",
            "O", "REG", "REGN", "RF", "RSG", "RMD", "RHI", "ROK", "ROL", "ROP",
            "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", "NOW",
            "SHW", "SPG", "SWKS", "SNA", "SO", "LUV", "SWK", "SBUX", "STT", "STE",
            "SYK", "SIVB", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TGT",
            "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO", "TJX", "TSCO",
            "TT", "TDG", "TRV", "TFC", "TWTR", "TYL", "TSN", "UDR", "ULTA", "USB",
            "UAA", "UA", "UNP", "UAL", "UNH", "UPS", "URI", "UHS", "UNM", "VFC",
            "VLO", "VAR", "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VIAC", "V", "VNO",
            "VMC", "WRB", "WAB", "WMT", "WBA", "DIS", "WM", "WAT", "WEC", "WFC",
            "WELL", "WST", "WDC", "WU", "WRK", "WY", "WHR", "WMB", "WLTW", "WYNN",
            "XEL", "XLNX", "XYL", "YUM", "ZBRA", "ZBH", "ZION", "ZTS"]


def render_sidebar():
    st.sidebar.title("Portfolio Filters")
    st.sidebar.info("Provide stock tickers and select a date range to analyze your portfolio performance.")
    
    # Ensure default symbols exist in session_state
    if 'default_symbols' not in st.session_state:
        st.session_state.default_symbols = random.sample(SP_500, 5)
        
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
    return symbols, start_date.isoformat(), end_date.isoformat(), period

