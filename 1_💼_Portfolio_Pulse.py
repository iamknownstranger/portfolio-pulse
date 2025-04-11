from datetime import date, timedelta
import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from streamlit_tags import st_tags
from common.sidebar import render_sidebar

st.set_page_config(page_title="Portfolio Analyzer", page_icon="💼", layout="wide")

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


st.title('Portfolio Analyzer')

if 'default_symbols' not in st.session_state:
    st.session_state.default_symbols = random.sample(SP_500, 5)

# Render the common sidebar filters
symbols, start_date, end_date = render_sidebar()

if not start_date:
    start_date = date.today() - timedelta(days=3*365)
if not end_date:
    end_date = date.today()
if start_date == end_date:
    st.warning("Start date and end date can't be the same")
elif start_date > end_date:
    st.warning("Start date cannot be greater than end date")

symbols_string = ", ".join(symbols)
st.write(f"Your portfolio consists of {len(symbols)} stocks and their symbols are **{symbols_string}**")

@st.cache_data(ttl=86400)
def get_data(tickers, start, end):
    try:
        tickers = tickers + ['BLL']
        # Attempt to download data for all tickers at once
        df = yf.download(tickers, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
        df.index = df.index.date
        df = df.dropna(axis=1, how='all')
        return df.replace(to_replace='None', value=np.nan).dropna(axis=0, how="all")
    except Exception as e:
        dfs = []
        # Loop over tickers individually to identify the ones that fail.
        for ticker in tickers:
            try:
                df_ticker = yf.download(ticker, start=start, end=end, auto_adjust=False, multi_level_index=False)["Close"]
                df_ticker.index = df_ticker.index.date
                dfs.append(df_ticker.rename(ticker))
            except Exception as e_indiv:
                st.error(f"Error for {ticker}: {e_indiv}")
        if dfs:
            combined = pd.concat(dfs, axis=1)
            return combined.replace(to_replace='None', value=np.nan).dropna(axis=0, how="all")
        return pd.DataFrame()

df = get_data(symbols, start_date, end_date)

if not df.empty:
    st.subheader("Historical close price data")
    st.dataframe(df)
    st.subheader("Closing price chart")
    st.line_chart(df)

    # Calculate and display analytics
    st.subheader("Correlation Matrix")
    st.write("""A Coefficient of **correlation** is a statistical measure of the relationship between two variables. It varies from -1 to 1, with 1 or -1 indicating perfect correlation. A correlation value close to 0 indicates no association between the variables. A correlation matrix is a table showing correlation coefficients between variables.""")
    correlation_matrix = df.corr(method='pearson')
    correlation_heatmap = px.imshow(correlation_matrix, title='Correlation between Stocks in your portfolio')
    st.plotly_chart(correlation_heatmap, use_container_width=True)

    st.subheader("Daily Simple Returns")
    st.write("**Daily Simple Returns** is the percentage change in prices calculated daily.")
    daily_simple_return = df.pct_change(1)
    daily_simple_return.dropna(inplace=True)
    st.dataframe(daily_simple_return)

    daily_simple_return_plot = px.line(daily_simple_return, x=daily_simple_return.index,
                                       y=daily_simple_return.columns,
                                       title="Volatility in Daily Simple Returns",
                                       labels={"x": "Date", "y": "Daily Simple Returns"})
    st.plotly_chart(daily_simple_return_plot, use_container_width=True)

    st.subheader("Average Daily Returns")
    daily_avg = daily_simple_return.mean() * 100
    cols_daily_avg = st.columns(len(daily_avg) + 1)
    for col, (label, value) in zip(cols_daily_avg[1:], daily_avg.items()):
        col.metric(label, f"{value:.2f}%")

    daily_simple_return_boxplot = px.box(daily_simple_return, title="Risk Box Plot")
    st.plotly_chart(daily_simple_return_boxplot, use_container_width=True)

    st.subheader("Annualized Standard Deviation")
    st.write("**Annualized Standard Deviation** (Volatility, 252 trading days) of individual stocks in your portfolio.")
    annual_std = daily_simple_return.std() * np.sqrt(252) * 100
    cols_std = st.columns(len(annual_std) + 1)
    for col, (label, value) in zip(cols_std[1:], annual_std.items()):
        col.metric(label, f"{value:.2f}%")
    
    st.subheader("Return Per Unit Of Risk")
    st.write("After adjusting for a risk-free rate, this ratio (also called Sharpe Ratio) measures risk-adjusted return.")
    return_per_unit_risk = daily_avg / (daily_simple_return.std() * np.sqrt(252)) * 100
    cols_risk = st.columns(len(return_per_unit_risk) + 1)
    for col, (label, value) in zip(cols_risk[1:], return_per_unit_risk.items()):
        col.metric(label, f"{value:.2f}")

    daily_returns = df.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()
    st.subheader("📈 Cumulative Returns")
    st.line_chart(cumulative_returns)
    
    st.subheader("Modern Portfolio Theory")
    st.write("""
    **Modern Portfolio Theory** (MPT) is a mathematical framework for constructing a portfolio that maximizes expected return for a given level of risk. It explains how investors can construct portfolios to optimize or maximize expected return based on a given level of market risk.
    """)
    st.write("""
    An **Efficient Frontier** represents all possible portfolio combinations. It shows the maximum return portfolio at one end and the minimum variance portfolio at the other.
    """)
    st.image("efficient_frontier.png")

    mean = expected_returns.mean_historical_return(df)
    st.subheader("Mean Historical Return")
    cols_mean_return = st.columns(len(mean) + 1)
    for col, (label, value) in zip(cols_mean_return[1:], mean.items()):
        col.metric(label, f"{value:.2f}")

    st.subheader("Sample Covariance Matrix")
    sample_covariance_matrix = risk_models.sample_cov(df)
    st.dataframe(sample_covariance_matrix)
    sample_covariance_matrix_heatmap = px.imshow(sample_covariance_matrix, title="Sample Covariance Matrix")
    st.plotly_chart(sample_covariance_matrix_heatmap, use_container_width=True)

    ef = EfficientFrontier(mean, sample_covariance_matrix)
    try:
        weights = ef.max_sharpe()
    except ValueError as e:
        st.error(f"Error optimizing portfolio using max_sharpe: {e}. Falling back to min_volatility.")
        weights = ef.min_volatility()
    st.subheader("Optimized Weights (Maximize Sharpe Ratio)")
    cols_weights = st.columns(len(weights) + 1)
    for col, (label, value) in zip(cols_weights[1:], weights.items()):
        col.metric(label, f"{value:.2f}")
    cleaned_weights = ef.clean_weights()
    labels = list(cleaned_weights.keys())
    values = list(cleaned_weights.values())
    pie_chart = px.pie(df, values=values, names=labels, title='Optimized Portfolio Allocation')
    st.plotly_chart(pie_chart, use_container_width=True)

    st.subheader("Portfolio Performance")
    mu, sigma, sharpe = ef.portfolio_performance()
    cols_perf = st.columns(4)
    cols_perf[1].metric("Expected Annual Return", f"{100 * mu:.2f}%")
    cols_perf[2].metric("Annualized Volatility", f"{100 * sigma:.2f}%")
    cols_perf[3].metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.balloons()

    # --- New Portfolio Performance Metrics ---
    portfolio_daily = df.pct_change().dropna().mean(axis=1)
    portfolio_daily.name = "portfolio_daily"  # assign name for merging
    portfolio_cum = (1 + portfolio_daily).cumprod()
    portfolio_vol = portfolio_daily.std() * np.sqrt(252) * 100
    portfolio_max_dd = ((portfolio_cum.cummax() - portfolio_cum) / portfolio_cum.cummax() * 100).max()
    downside_std_pf = portfolio_daily[portfolio_daily < 0].std() * np.sqrt(252)
    portfolio_sortino = (portfolio_daily.mean() * 252) / downside_std_pf if downside_std_pf != 0 else np.nan

    sp500_pf = yf.download("^GSPC", start=start_date, end=end_date)
    sp500_pf["daily_return"] = sp500_pf["Close"].pct_change()
    sp500_daily = sp500_pf["daily_return"].dropna()
    # --- New: Convert indices to datetime for merging ---
    portfolio_daily.index = pd.to_datetime(portfolio_daily.index)
    sp500_daily.index = pd.to_datetime(sp500_daily.index)
    portfolio_daily.index.name = "date"
    sp500_daily.index.name = "date"
    merged_pf = pd.merge(portfolio_daily.reset_index(),
                         sp500_daily.rename("sp500_return").reset_index(),
                         on="date", how="inner")
    beta_pf = merged_pf["portfolio_daily"].cov(merged_pf["sp500_return"]) / merged_pf["sp500_return"].var() if not merged_pf.empty else np.nan

    st.subheader("📊 Portfolio Performance Metrics")
    cols_pf = st.columns(5)
    cols_pf[0].metric("Cumulative Return", f"{(portfolio_cum.iloc[-1]-1)*100:.2f}%")
    cols_pf[1].metric("Annualized Vol", f"{portfolio_vol:.2f}%")
    cols_pf[2].metric("Max Drawdown", f"{portfolio_max_dd:.2f}%")
    cols_pf[3].metric("Sortino Ratio", f"{portfolio_sortino:.2f}")
    cols_pf[4].metric("Beta vs S&P500", f"{beta_pf:.2f}")

    st.subheader("📈 Cumulative Returns (Portfolio)")
    st.line_chart(portfolio_cum)

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