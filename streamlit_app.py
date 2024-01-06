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

st.set_page_config(page_title="Portfolio Analyzer")
st.title('Portfolio Analyzer')
with st.form(key='form'):

    # TODO: Need to add in a feature inorder to validate a stock name
    symbols = st_tags(
        label='Enter the name or symbol of the stock in your portfolio to analyze',
        text='Press enter to add more',
        value= random.sample(SP_500, 5),
        suggestions = SP_500
        )

    start_date_value = date.today() - timedelta(days=365)
    end_date_value = date.today()

    # Take the date input of the timeframe
    start_date = st.date_input('Select the start date', value=start_date_value)
    end_date = st.date_input('Select the end date', value=end_date_value)

    analyze_button = st.form_submit_button('Analyze')

    # start analysis if the analyze button is clicked
    if analyze_button:

        # Check for any invalid date inputs
        if not start_date:
            start_date = date.today() - timedelta(days=3*365)

        if not end_date:
            end_date = date.today()

        if start_date == end_date:
            st.warning("Start date and end date can't be the same")

        elif start_date > end_date:
            st.warning("Start date cannot be greater than start date")
        symbols_string = ", ".join(symbols)
        st.write(
            f"Your portfolio consists for {len(symbols)} stocks and their symbols are **{symbols_string}**")

        df = pd.DataFrame()
        for i in range(len(symbols)):
            data = yf.download(symbols[i], start=start_date, end=end_date)[['Close']]
            if not data.empty:
                data = data.rename(columns={'Close': symbols[i]})
                if i == 0:
                    df = data
                if i != 0:
                    df = df.join(data)
            else:
                st.warning(f"Invalid stock symbol: {symbols[i]}")

        if not df.empty:

            st.subheader("Historical close price data")
            st.dataframe(df)
            st.subheader("Closing price chart")
            history_chart = px.line(
                df, x=df.index, y=df.columns, title="Portfolio Close Price History", labels={
                    "x": "Date", "y": "Close Price INR (₨)"
                })

            st.plotly_chart(history_chart, use_container_width=True)

            st.subheader("Correlation Matrix")
            st.write("""A Coefficient of **correlation** is a statistical measure of the relationship between two variables. It varies from -1 to 1, with 1 or -1 indicating perfect correlation. A correlation value close to 0 indicates no association between the variables. A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The correlation matrix will tell us the strength of the relationship between the stocks in our portfolio, which essentially can be used for effective diversification.""")
            correlation_matrix = df.corr(method='pearson')
            correlation_heatmap = px.imshow(
                correlation_matrix, title='Correlation between Stocks in your portfolio')
            st.plotly_chart(correlation_heatmap, use_container_width=True)

            st.subheader("Daily Simple Returns")
            st.write(
                "**Daily Simple Returns** is essentially the percentage change in the Prices being calculated daily.")
            daily_simple_return = df.pct_change(1)
            daily_simple_return.dropna(inplace=True)
            st.dataframe(daily_simple_return)

            daily_simple_return_plot = px.line(daily_simple_return, x=daily_simple_return.index,
                                            y=daily_simple_return.columns, title="Volatility in Daily simple returns", labels={"x": "Date", "y": "Daily simple returns"})
            st.plotly_chart(daily_simple_return_plot, use_container_width=True)

            st.subheader("Average Daily returns")
            daily_avg = daily_simple_return.mean()
            daily_avg = daily_avg*100
            daily_avg.name = "Average Daily returns"
            st.dataframe(daily_avg)

            daily_simple_return_boxplot = px.box(
                daily_simple_return, title="Risk Box Plot")

            st.plotly_chart(daily_simple_return_boxplot,
                            use_container_width=True)

            st.subheader("Annualized Standard Deviation")
            st.write('**Annualized Standard Deviation** (Volatality(%), 252 trading days) of individual stocks in your portfolio on the basis of daily simple returns.')
            annual_std = daily_simple_return.std() * np.sqrt(252) * 100
            annual_std.name = "Annualized Standard Deviation"
            st.write(annual_std)

            st.subheader("Return Per Unit Of Risk")
            st.write("""The higher this ratio, the better it is. After adjusting for a risk-free rate, this ratio is also called Sharpe Ratio, a measure of risk-adjusted return. It describes how much excess return you receive for the volatility of holding a riskier asset.""")
            return_per_unit_risk = daily_avg / \
                (daily_simple_return.std() * np.sqrt(252)) * 100
            return_per_unit_risk.name = "Return Per Unit Of Risk"
            st.dataframe(return_per_unit_risk)
            st.subheader("Cumulative Returns")
            daily_cummulative_simple_return = (daily_simple_return+1).cumprod()
            st.dataframe(daily_cummulative_simple_return)
            daily_cummulative_simple_return_plot = px.line(
                daily_cummulative_simple_return, x=daily_cummulative_simple_return.index, y=daily_cummulative_simple_return.columns, title="Daily Cummulative Simple returns/growth of investment", labels={"x": "Date", "y": "Growth of ₨ 1 investment"})

            st.plotly_chart(daily_cummulative_simple_return_plot,
                            use_container_width=True)
            st.subheader("Modern Portfolio Theory")
            st.write("""
            **Modern Portfolio Theory**, or MPT (also known as mean-variance analysis), is a mathematical framework for constructing a portfolio of assets to maximize expected return for a given level of market risk (Standard Deviation of Portfolio Returns). Since risk is associated with variability in profit, we can quantify it using measures of dispersion such as variance and standard deviation.
            
            > **The trade-off between Risk & return forms the basis of the portfolio construction**. 

            It is imperative that the Higher the Risk, Higher will be the Return, So different investors will evaluate the trade-off differently based on individual risk aversion characteristics.
            **It is possible to reduce risk while increasing the returns through efficient diversification**,, i.e., by combining negatively correlated assets. It is possible to construct an efficient set of portfolios that have the least risk for a given return or highest return for a given level of risk; investors can choose a point on this **efficient frontier** depending on their risk-return preferences. This process of constructing an efficient set of portfolios is labeled as portfolio optimization, which is quite a complex task mathematically.
            The expected return of the portfolio is calculated as a weighted sum of the individual assets' returns. The Portfolio risk depends on the proportion (weights) invested in each security, their individual risks, and their correlation or covariance. These two terms are used interchangeably, but there lies a difference between the two,

            - **Covariance** - The covariance can measure the extent to which two random variables vary together.
            - **Correlation** - The problem with Covariance is that it's not standardized & to do so, we divide the Covariance between two variables by their standard deviation, which gives us the coefficient of correlation ranging from -1 to 1.
            """)

            st.write("""
            An **Efficient Frontier** represents all possible portfolio combinations. It has the maximum return portfolio, consisting of a single asset with the highest return at the extreme right and the minimum variance portfolio on the extreme left. The returns represent the y-axis, while the level of risk lies on the x-axis.
            """)
            st.image("efficient_frontier.png")

            # calculating expected annual return and annualized sample covariance matrix of daily assets returns
            mean = expected_returns.mean_historical_return(df)
            mean.name = "Mean Historical Return"
            st.subheader("Mean Historical Return")
            st.write(mean)
            st.subheader("Sample covariance matrix")
            sample_covariance_matrix = risk_models.sample_cov(
                df)  # for sample covariance matrix
            st.dataframe(sample_covariance_matrix)
            sample_covariance_matrix_heatmap = px.imshow(
                sample_covariance_matrix, title="Sample Covariance Matrix")
            st.plotly_chart(sample_covariance_matrix_heatmap,
                            use_container_width=True)

            ef = EfficientFrontier(mean, sample_covariance_matrix)
            weights = ef.max_sharpe()  # for maximizing the Sharpe ratio #Optimization
            st.subheader("Sharpe ratio")
            st.write(weights)
            cleaned_weights = ef.clean_weights()  # to clean the raw weights
            # Get the Keys and store them in a list
            labels = list(cleaned_weights.keys())
            # Get the Values and store them in a list
            values = list(cleaned_weights.values())

            pie_chart = px.pie(df, values=values, names=labels,
                            title='Optimized Portfolio Allocation')
            st.plotly_chart(pie_chart, use_container_width=True)

            st.subheader("Portfolio Performance")
            mu, sigma, sharpe = ef.portfolio_performance()
            portfolio_performance_string = """Expected annual return: {:.1f}% \nAnnual volatility: {:.1f}% \nSharpe Ratio: {:.2f}""".format(100 * mu, 100 * sigma, sharpe)
            st.text(portfolio_performance_string)
            st.balloons()


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:"Made with 💓 by Chandra Sekhar Mullu"; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
