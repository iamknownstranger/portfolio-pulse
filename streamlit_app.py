import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, timedelta
from streamlit_tags import st_tags
import plotly.express as px
from nsepy import get_history


st.title('Portfolio Analyzer')

with st.form(key='form'):

    # TODO: Need to add in a feature inorder to validate a stock name
    symbols = st_tags(
        label='Enter the name or symbol of the stock in your portfolio to analyze',
        text='Press enter to add more',
        value=["SBIN", "HDFCBANK", "ITC", "ASIANPAINT"],
        suggestions=['CIPLA', 'BPCL', 'SUNPHARMA', 'JSWSTEEL', 'IOC', 'DRREDDY', 'POWERGRID', 'COALINDIA', 'ITC', 'TITAN', 'DIVISLAB', 'SHREECEM', 'GRASIM', 'ASIANPAINT', 'ONGC', 'BAJAJFINSV', 'SBIN', 'BAJFINANCE', 'HEROMOTOCO', 'SBILIFE', 'KOTAKBANK', 'HDFCBANK', 'ICICIBANK', 'TECHM', 'HCLTECH', 'BAJAJ-AUTO', 'RELIANCE', 'UPL', 'LT', 'INFY', 'HDFC', 'HINDUNILVR', 'EICHERMOT', 'WIPRO', 'TATAMOTORS', 'INDUSINDBK', 'BHARTIARTL', 'TCS', 'AXISBANK', 'ULTRACEMCO', 'HDFCLIFE', 'ADANIPORTS', 'M&M', 'BRITANNIA', 'TATASTEEL', 'NTPC', 'HINDALCO', 'TATACONSUM', 'MARUTI', 'NESTLEIND'])

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
        
        st.write(
            f"Your portfolio consists for {len(symbols)} stocks and their symbols are *{symbols}*")
        
        df = pd.DataFrame()
        for i in range(len(symbols)):
            data = get_history(symbol=symbols[i], start=start_date, end=(end_date))[
                ['Symbol', 'Close']]
            if not data.empty:
                data.rename(columns={'Close': data['Symbol'][0]}, inplace=True)
                data.drop(['Symbol'], axis=1, inplace=True)
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
                    "x":"Date", "y":"Close Price INR (₨)"
                })

            st.plotly_chart(history_chart)

            st.subheader("Correlation Matrix")
            st.write("""A Coefficient of correlation is a statistical measure of the relationship between two variables. It varies from -1 to 1, with 1 or -1 indicating perfect correlation. A correlation value close to 0 indicates no association between the variables. A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The correlation matrix will tell us the strength of the relationship between the stocks in our portfolio, which essentially can be used for effective diversification.""")
            correlation_matrix = df.corr(method='pearson')
            correlation_heatmap = px.imshow(
                correlation_matrix, title='Correlation between Stocks in your portfolio')
            st.plotly_chart(correlation_heatmap)

            st.subheader("Daily Simple Returns")
            st.write(
                "Daily Simple Returns is essentially the percentage change in the Prices being calculated daily.")
            daily_simple_return = df.pct_change(1)
            daily_simple_return.dropna(inplace=True)
            st.dataframe(daily_simple_return)

            daily_simple_return_plot = px.line(daily_simple_return, x=daily_simple_return.index,
                                               y=daily_simple_return.columns, title="Volatility in Daily simple returns", labels={"x":"Date", "y":"Daily simple returns"})
            st.plotly_chart(daily_simple_return_plot)

            st.subheader("Average Daily returns")
            # print('Average Daily returns(%) of stocks in your portfolio')
            daily_avg = daily_simple_return.mean()
            daily_avg = daily_avg*100
            daily_avg.columns = ["Ticker", "Average Daily returns"]
            st.dataframe(daily_avg)

            daily_simple_return_boxplot = px.box(daily_simple_return, title="Risk Box Plot")

            st.plotly_chart(daily_simple_return_boxplot)

            st.subheader("Annualized Standard Deviation")
            st.write('Annualized Standard Deviation (Volatality(%), 252 trading days) of individual stocks in your portfolio on the basis of daily simple returns.')
            st.write(daily_simple_return.std() * np.sqrt(252) * 100)

            st.subheader("Return Per Unit Of Risk")
            st.write("""The higher this ratio, the better it is. After adjusting for a risk-free rate, this ratio is also called Sharpe Ratio, a measure of risk-adjusted return. It describes how much excess return you receive for the volatility of holding a riskier asset.""")
            return_per_unit_risk = daily_avg / \
                (daily_simple_return.std() * np.sqrt(252)) * 100
            st.dataframe(return_per_unit_risk)
            st.subheader("Cumulative Returns")
            daily_cummulative_simple_return = (daily_simple_return+1).cumprod()
            st.dataframe(daily_cummulative_simple_return)
            daily_cummulative_simple_return_plot = px.line(
                daily_cummulative_simple_return, x=daily_cummulative_simple_return.index, y=daily_cummulative_simple_return.columns, title="Daily Cummulative Simple returns/growth of investment", labels={"x":"Date", "y":"Growth of ₨ 1 investment"})

            st.plotly_chart(daily_cummulative_simple_return_plot)
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
