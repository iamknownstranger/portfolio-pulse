import random
from datetime import date, timedelta
import streamlit as st
from streamlit_tags import st_tags

# Common list of suggestions (could also be imported from another module)
SP_500 = ["MMM", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADBE", "AAP", "AMD", "AES",
          "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALXN", "ALGN", "ALLE",
          "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP",
          # ... (rest of symbols)
         ]

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
        st.form_submit_button('Apply Filters')
    return symbols, start_date.isoformat(), end_date.isoformat()
