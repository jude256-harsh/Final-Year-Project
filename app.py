import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Ugandan Crop Price Prediction System", page_icon="🌾", layout="centered")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Load saved model and metadata
@st.cache_resource
def load_artifacts():
    model = joblib.load('crop_price_xgboost_model.pkl')
    meta = joblib.load('model_meta.pkl')
    return model, meta['markets'], meta['commodities'], meta['market2code'], meta['commodity2code']

model, markets, commodities, market2code, commodity2code = load_artifacts()

# --- AUTHENTICATION SCREEN ---
if not st.session_state['logged_in']:
    st.title("🌾 Ugandan Crop Price Prediction System")
    st.subheader("Farmer Portal Login")
    st.write("Please log in with your credentials or farmer ID to access market forecasts.")

    with st.form("login_form"):
        username = st.text_input("Farmer ID / Username", placeholder="e.g. farmer_001")
        password = st.text_input("Access PIN / Password", type="password", placeholder="e.g. 1234")
        login_btn = st.form_submit_button("Log In")

        if login_btn:
            # Simple validation logic (you can connect this to a database later for your project)
            if username.strip() != "" and password.strip() != "":
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Please enter a valid Farmer ID and PIN.")

# --- MAIN PREDICTION DASHBOARD (AFTER LOGIN) ---
else:
    st.title("🇺🇬 Ugandan Crop Price Prediction Dashboard")
    st.success(f"Welcome, Farmer **{st.session_state['username']}**! Ready to forecast market prices.")

    if st.button("Log Out"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.markdown("---")

    # Sidebar inputs
    st.sidebar.header("Prediction Parameters")
    selected_market = st.sidebar.selectbox("Select Market", markets)
    selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)

    lag_1_price = st.sidebar.number_input("Last Month Price (UGX)", value=1500.0, step=50.0)
    lag_3_price = st.sidebar.number_input("Price 3 Months Ago (UGX)", value=1450.0, step=50.0)
    rolling_mean = st.sidebar.number_input("3-Month Rolling Average (UGX)", value=1475.0, step=50.0)

    month = st.sidebar.slider("Forecast Month", 1, 12, 6)
    year = st.sidebar.slider("Forecast Year", 2026, 2030, 2026)

    if st.button("Predict Future Price", type="primary"):
        m_enc = market2code[selected_market]
        c_enc = commodity2code[selected_commodity]

        # Default lat/long reference point
        lat, lon = 0.3476, 32.5825

        input_data = pd.DataFrame([[
            m_enc, c_enc, lag_1_price, lag_3_price, 
            rolling_mean, month, year, lat, lon
        ]], columns=[
            'market_encoded', 'commodity_encoded', 'lag_1', 'lag_3', 
            'rolling_mean_3', 'month', 'year', 'latitude', 'longitude'
        ])

        prediction = model.predict(input_data)[0]

        st.balloons()
        st.markdown(f"### 📊 Forecast Results")
        st.info(f"Estimated Price for **{selected_commodity}** in **{selected_market}** market:")
        st.success(f"**{prediction:,.2f} UGX / KG**")