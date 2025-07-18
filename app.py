import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
@st.cache_data
def load_model():
    return joblib.load("model.joblib")

# Load dataset (commented out to avoid FileNotFoundError)
# @st.cache_data
# def load_data():
#     return pd.read_csv("Ecommerce_Customers.csv")

model = load_model()
# df = load_data()  # Commented out

st.title("📱 Mobile App vs Website Prediction")
st.markdown("### Predict yearly amount spent based on customer behavior")

st.sidebar.header("Input Features")

# Sidebar inputs
avg_session_length = st.sidebar.number_input("Avg. Session Length", min_value=20.0, max_value=40.0, value=30.0)
time_on_app = st.sidebar.number_input("Time on App", min_value=5.0, max_value=20.0, value=12.0)
time_on_web = st.sidebar.number_input("Time on Website", min_value=30.0, max_value=50.0, value=38.0)
membership_length = st.sidebar.number_input("Length of Membership (Years)", min_value=0.0, max_value=10.0, value=5.0)

# Prediction
if st.button("Predict Yearly Amount Spent"):
    input_data = np.array([[avg_session_length, time_on_app, time_on_web, membership_length]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Yearly Amount Spent: **${prediction:,.2f}**")

# Show dataset preview (commented out since dataset is not loaded)
# if st.checkbox("Show Dataset Preview"):
#     st.dataframe(df.head())

st.markdown("---")
st.write("Model trained using **Linear Regression** with R² = 0.981")
