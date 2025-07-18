import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
@st.cache_data
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.title("📱 Mobile App vs Website Prediction")
st.markdown("### Predict yearly amount spent based on customer behavior")

st.sidebar.header("Input Features")

avg_session_length = st.sidebar.number_input("Avg. Session Length", min_value=20.0, max_value=40.0, value=30.0)
time_on_app = st.sidebar.number_input("Time on App", min_value=5.0, max_value=20.0, value=12.0)
time_on_web = st.sidebar.number_input("Time on Website", min_value=30.0, max_value=50.0, value=38.0)
membership_length = st.sidebar.number_input("Length of Membership (Years)", min_value=0.0, max_value=10.0, value=5.0)

if st.button("Predict Yearly Amount Spent"):
    input_data = np.array([[avg_session_length, time_on_app, time_on_web, membership_length]])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Yearly Amount Spent: **${prediction:,.2f}**")

# Show model coefficients & intercept
coefficients = model.coef_
intercept = model.intercept_
feature_names = ["Avg. Session Length", "Time on App", "Time on Website", "Membership Length"]

st.subheader("Model Coefficients")
coef_df = {name: coef for name, coef in zip(feature_names, coefficients)}
st.write(coef_df)
st.write(f"Intercept: {intercept:.2f}")

# Bar chart of coefficients
st.subheader("Feature Importance (Coefficient Magnitude)")

coef_magnitude = np.abs(coefficients)
fig, ax = plt.subplots()
ax.bar(feature_names, coef_magnitude, color='skyblue')
ax.set_ylabel("Coefficient Magnitude")
ax.set_title("Feature Importance in Linear Regression")
plt.xticks(rotation=30, ha='right')  
plt.tight_layout() 

st.pyplot(fig)

st.markdown("---")
st.write("Model trained using **Linear Regression** with R² = 0.981")
