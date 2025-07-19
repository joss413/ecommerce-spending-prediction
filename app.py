import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
@st.cache_data
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# feature names 
feature_names = ["Avg. Session Length", "Time on App", "Time on Website", "Membership Length"]

st.set_page_config(page_title="Yearly Amount Spent Prediction", layout="wide")
st.title("📱 Mobile App vs Website Prediction")
st.markdown("### Predict yearly amount spent based on customer behavior")

# Sidebar inputs
st.sidebar.header("Input Features")
avg_session_length = st.sidebar.number_input("Avg. Session Length", min_value=20.0, max_value=40.0, value=30.0, step=0.1)
time_on_app = st.sidebar.number_input("Time on App", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
time_on_web = st.sidebar.number_input("Time on Website", min_value=30.0, max_value=50.0, value=38.0, step=0.1)
membership_length = st.sidebar.number_input("Length of Membership (Years)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

input_features = np.array([[avg_session_length, time_on_app, time_on_web, membership_length]])

# Predict button and prediction display
if st.sidebar.button("Predict Yearly Amount Spent"):
    prediction = model.predict(input_features)[0]
    st.success(f"💰 Predicted Yearly Amount Spent: **${prediction:,.2f}**")

    # Feature contributions (value * coefficient)
    contributions = input_features.flatten() * model.coef_

    # Plot feature contributions
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(feature_names, contributions, color='mediumseagreen')
    ax.set_ylabel("Contribution to Prediction ($)")
    ax.set_title("Feature Contributions to Predicted Amount")
    ax.axhline(0, color='grey', linewidth=0.8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, color='black')

    st.pyplot(fig)

# model coefficients & intercept
st.subheader("Model Coefficients & Intercept")
coef_df = {name: coef for name, coef in zip(feature_names, model.coef_)}
st.table(coef_df)
st.write(f"Intercept: {model.intercept_:.2f}")

# Coefficient magnitude bar chart
st.subheader("Feature Importance (Coefficient Magnitude)")
coef_magnitude = np.abs(model.coef_)
fig2, ax2 = plt.subplots(figsize=(8,4))
bars2 = ax2.bar(feature_names, coef_magnitude, color='skyblue')
ax2.set_ylabel("Coefficient Magnitude")
ax2.set_title("Feature Importance in Linear Regression")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=9, color='black')

st.pyplot(fig2)

st.markdown("---")
st.write("Model trained using **Linear Regression** with R² = 0.981")
