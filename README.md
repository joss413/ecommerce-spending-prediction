# 🛍️ E-Commerce Spending Prediction App

# 📱 Mobile App vs Website Yearly Spending Prediction App

This Streamlit app predicts the yearly amount spent by customers based on their online behavior, using a Linear Regression model trained on the [Ecommerce Customers Dataset](https://www.kaggle.com/datasets/kolawale/focusing-on-mobile-app-or-website).

---

## 🚀 Getting Started

### Dataset

Due to Kaggle licensing, the original dataset is **NOT** included in this repo.

Please download the dataset manually from:

[Ecommerce Customers Dataset on Kaggle](https://www.kaggle.com/datasets/kolawale/focusing-on-mobile-app-or-website)

and place the file `Ecommerce_Customers.csv` in the project root folder if you want to train or test locally.

---

### Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/ecommerce-spending-prediction.git
cd ecommerce-spending-prediction
 ```
2. **Install dependencies:**:   
   ```commandline
    pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**:   
   ```commandline
    streamlit run app.py
   ```

🔧 Features

Predict yearly amount spent from customer behavior inputs.

Interactive UI with sidebar controls.

Demo EDA using synthetic data (no private dataset included).

Trained Linear Regression model with R² score of 0.981 on real data.   

🔧 Files in This Repo

app.py — Streamlit app code.

model.joblib — Trained Linear Regression model.

requirements.txt — Python dependencies.

runtime.txt — Streamlit Cloud Python version.


## 👨‍💻 Author

Yoseph Negash

📧 yosephn22@gmail.com

📅 2025
