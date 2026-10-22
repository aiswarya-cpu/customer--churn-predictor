import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    return rf, xgb

rf_model, xgb_model = load_models()

st.title("🛍️ Customer Churn Prediction in E-Commerce")
st.write("Predict customer churn risk using machine learning.")
st.caption("Group 7 | Random Forest and XGBoost")

model_choice = st.selectbox(
    "Choose a machine learning model",
    ["XGBoost", "Random Forest"]
)

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0.0, value=12.0)
    warehouse = st.number_input("Warehouse to Home Distance", min_value=0.0, value=10.0)
    devices = st.number_input("Number of Registered Devices", min_value=0, value=3)
    category = st.selectbox(
        "Preferred Order Category",
        ["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Others"]
    )
    satisfaction = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5])

with col2:
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    addresses = st.number_input("Number of Addresses", min_value=1, value=2)
    complain = st.selectbox("Has Complained?", [0, 1])
    days_since = st.number_input("Days Since Last Order", min_value=0.0, value=3.0)
    cashback = st.number_input("Cashback Amount", min_value=0.0, value=100.0)

if st.button("Predict Churn", type="primary"):
    customer = pd.DataFrame([{
        "Tenure": tenure,
        "WarehouseToHome": warehouse,
        "NumberOfDeviceRegistered": devices,
        "PreferedOrderCat": category,
        "SatisfactionScore": satisfaction,
        "MaritalStatus": marital,
        "NumberOfAddress": addresses,
        "Complain": complain,
        "DaySinceLastOrder": days_since,
        "CashbackAmount": cashback
    }])

    model = xgb_model if model_choice == "XGBoost" else rf_model
    prediction = int(model.predict(customer)[0])
    probabilities = model.predict_proba(customer)[0]
    churn_probability = float(probabilities[1])

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{churn_probability * 100:.1f}%")

    if churn_probability >= 0.70:
        st.error("🔴 High Churn Risk")
    elif churn_probability >= 0.40:
        st.warning("🟠 Medium Churn Risk")
    else:
        st.success("🟢 Low Churn Risk")

    st.write(
        "The model predicts that this customer is likely to churn."
        if prediction == 1
        else "The model predicts that this customer is likely to stay."
    )

    st.caption("This is a model-based estimate, not a guarantee of customer behavior.")
