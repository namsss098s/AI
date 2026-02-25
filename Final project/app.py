import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "medical_insurance_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "insurance_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

data = load_data()
model, feature_columns = load_model()

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop("charges", axis=1)
y = data_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_test = X_test.reindex(columns=feature_columns, fill_value=0)

y_pred_test = model.predict(X_test)

st.title("🏥 Medical Insurance Cost Prediction App")

st.write(
    "This application predicts medical insurance charges "
    "based on personal attributes."
)

st.markdown("---")
st.header("🔧 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

st.markdown("---")
st.header("📊 Prediction Result")

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.success(f"Predicted Insurance Cost: **${prediction:,.2f}**")

st.markdown("---")
st.header("📈 Model Evaluation")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred_test, alpha=0.6)

min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--")

ax1.set_xlabel("Actual Insurance Cost")
ax1.set_ylabel("Predicted Insurance Cost")
ax1.set_title("Actual vs Predicted")

st.pyplot(fig1)

residuals = y_test - y_pred_test

fig2, ax2 = plt.subplots()
ax2.scatter(y_pred_test, residuals, alpha=0.6)
ax2.axhline(0, linestyle="--")

ax2.set_xlabel("Predicted Insurance Cost")
ax2.set_ylabel("Residual (Actual - Predicted)")
ax2.set_title("Residual Plot")

st.pyplot(fig2)

st.markdown("---")
st.write("🔹 Author: Tran Hoang Nam")