import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ====== PATH SETUP ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "medical_insurance_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "insurance_model.pkl")

print("Loading data from:", DATA_PATH)
print("Loading model from:", MODEL_PATH)

# ====== LOAD DATA ======
data = pd.read_csv(DATA_PATH)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

X = data.drop("charges", axis=1)
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== LOAD MODEL ======
model = joblib.load(MODEL_PATH)

# 🔥 Đảm bảo đúng thứ tự feature khi predict
X_test = X_test[model.feature_names_in_]

# ====== PREDICT ======
y_pred = model.predict(X_test)

# ====== EVALUATION METRICS ======
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# ====== ACTUAL VS PREDICTED ======
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")
plt.title(f"Actual vs Predicted Insurance Cost\nR² = {r2:.4f}")

plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

# ====== RESIDUAL PLOT ======
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Insurance Cost")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")

plt.grid(True)

plt.show()