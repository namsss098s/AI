import os
import sys
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocessing import load_and_split_data


DATA_PATH = os.path.join(BASE_DIR, "data", "medical_insurance_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "insurance_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed.")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

joblib.dump(model, MODEL_PATH)

joblib.dump(X_train.columns.tolist(), FEATURE_PATH)

print(f"\nModel saved successfully at: {MODEL_PATH}")
print(f"Feature columns saved at: {FEATURE_PATH}")