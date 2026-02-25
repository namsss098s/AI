import os
import sys
import joblib
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "insurance_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

# Load model + feature list
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


def predict_insurance(age, bmi, children, smoker, sex, region):
    """
    Predict insurance cost
    """

    # Tạo dict chứa toàn bộ feature = 0
    input_dict = {col: 0 for col in feature_columns}

    # Gán giá trị numeric
    input_dict["age"] = age
    input_dict["bmi"] = bmi
    input_dict["children"] = children

    # Encode smoker
    if "smoker_yes" in input_dict:
        input_dict["smoker_yes"] = 1 if smoker == 1 else 0

    # Encode sex
    if "sex_male" in input_dict:
        input_dict["sex_male"] = 1 if sex == 1 else 0

    # Encode region
    region_column = f"region_{region}"
    if region_column in input_dict:
        input_dict[region_column] = 1

    # Convert to DataFrame đúng thứ tự
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)

    return prediction[0]


if __name__ == "__main__":
    print("\n===== INSURANCE COST PREDICTION =====")

    age = int(input("Age: "))
    bmi = float(input("BMI: "))
    children = int(input("Number of children: "))
    smoker = int(input("Smoker? (1=yes, 0=no): "))
    sex = int(input("Male? (1=yes, 0=no): "))

    print("\nRegion options:")
    print("northeast | northwest | southeast | southwest")
    region = input("Region: ").strip().lower()

    predicted_cost = predict_insurance(
        age,
        bmi,
        children,
        smoker,
        sex,
        region
    )

    print(f"\nPredicted Insurance Cost: ${predicted_cost:.2f}")