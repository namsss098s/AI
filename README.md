# 🏥 Medical Insurance Cost Prediction

A Machine Learning project that predicts medical insurance charges based on personal and demographic attributes using regression models.

---

## 📌 Project Overview

This project applies regression-based machine learning techniques to estimate medical insurance costs (`charges`) using the following features:

- Age
- Sex
- BMI
- Number of Children
- Smoking Status
- Region

The system includes:

- Data preprocessing
- Model training
- Model saving & loading
- Model evaluation
- Visualization
- Streamlit Web Application

---



## ⚙️ Installation

### 1️⃣ Clone Repository

bash
git clone <your-repository-url>
cd medical-insurance-prediction

2️⃣ Create Virtual Environment (Recommended)

python -m venv venv

source venv/bin/activate        # Mac/Linux

venv\Scripts\activate           # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

Or manually:

pip install pandas numpy scikit-learn matplotlib streamlit joblib

🚀 How to Train the Model

python model/train.py

This will:

Load dataset

Preprocess data

Train Linear Regression model

Save:

insurance_model.pkl

feature_columns.pkl

🔍 How to Evaluate the Model

python plots/compare.py

This will:

Load trained model

Calculate evaluation metrics:

MSE

RMSE

R² Score

Generate:

Actual vs Predicted plot

Residual plot

🌐 Run Web Application

streamlit run app.py

The application allows users to:

Input personal data

Predict insurance cost

View model evaluation charts

📊 Model Evaluation Metrics

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Interpretation:

R² close to 1 → Model explains most variance

Random residual distribution → Linear regression assumptions are satisfied

Patterned residuals → Possible need for advanced models

🧠 Machine Learning Approach

Algorithm Used:

Linear Regression

Data Processing:

One-hot encoding for categorical variables

Train/Test split (80/20)

Feature alignment using feature_columns.pkl

🔮 Future Improvements

Add Random Forest & Gradient Boosting comparison

Cross-validation

Hyperparameter tuning

Feature importance visualization

Deploy to Streamlit Cloud
