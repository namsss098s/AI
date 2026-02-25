import numpy as np
import pandas as pd

np.random.seed(42)

num_samples = 1000

age = np.random.randint(18, 66, num_samples)

sex = np.random.choice(["male", "female"], num_samples)

bmi = np.random.uniform(15, 40, num_samples)

children = np.random.randint(0, 6, num_samples)

smoker = np.random.choice(["yes", "no"], num_samples, p=[0.2, 0.8])

region = np.random.choice(
    ["northeast", "northwest", "southeast", "southwest"],
    num_samples
)

charges = (
    2000
    + age * 250
    + bmi * 350
    + children * 500
    + np.where(smoker == "yes", 20000, 0)
    + np.random.normal(0, 3000, num_samples)
)

data = pd.DataFrame({
    "age": age,
    "sex": sex,
    "bmi": bmi.round(2),
    "children": children,
    "smoker": smoker,
    "region": region,
    "charges": charges.astype(int)
})

print(data.head())
print("\nDataset shape:", data.shape)

data.to_csv("medical_insurance_data.csv", index=False)
print("\nDataset saved as medical_insurance_data.csv")