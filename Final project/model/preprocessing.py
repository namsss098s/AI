import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(csv_path):
    """
    Load insurance dataset và chia train/test
    """

    data = pd.read_csv(csv_path)

    print("Dataset preview:")
    print(data.head())

    y = data["charges"]
    X = data.drop("charges", axis=1)

    categorical_cols = ["sex", "smoker", "region"]

    X = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    print("\nFeature columns after encoding:")
    print(X.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, y_train, y_test