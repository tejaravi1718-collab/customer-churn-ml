from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def split_data(df, target_column="Churn Value"):
    """
    Splits dataframe into train and test sets.
    Removes leakage columns before training.
    """

    leakage_cols = [
        "Churn Value",
        "Churn Label",
        "Churn Score",
        "Churn Reason"
    ]

    X = df.drop(columns=leakage_cols)
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


def build_preprocessor(X):
    """
    Builds preprocessing pipeline.
    """

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor