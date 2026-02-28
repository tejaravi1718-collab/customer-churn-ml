from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
import os


def train_xgboost(preprocessor, X_train, y_train):
    """
    Builds and trains XGBoost pipeline.
    Saves trained model.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save full pipeline
    joblib.dump(model, "models/xgb_model.pkl")

    return model