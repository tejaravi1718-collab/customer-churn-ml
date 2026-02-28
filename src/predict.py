import joblib
import pandas as pd


def load_model(model_path="models/xgb_model.pkl"):
    """
    Loads trained model pipeline.
    """
    return joblib.load(model_path)


def predict_customer(model, customer_dict, threshold=0.3):
    """
    Predict churn for a single customer.
    """

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame([customer_dict])

    # Predict probability
    prob = model.predict_proba(customer_df)[:, 1][0]

    # Apply threshold
    prediction = int(prob >= threshold)

    return {
        "churn_probability": round(float(prob), 4),
        "prediction": prediction,
    }