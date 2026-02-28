from src.predict import load_model, predict_customer


def main():
    model = load_model()

    # Example customer (you can modify later)
    sample_customer = {
        "CustomerID": "0000-TEST",
        "Count": 1,
        "Country": "United States",
        "State": "California",
        "City": "Los Angeles",
        "Zip Code": 90001,
        "Lat Long": "34.0522,-118.2437",
        "Latitude": 34.0522,
        "Longitude": -118.2437,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 2,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.5,
        "Total Charges": 170.5,
        "CLTV": 3500
    }

    result = predict_customer(model, sample_customer, threshold=0.3)

    print("\nPrediction Result:")
    print(result)


if __name__ == "__main__":
    main()