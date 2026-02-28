from src.data_loader import load_data
from src.preprocessing import split_data, build_preprocessor
from src.train import train_xgboost
from src.evaluate import evaluate_model


def main():
    df = load_data("data/telco_churn.xlsx")

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    model = train_xgboost(preprocessor, X_train, y_train)

    evaluate_model(model, X_test, y_test, threshold=0.5)
    evaluate_model(model, X_test, y_test, threshold=0.3)


if __name__ == "__main__":
    main()