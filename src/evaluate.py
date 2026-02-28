from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates trained model with custom threshold.
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

    return y_prob