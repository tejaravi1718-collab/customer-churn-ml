# Customer Churn Prediction (XGBoost)

## 📌 Problem Statement
Predict customer churn using telecom customer data.

## 📊 Dataset
Telco Customer Churn dataset containing demographic and service usage features.

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib

## 🏗️ Project Structure
- Modular pipeline (data loading, preprocessing, training, evaluation)
- Model persistence
- Threshold tuning
- Standalone prediction script

## 📈 Model Performance
- ROC-AUC: ~0.85
- Recall (0.3 threshold): ~0.74
- Precision (0.3 threshold): ~0.54

## 🚀 How to Run

### Train Model
```bash
python main.py