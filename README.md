# Customer Churn Prediction (XGBoost)

## 📌 Overview
This project builds a modular, production-ready machine learning pipeline to predict telecom customer churn using XGBoost.

The system includes:
- Clean project structure
- Data leakage prevention
- Feature preprocessing pipeline
- Threshold-based decision control
- Model persistence
- Standalone inference script

---

## 📊 Dataset
Telco Customer Churn dataset with demographic and service usage features.

Target:
`Churn Value` (0 = No churn, 1 = Churn)

---

## 🛠 Tech Stack
- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Git

---

## 🏗 Project Structure

customer-churn-ml/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── predict.py
├── requirements.txt
└── README.md

---

## 📈 Model Performance

**ROC-AUC:** ~0.85  
**Recall (Threshold = 0.3):** ~0.74  
**Precision (Threshold = 0.3):** ~0.54  

The model supports adjustable thresholds to optimize business trade-offs between recall and precision.

---

## 🚀 How To Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt