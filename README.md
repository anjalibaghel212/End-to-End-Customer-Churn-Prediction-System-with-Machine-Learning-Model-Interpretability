# 📊 Customer Churn Prediction Engine

An end-to-end Machine Learning pipeline built in Python to predict customer churn risk using customer demographics, contract details, and account metrics. The project includes automated preprocessing, model evaluation, feature importance analysis, and real-time inference wrappers.

---

## 🎯 Problem Statement
Acquiring new customers is 5x to 25x more expensive than retaining existing ones. Subscription-based businesses need proactive systems to identify high-risk accounts before they churn, enabling targeted retention campaigns.

---

## 💡 Solution Overview
* **Predictive Modeling:** Uses a tuned **Random Forest Classifier** wrapped in a Scikit-Learn `Pipeline` to prevent data leakage.
* **Automated Preprocessing:** Handles numeric scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) seamlessly.
* **Explainable Output:** Extracts feature importances to highlight key churn drivers (e.g., tenure length, contract types, monthly charges).
* **Inference Pipeline:** Exposes a scoring function that converts individual customer payloads into actionable churn probabilities.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `ColumnTransformer`, `Pipeline`)
* **Evaluation Metrics:** ROC-AUC Score, Classification Report, Confusion Matrix

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/customer-churn-prediction.git](https://github.com/YOUR_USERNAME/customer-churn-prediction.git)
cd customer-churn-prediction
