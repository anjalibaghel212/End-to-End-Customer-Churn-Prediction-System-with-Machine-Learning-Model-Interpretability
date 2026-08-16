import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ==========================================
# 1. SYNTHETIC DATASET CREATION
# ==========================================
def generate_synthetic_data(n_samples=1000, seed=42):
    """Generates realistic customer churn dataset."""
    np.random.seed(seed)
    
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples)
    tech_support = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
    
    # Define churn probability logic based on domain assumptions
    churn_prob = (
        0.3 * (contract_type == 'Month-to-month') 
        + 0.25 * (monthly_charges > 70) 
        - 0.2 * (tenure > 24) 
        - 0.15 * (tech_support == 'Yes') 
        + np.random.normal(0, 0.1, size=n_samples)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly_charges,
        'Contract_Type': contract_type,
        'Payment_Method': payment_method,
        'Tech_Support': tech_support,
        'Churn': churn
    })
    return df

# Load data
df = generate_synthetic_data()
X = df.drop(columns=['Churn'])
y = df['Churn']

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
numeric_features = ['Tenure_Months', 'Monthly_Charges']
categorical_features = ['Contract_Type', 'Payment_Method', 'Tech_Support']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 3. MODEL TRAINING PIPELINE
# ==========================================
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model_pipeline.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 5. FEATURE IMPORTANCE ANALYSIS
# ==========================================
ohe_feature_names = model_pipeline.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .get_feature_names_out(categorical_features)

all_feature_names = numeric_features + list(ohe_feature_names)
importances = model_pipeline.named_steps['classifier'].feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n=== TOP FEATURE IMPORTANCES ===")
print(feature_imp_df)

# ==========================================
# 6. INFERENCE FUNCTION FOR NEW CUSTOMERS
# ==========================================
def predict_new_customer(customer_dict):
    """Function to score new customer data in real time."""
    input_df = pd.DataFrame([customer_dict])
    pred = model_pipeline.predict(input_df)[0]
    prob = model_pipeline.predict_proba(input_df)[0][1]
    return {"Churn_Predicted": bool(pred), "Churn_Probability": round(prob, 4)}

# Example inference
sample_customer = {
    'Tenure_Months': 3,
    'Monthly_Charges': 95.50,
    'Contract_Type': 'Month-to-month',
    'Payment_Method': 'Electronic check',
    'Tech_Support': 'No'
}

print("\n=== SAMPLE CUSTOMER RISK SCORE ===")
print(predict_new_customer(sample_customer))
