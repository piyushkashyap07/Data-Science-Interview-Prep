# Day 36 - End-to-End ML Project: Customer Churn
 
 **Project:** Predicting Customer Churn (Binary Classification) using Telco Dataset.
 **Focus:** EDA, Handling Imbalance (SMOTE), Pipeline, Model Selection, Evaluation, and API Deployment.
 
 ---
 
 ## 1. Problem Statement
 
 **Business Goal:** Identify customers likely to cancel their subscription (Churn = Yes) so the marketing team can send them a retention offer.
 **Metric:** Recall (Minimize False Negatives - we don't want to miss a churning customer) or F1-Score. Accuracy is misleading because Churn is rare (e.g., 20%).
 
 ---
 
 ## 2. Data Preprocessing & EDA
 
 **Key Steps:**
 1. **Data Cleaning:** Convert `TotalCharges` to numeric (handle errors).
 2. **EDA:** 
    - Churn Rate is ~26%. (Imbalanced).
    - Fiber Optic users churn more than DSL.
    - Monthly Contracts churn way more than Year-long contracts.
 3. **Encoding:**
    - Label Encoder: `Gender`, `Partner` (Binary).
    - One-Hot Encoder: `InternetService`, `PaymentMethod` (Multi-class).
 
 ```python
 import pandas as pd
 from sklearn.preprocessing import LabelEncoder
 
 df = pd.read_csv('telco_churn.csv')
 df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
 df.dropna(inplace=True)
 
 df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
 ```
 
 ---
 
 ## 3. Handling Class Imbalance
 
 **Technique: SMOTE (Synthetic Minority Over-sampling Technique)**
 - Instead of duplicating minority samples (Over-sampling), creates *new synthetic* points between neighbors.
 
 ```python
 from imblearn.over_sampling import SMOTE
 
 X = df.drop('Churn', axis=1)
 y = df['Churn']
 
 smote = SMOTE(random_state=42)
 X_res, y_res = smote.fit_resample(X, y)
 # Now class 0 and class 1 counts are equal.
 ```
 
 ---
 
 ## 4. Model Training (XGBoost)
 
 **Why XGBoost?** Handles tabular data best, robust to outliers, handles missing values internally.
 
 ```python
 import xgboost as xgb
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import classification_report
 
 X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
 
 model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
 model.fit(X_train, y_train)
 
 # Evaluate
 preds = model.predict(X_test)
 print(classification_report(y_test, preds))
 ```
 
 ---
 
 ## 5. Model Inference Pipeline
 
 Save the model and encoders for production.
 
 ```python
 import joblib
 
 # Save artifacts
 joblib.dump(model, 'churn_model.pkl')
 # Note: In real life, save your ColumnTransformer/Scalers too!
 ```
 
 ---
 
 ## 6. Deployment (FastAPI Stub)
 
 Create a `main.py` file to serve the model.
 
 ```python
 from fastapi import FastAPI
 import joblib
 import pandas as pd
 
 app = FastAPI()
 model = joblib.load('churn_model.pkl')
 
 @app.post("/predict")
 def predict_churn(data: dict):
     # Convert JSON to DataFrame
     df_input = pd.DataFrame([data])
     
     # Preprocessing steps (Must match training!)
     # ... applying encoders ...
     
     prediction = model.predict(df_input)
     probability = model.predict_proba(df_input)[0][1]
     
     return {
         "churn": int(prediction[0]),
         "probability": float(probability)
     }
 ```
 
 ---
 
 ## 7. Interview Questions on this Project
 
 **Q: Why did you choose SMOTE over Random Oversampling?**
 A: Random oversampling emphasizes outliers and causes overfitting to specific points. SMOTE generalizes the decision boundary by interpolating.
 
 **Q: Your model has 90% Accuracy but only 0.4 Recall. Is it good?**
 A: No. For churn, missing a leaver is expensive. I would lower the probability threshold (e.g., predict Churn if prob > 0.3) to increase Recall, even if Precision drops.
 
 **Q: How do you handle a new Category in 'PaymentMethod' appearing in Production?**
 A: During training, use `OneHotEncoder(handle_unknown='ignore')`. It will produce all zeros for that column during inference, preventing a crash.
 
 ---
 
 ## Key Takeaways
 
 - **Business Value First:** Define the right metric (Recall vs Precision).
 - **Data Leakage:** Split Train/Test *before* SMOTE.
 - **Reproducibility:** Use Pipelines and consistent random seeds.
 
 **Next:** [Day 37 - End-to-End NLP Project](../Day-37/README.md)
