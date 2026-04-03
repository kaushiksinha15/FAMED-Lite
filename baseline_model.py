import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import os

def run_baseline():
    print("Training baseline Logistic Regression on Site A (Heart modality only)...")
    # Load Site A data
    try:
        heart_a = pd.read_csv('data/processed/heart_a.csv')
        sepsis_a = pd.read_csv('data/processed/sepsis_a.csv')
    except Exception as e:
        print("Please run data_processor.py first.")
        return

    # Merge by patient_id to align the features with the target
    merged = pd.merge(heart_a, sepsis_a, on='patient_id', how='inner')
    
    # Target column is 'sepsis_survival_status'
    # Drop IDs and non-predictive columns
    X = merged.drop(columns=['patient_id', 'sepsis_survival_status', 'hospital_duration'])
    y = merged['sepsis_survival_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"🔥 Baseline AUC (Single Modality - Logistic Regression): {auc:.4f}")
    
    # Save the score to a benchmark file for reference
    os.makedirs('logs', exist_ok=True)
    with open('logs/benchmark.json', 'w') as f:
        json.dump({"baseline_auc": auc}, f)

if __name__ == "__main__":
    run_baseline()
