import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import shap
import os
from modules.debiasing import train_fairness_optimizer, calculate_disparity
from modules.xai_router import route_explanation

def run_week3():
    print("Testing Debiasing (Novelty 2)...")
    # Load Site A data
    heart_a = pd.read_csv('data/processed/heart_a.csv')
    diabetes_a = pd.read_csv('data/processed/diabetes_a.csv')
    sepsis_a = pd.read_csv('data/processed/sepsis_a.csv')
    df_a = heart_a.merge(diabetes_a, on='patient_id').merge(sepsis_a, on='patient_id')
    
    # Load Models
    enc_heart = joblib.load('models/enc_heart.pkl')
    enc_diabetes = joblib.load('models/enc_diabetes.pkl')
    meta_learner = joblib.load('models/meta_learner.pkl')
    
    # Recreate meta_X
    heart_features = heart_a.drop(columns=['patient_id']).columns
    diabetes_features = diabetes_a.drop(columns=['patient_id']).columns
    
    X_heart = df_a[heart_features]
    X_diabetes = df_a[diabetes_features]
    y_train = df_a['sepsis_survival_status']
    
    from modules.fusion import get_meta_features
    meta_X = get_meta_features([enc_heart, enc_diabetes], [X_heart, X_diabetes], mask=[True, True])
    
    # Sex is in the heart dataset
    sensitive_features = df_a['sex']
    
    # Before debiasing
    y_pred_before = meta_learner.predict(meta_X)
    disp_before = calculate_disparity(y_train, y_pred_before, sensitive_features)
    
    # After debiasing
    optimizer = train_fairness_optimizer(meta_learner, meta_X, y_train, sensitive_features)
    y_pred_after = optimizer.predict(meta_X, sensitive_features=sensitive_features)
    disp_after = calculate_disparity(y_train, y_pred_after, sensitive_features)
    
    print(f"✅ Disparity BEFORE: {disp_before:.4f}")
    print(f"✅ Disparity AFTER:  {disp_after:.4f}")
    
    with open('logs/benchmark.json', 'r') as f:
        metrics = json.load(f)
    metrics['disparity_before'] = disp_before
    metrics['disparity_after'] = disp_after
    with open('logs/benchmark.json', 'w') as f:
        json.dump(metrics, f)
        
    print("Testing XAI Router (Novelty 3)...")
    os.makedirs('logs/plots', exist_ok=True)
    
    # Pick a high risk and low risk patient
    probs_heart = enc_heart.predict_proba(X_heart)[:, 1]
    
    high_risk_idx = probs_heart.argmax()
    low_risk_idx = probs_heart.argmin()
    
    # Explain high risk (fast)
    x_high = X_heart.iloc[[high_risk_idx]]
    res_high = route_explanation(enc_heart, x_high, probs_heart[high_risk_idx], "vitals")
    print(f"Routed high risk to: {res_high['method']}")
    
    # Explain low risk (waterfall)
    x_low = X_heart.iloc[[low_risk_idx]]
    res_low = route_explanation(enc_heart, x_low, probs_heart[low_risk_idx], "vitals")
    print(f"Routed low risk to: {res_low['method']}")
    
    # Save a plot for the patent
    if res_low['method'] == "SHAP_waterfall":
        # Extract the Explanation object for the single instance
        exp = res_low['values'][0]
        plt.figure()
        shap.plots.waterfall(exp, show=False)
        plt.savefig('logs/plots/shap_waterfall.png', bbox_inches='tight')
        print("✅ Saved logs/plots/shap_waterfall.png")

if __name__ == "__main__":
    run_week3()
