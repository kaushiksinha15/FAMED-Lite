import os, sys, joblib
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))
from fusion import get_meta_features

enc_heart = joblib.load('models/enc_heart.pkl')
enc_diabetes = joblib.load('models/enc_diabetes.pkl')
meta_learner = joblib.load('models/meta_learner.pkl')

heart_b = pd.read_csv('data/processed/heart_b.csv')
diabetes_b = pd.read_csv('data/processed/diabetes_b.csv')
valid_patients = sorted(list(set(heart_b['patient_id']).intersection(set(diabetes_b['patient_id']))))

print("Testing the 20 Dropdown Patients Sepsis Risks:")
critical_count = 0
safe_count = 0

for pid in valid_patients[:20]:
    patient_h = heart_b[heart_b['patient_id'] == pid].drop(columns=['patient_id'])
    patient_d = diabetes_b[diabetes_b['patient_id'] == pid].drop(columns=['patient_id'])
    
    meta_x = get_meta_features([enc_heart, enc_diabetes], [patient_h, patient_d], mask=[True, True])
    risk = meta_learner.predict_proba(meta_x)[0]
    
    status = "🚨 CRITICAL" if risk > 0.70 else "✅ SAFE"
    
    if risk > 0.70: critical_count += 1
    else: safe_count += 1
    
    print(f"Patient {pid}: Risk = {risk:.4f} ({status})")

print(f"\nFinal Tally: {critical_count} CRITICAL, {safe_count} SAFE")
