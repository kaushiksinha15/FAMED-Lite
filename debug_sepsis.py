import sys
import os
import joblib
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
patient_list = valid_patients[:20]

for patient_id in patient_list:
    patient_h = heart_b[heart_b['patient_id'] == patient_id].drop(columns=['patient_id']).copy()
    patient_d = diabetes_b[diabetes_b['patient_id'] == patient_id].drop(columns=['patient_id']).copy()
    
    meta_x = get_meta_features([enc_heart, enc_diabetes], [patient_h, patient_d], mask=[True, True])
    risk_score = meta_learner.predict_proba(meta_x)[0]
    print(f"Patient {patient_id}: {risk_score:.3f}")
