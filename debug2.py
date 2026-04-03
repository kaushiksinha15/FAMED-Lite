import os, sys, joblib
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))
from fusion import get_meta_features

enc_heart = joblib.load('models/enc_heart.pkl')
enc_diabetes = joblib.load('models/enc_diabetes.pkl')
meta_learner = joblib.load('models/meta_learner.pkl')

patient_h = pd.DataFrame({
    'age': [85], 'sex': [0], 'chest_pain_type': [0], 'resting_bp': [180],
    'cholesterol': [240], 'fasting_bs': [1], 'resting_ecg': [1], 
    'max_hr': [200], 'exercise_angina': [1], 'oldpeak': [1.0], 'st_slope': [1]
})
patient_d = pd.DataFrame({
    'pregnancies': [0], 'glucose': [280], 'blood_pressure': [50],
    'skin_thickness': [25], 'insulin': [250.0], 'bmi': [32.0], 'diabetes_pedigree_function': [0.5]
})

meta_x = get_meta_features([enc_heart, enc_diabetes], [patient_h, patient_d], mask=[True, True])
risk_score = meta_learner.predict_proba(meta_x)[0]

print("Default Manual Patient Risk:", risk_score)
