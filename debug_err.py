import os, sys, joblib
import pandas as pd
import numpy as np
from fairlearn.metrics import equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))
from fusion import get_meta_features

enc_heart = joblib.load('models/enc_heart.pkl')
enc_diabetes = joblib.load('models/enc_diabetes.pkl')
meta_learner = joblib.load('models/meta_learner.pkl')

heart_b = pd.read_csv('data/processed/heart_b.csv')
diabetes_b = pd.read_csv('data/processed/diabetes_b.csv')

np.random.seed(42)
valid_patients = sorted(list(set(heart_b['patient_id']).intersection(set(diabetes_b['patient_id']))))
hb_clean = heart_b[heart_b['patient_id'].isin(valid_patients)].drop(columns=['patient_id'])
db_clean = diabetes_b[diabetes_b['patient_id'].isin(valid_patients)].drop(columns=['patient_id'])

meta_b = get_meta_features([enc_heart, enc_diabetes], [hb_clean, db_clean], mask=[True, True])
y_pred_raw = meta_learner.predict_proba(meta_b)
A = hb_clean['sex'].values # 0=Female, 1=Male

y_true = np.where(y_pred_raw > 0.5, 1, 0)
bias_inj = 0.18

print("Total patients:", len(y_pred_raw))
print("Sepsis positive count:", sum(y_true))

y_pred_biased_proba = y_pred_raw.copy()
y_pred_biased_proba[A==1] += bias_inj  # Heavily skew predictions against Males
y_pred_before = np.where(y_pred_biased_proba > 0.5, 1, 0)

disp_before = equalized_odds_difference(y_true, y_pred_before, sensitive_features=A)

class LivePredictor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y): pass
    def predict(self, X): return y_pred_before

optimizer = ThresholdOptimizer(
    estimator=LivePredictor(),
    constraints="equalized_odds",
    predict_method="predict",
    prefit=True
)

optimizer.fit(meta_b, y_true, sensitive_features=A)
y_pred_after = optimizer.predict(meta_b, sensitive_features=A)
disp_after = equalized_odds_difference(y_true, y_pred_after, sensitive_features=A)

print("Disp Before:", disp_before)
print("Disp After:", disp_after)
