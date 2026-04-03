import pandas as pd
from sklearn.metrics import roc_auc_score
from modules.fusion import train_modality_encoder, get_meta_features, FamedMetaLearner
import joblib
import json
import os

def run_fusion_training():
    print("Loading processed Site A data for training...")
    heart_a = pd.read_csv('data/processed/heart_a.csv')
    diabetes_a = pd.read_csv('data/processed/diabetes_a.csv')
    sepsis_a = pd.read_csv('data/processed/sepsis_a.csv')

    # Ensure all data has the exact same patient order
    df_a = heart_a.merge(diabetes_a, on='patient_id').merge(sepsis_a, on='patient_id')
    
    # 1. Prepare modalities
    # Modality 1: Heart features
    heart_features = heart_a.drop(columns=['patient_id']).columns
    X_heart = df_a[heart_features]
    
    # Modality 2: Diabetes features
    diabetes_features = diabetes_a.drop(columns=['patient_id']).columns
    X_diabetes = df_a[diabetes_features]

    y_train = df_a['sepsis_survival_status']
    
    # 2. Train encoders
    print("Training encoders for Novelty 1...")
    enc_heart = train_modality_encoder(X_heart, y_train, "Heart")
    enc_diabetes = train_modality_encoder(X_diabetes, y_train, "Diabetes")
    
    # 3. Create meta features and train stacker
    # Use full mask for training [True, True]
    X_list = [X_heart, X_diabetes]
    encoders = [enc_heart, enc_diabetes]
    
    meta_X_train = get_meta_features(encoders, X_list, mask=[True, True])
    
    meta_learner = FamedMetaLearner()
    meta_learner.fit(meta_X_train, y_train)
    
    # Save the models
    os.makedirs('models', exist_ok=True)
    joblib.dump(enc_heart, 'models/enc_heart.pkl')
    joblib.dump(enc_diabetes, 'models/enc_diabetes.pkl')
    joblib.dump(meta_learner, 'models/meta_learner.pkl')
    
    # 4. Evaluate on holdout Site B (genuine out-of-sample test set)
    print("Evaluating on holdout Site B...")
    heart_b = pd.read_csv('data/processed/heart_b.csv')
    diabetes_b = pd.read_csv('data/processed/diabetes_b.csv')
    sepsis_b = pd.read_csv('data/processed/sepsis_b.csv')
    df_b = heart_b.merge(diabetes_b, on='patient_id').merge(sepsis_b, on='patient_id')

    X_heart_b = df_b[heart_features]
    X_diabetes_b = df_b[diabetes_features]
    y_test = df_b['sepsis_survival_status']

    # Mask 1: Full modalities (holdout)
    scores_full = meta_learner.predict_proba(get_meta_features(encoders, [X_heart_b, X_diabetes_b], [True, True]))
    auc_full = roc_auc_score(y_test, scores_full)

    # Mask 2: Missing Modality 2 (holdout)
    scores_missing = meta_learner.predict_proba(get_meta_features(encoders, [X_heart_b, X_diabetes_b], [True, False]))
    auc_missing = roc_auc_score(y_test, scores_missing)

    print(f"✅ Holdout Fusion AUC (Both modalities present): {auc_full:.4f}")
    print(f"✅ Holdout Missing-Modality AUC (Diabetes masked): {auc_missing:.4f}")
    
    # Logs
    with open('logs/benchmark.json', 'r') as f:
        metrics = json.load(f)
    
    metrics['famed_auc_full'] = auc_full
    metrics['famed_auc_missing'] = auc_missing
    with open('logs/benchmark.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    run_fusion_training()
