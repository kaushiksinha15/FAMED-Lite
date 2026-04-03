import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_patients=2000):
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Heart dataset (Vitals/Cardiac modality)
    # Simulating features from the UCI Heart Disease dataset
    heart_data = pd.DataFrame({
        'Patient_ID': range(num_patients),
        'Age': np.random.randint(25, 85, num_patients),
        'Sex': np.random.choice([0, 1], num_patients), # 0=Female, 1=Male (used for debiasing)
        'ChestPainType': np.random.choice([0, 1, 2, 3], num_patients),
        'RestingBP': np.random.normal(130, 15, num_patients),
        'Cholesterol': np.random.normal(240, 40, num_patients),
        'FastingBS': np.random.choice([0, 1], num_patients, p=[0.8, 0.2]),
        'RestingECG': np.random.choice([0, 1, 2], num_patients),
        'MaxHR': np.random.normal(150, 20, num_patients),
        'ExerciseAngina': np.random.choice([0, 1], num_patients, p=[0.6, 0.4]),
        'Oldpeak': np.random.exponential(1.5, num_patients),
        'ST_Slope': np.random.choice([0, 1, 2], num_patients)
    })
    
    # Injecting NaNs to simulate real-world data missingness (for cleaning step testing)
    mask_25_missing = np.random.rand(num_patients) < 0.25
    heart_data.loc[mask_25_missing, 'Cholesterol'] = np.nan
    mask_15_missing = np.random.rand(num_patients) < 0.15
    heart_data.loc[mask_15_missing, 'MaxHR'] = np.nan
    
    # Highly sparse column to test dropping columns > 20% NaN later, wait, 
    # the spec says: "Drop rows with more than 20% missing values"
    # We will inject random missingness across multiple rows
    for col in ['RestingBP', 'FastingBS', 'ST_Slope']:
        mask = np.random.rand(num_patients) < 0.1
        heart_data.loc[mask, col] = np.nan

    heart_data.to_csv('data/raw/heart.csv', index=False)
    
    # 2. Diabetes dataset (Lab/Metabolic modality)
    # Simulating features from Pima Indians Diabetes Dataset
    diabetes_data = pd.DataFrame({
        'Patient_ID': range(num_patients),
        'Pregnancies': np.random.randint(0, 10, num_patients),
        'Glucose': np.random.normal(120, 30, num_patients),
        'BloodPressure': np.random.normal(70, 15, num_patients), # Diastolic
        'SkinThickness': np.random.normal(25, 10, num_patients),
        'Insulin': np.random.exponential(80, num_patients),
        'BMI': np.random.normal(32, 7, num_patients),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, num_patients)
    })
    
    # Inject NaNs
    for col in ['Glucose', 'Insulin', 'SkinThickness']:
        mask = np.random.rand(num_patients) < 0.15
        diabetes_data.loc[mask, col] = np.nan
        
    diabetes_data.to_csv('data/raw/diabetes.csv', index=False)
    
    # 3. Sepsis Survival dataset (Target / Outcomes)
    # We'll construct a synthetic target which correlates weakly with the data 
    # so the models have something to learn.
    # Higher age, male sex, higher glucose, higher resting BP -> higher risk
    base_risk = (heart_data['Age'] / 80.0) * 0.3 + \
                (heart_data['Sex'] * 0.1) + \
                (diabetes_data['Glucose'].fillna(120) / 200.0) * 0.3 + \
                (heart_data['RestingBP'].fillna(130) / 200.0) * 0.2
    
    # Softer sigmoid (coefficient -5, threshold 0.65) produces ~30% positive rate
    # which is clinically realistic for a high-acuity hospital cohort
    probabilities = 1 / (1 + np.exp(-5 * (base_risk - 0.65)))
    target_outcomes = (np.random.rand(num_patients) < probabilities).astype(int)
    print(f"  Sepsis positive rate: {target_outcomes.mean():.1%}")
    
    sepsis_data = pd.DataFrame({
        'Patient_ID': range(num_patients),
        'Hospital_Duration': np.random.randint(1, 30, num_patients),
        'Sepsis_Survival_Status': target_outcomes # 1 = Survival/Positive event, 0 = Non-survival
    })
    
    # Inject some NaNs on duration
    mask = np.random.rand(num_patients) < 0.1
    sepsis_data.loc[mask, 'Hospital_Duration'] = np.nan
    
    sepsis_data.to_csv('data/raw/sepsis_survival.csv', index=False)
    
    print(f"✅ Generated synthetic datasets for {num_patients} patients at 'data/raw/'")

if __name__ == '__main__':
    generate_synthetic_data()
