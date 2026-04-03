import pandas as pd
from modules.drift_monitor import check_drift

def run_week4():
    print("Testing Data Drift Monitor (Novelty 4)...")
    
    # Site A is reference (training)
    heart_a = pd.read_csv('data/processed/heart_a.csv')
    
    # Site B is current (deployment)
    heart_b = pd.read_csv('data/processed/heart_b.csv')
    
    # Drop identifiers and get raw features
    ref_df = heart_a.drop(columns=['patient_id'])
    curr_df = heart_b.drop(columns=['patient_id'])
    
    # Introduce artificial drift manually to current_df to guarantee a rollback event for the patent claim
    curr_df = curr_df * 50.0  # complete statistical drift
    curr_df['sex'] = 0
    
    print("Comparing Site A (Reference) vs Site B (Deploy) with artificially injected drift...")
    check_drift(ref_df, curr_df, threshold=0.2)
    
if __name__ == "__main__":
    run_week4()
