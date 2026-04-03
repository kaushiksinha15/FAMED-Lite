import pandas as pd
import numpy as np
import os

def clean_and_split_dataset(file_path, name):
    df = pd.read_csv(file_path)
    
    # 1. Standardize column names to snake_case
    # Insert _ before capital letters then lower string, or just handle manually
    import re
    def to_snake_case(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower().replace('__', '_')
    
    df.columns = [to_snake_case(c) for c in df.columns]
    
    # 2. Drop rows with more than 20% missing values
    threshold = int((1 - 0.2) * df.shape[1])
    initial_rows = len(df)
    df = df.dropna(thresh=threshold)
    dropped_rows = initial_rows - len(df)
    print(f"{name}: Dropped {dropped_rows} rows due to >20% missing values.")
    
    # 3. Fill remaining nulls with column median
    # Select only numeric, though all ours are numeric
    df = df.fillna(df.median())
    
    # 4. Split dataset 50/50 by row index
    mid_idx = len(df) // 2
    site_a = df.iloc[:mid_idx]
    site_b = df.iloc[mid_idx:]
    
    # 5. Save the 6 files into data/processed
    os.makedirs('data/processed', exist_ok=True)
    site_a.to_csv(f'data/processed/{name}_a.csv', index=False)
    site_b.to_csv(f'data/processed/{name}_b.csv', index=False)
    print(f"✅ Saved clean {name} (Site A: {len(site_a)}, Site B: {len(site_b)})")

def run_pipeline():
    # Execute for all three modalities
    if not os.path.exists('data/raw/heart.csv'):
        print("Run data_generator.py first!")
        return
    clean_and_split_dataset('data/raw/heart.csv', 'heart')
    clean_and_split_dataset('data/raw/diabetes.csv', 'diabetes')
    clean_and_split_dataset('data/raw/sepsis_survival.csv', 'sepsis')

if __name__ == "__main__":
    run_pipeline()
