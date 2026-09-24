import pandas as pd

def load_and_clean(raw_path="../data/raw/Telco-Customer-Churn.csv"):
    df = pd.read_csv(raw_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

if __name__ == "__main__":
    df = load_and_clean()
    df.to_csv("../data/processed/telco_churn_clean.csv", index=False)
    print(f"Cleaned data saved: {df.shape}")