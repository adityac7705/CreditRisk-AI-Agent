# Preprocessing of acquired data
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_subset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "final_model_data.csv")

def preprocess_data() :
    if not os.path.exists(INPUT_PATH) :
        print(f"Error: File not found at {INPUT_PATH} please re-run data_loader.py")
        return None
    
    print("*---*Loading Data for Preprocessing*---*")
    df = pd.read_csv(INPUT_PATH)

    print("1. Converting 'term' and 'emp_length' to numbers...")
    #AI cannot understand textual data like "36 months", hence we convert it to int value.
    df["term"] = df["term"].str.replace(" months", "").str.strip().astype(int)
    df["emp_length"] = df["emp_length"].str.extract(r"(\d+)", expand = False).fillna(0).astype(int)

    print("2. Filling missing values with Median...")
    #Median is used since it is not affected by outliers.
    df["annual_inc"] = df["annual_inc"].fillna(df["annual_inc"].median())
    df["dti"] = df["dti"].fillna(df["dti"].median())
    df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())

    print("3. Encoding categorical values (Grade, Home, Verification)...")
    # Convert textual information into binary.
    categorical_cols = ["grade", "home_ownership", "verification_status"]
    df = pd.get_dummies(df, columns = categorical_cols, drop_first = True)

    print(f"Preprocessing Complete! Final Shape {df.shape}")
    return df

if __name__ == "__main__" :
    df_clean = preprocess_data()

    if df_clean is not None :
        df_clean.to_csv(OUTPUT_PATH, index = False)
        print(f"Saved clean data to {OUTPUT_PATH}")
        