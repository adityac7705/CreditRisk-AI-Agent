# Load data from a huge 1.1GB file containing 1000000 records to a smaller file consisting of 100000 records.
import pandas as pd
import os

# This is done to ensure that the OS used doesn't affect the working of this program.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "loan.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_subset.csv")

def load_data() :
    # Returns None if file is not present at given location
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: File not found at {RAW_DATA_PATH}")
        return None

    df = pd.read_csv(RAW_DATA_PATH, nrows = 100000, low_memory = False)

    # We do not consider loans that are in progress (i.e. loans with status showing "Current") for our project.
    valid_statuses = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(valid_statuses)].copy()

    # We create a new column (target) in our csv file. From the raw data, we only select loans that are either "Fully Paid" (denoted by 0) or "Charged Off" or "Default" (both denoted by 1). The target column has only 0 - 1 values because AI understands numerical values not textual values.
    df["target"] = df["loan_status"].map({
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1
    })

    # Important columns from loan approval point of view.
    important_cols = [
        'target', 'loan_amnt', 'term', 'int_rate', 'installment', 
        'grade', 'emp_length', 'home_ownership', 'annual_inc', 
        'verification_status', 'dti', 'delinq_2yrs', 'revol_util'
    ]
    df = df[important_cols]

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok = True)

    # Loads the important data into new file named "loan_subset.csv"
    data = load_data()
    if data is not None:
        data.to_csv(PROCESSED_DATA_PATH, index = False)
        