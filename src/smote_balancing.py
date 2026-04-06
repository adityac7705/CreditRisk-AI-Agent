import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "final_model_data.csv")
OUTPUT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
OUTPUT_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")

def balance_data() :
    if not os.path.exists(INPUT_PATH) :
        print(f"Error: File not found at {INPUT_PATH}. Please run preprocessing.py first.")
        return None
    
    print("*---*Loading Data for Balancing*---*")
    df = pd.read_csv(INPUT_PATH)

    X = df.drop("target", axis = 1)
    y = df["target"]

    print("Original Class Distribution:")
    print(y.value_counts())

    # Split the data before applying SMOTE. This is done so that AI trains on the raw data. We use 20% of the data to test the AI.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    print("Applying SMOTE to training data only...")
    smote = SMOTE(random_state = 42, k_neighbors = 1)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("New Balanced Training Distribution:")
    print(y_train_resampled.value_counts())

    train_df = pd.concat([pd.DataFrame(X_train_resampled, columns = X.columns), pd.DataFrame(y_train_resampled, columns = ["target"])], axis = 1)
    train_df.to_csv(OUTPUT_TRAIN_PATH, index = False)

    test_df = pd.concat([pd.DataFrame(X_test, columns = X.columns), pd.DataFrame(y_test, columns = ["target"])], axis = 1)
    test_df.to_csv(OUTPUT_TEST_PATH, index = False)

    print("Balancing Complete!")
    print(f"Balanced Train Data saved to {OUTPUT_TRAIN_PATH}")
    print(f"Pure Test Data saved to {OUTPUT_TEST_PATH}")

if __name__ == "__main__" :
    balance_data()
