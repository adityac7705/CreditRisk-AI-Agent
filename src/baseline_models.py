import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_baselines() :
    if not os.path.exists(TRAIN_PATH) :
        print("Error: Processed data not found. Run smote_balancing.py first")
        return
    
    os.makedirs(MODELS_DIR, exist_ok = True)

    print("*---*Loading SMOTE Balanced Data*---*")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("target", axis = 1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis = 1)
    y_test = test_df["target"]

    print(f"Training Features: {X_train.shape[1]}")
    print(f"Training Samples: {len(X_train)}")

    models = {
        "Logistic_Regression" : LogisticRegression(max_iter = 2000, random_state = 42),
        "Random_Forest" : RandomForestClassifier(n_estimators = 100, random_state = 42),
        "XGBoost" : XGBClassifier(use_label_encoder = False, eval_metric = "logloss", random_state = 42)
    }

    print("*---*Starting Baseline Training*---*")
    results = {}

    for name, model in models.items() :
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"Name : {name}, Accuracy : {acc : .2%}")

        save_path = os.path.join(MODELS_DIR, f"{name}_baseline.pkl")
        joblib.dump(model, save_path)
        print(f"Saved to : {save_path}")

    print("\nPerformance Report")
    best_model_name = max(results, key = results.get)

    for name, acc in results.items() :
        print(f"{name.ljust(20)} : {acc : .2%}")

    print(f"Best Baseline Model : {best_model_name} ({results[best_model_name] : .2%})")
    print("All baseline models saved successfully...")

if __name__ == "__main__" :
    train_baselines()
