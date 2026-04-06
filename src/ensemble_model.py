import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_ensemble() :
    if not os.path.exists(TRAIN_PATH) :
        print("Error: File not found!")
        return
    
    print("*---*Loading SMOTE Data for Ensemble*---*")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("target", axis = 1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis = 1)
    y_test = test_df["target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok = True)
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "model_columns.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    clf1 = LogisticRegression(max_iter = 3000, random_state = 42, class_weight = {0 : 1, 1 : 500})
    clf2 = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight = {0 : 1, 1 : 500})
    clf3 = XGBClassifier(use_label_encoder = False, eval_metric = "logloss", random_state = 42, scale_pos_weight = 500)

    print("Building Voting Classifier...")
    ensemble_model = VotingClassifier(
        estimators = [
            ("lr", clf1),
            ("rf", clf2),
            ("xgb", clf3)
        ],
        voting = "soft",
        weights = [1, 1, 20]
    )

    print("Training Ensemble Model...")
    ensemble_model.fit(X_train_scaled, y_train)
    print("Model Training Complete!")

    probs = ensemble_model.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.04
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)

    print("Final Ensemble Report :")
    print(f"Ensemble Accuracy : {acc : .2%}")
    print("Classification Report :")
    print(classification_report(y_test, preds))
    print("Confusion Matrix :")
    print(confusion_matrix(y_test, preds))

    save_path = os.path.join(MODELS_DIR, "final_ensemble_model.pkl")
    joblib.dump(ensemble_model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__" :
    train_ensemble()
