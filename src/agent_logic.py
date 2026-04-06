import pandas as pd
import os
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_ensemble_model.pkl")
COLS_PATH = os.path.join(BASE_DIR, "models", "model_columns.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

class CreditAgent:
    def __init__(self):
        # Load trained machine learning artifacts
        self.model = joblib.load(MODEL_PATH)
        self.columns = joblib.load(COLS_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        # Calibrated risk thresholds
        self.reject_threshold = 0.65  
        self.review_threshold = 0.35  
        
        # Load SHAP explainer for AI Explainability (XAI)
        xgb_model = self.model.named_estimators_["xgb"]
        self.explainer = shap.TreeExplainer(xgb_model)

    def get_human_explanation(self, scaled_input, status):
        shap_values = self.explainer.shap_values(scaled_input)
        feature_importance = dict(zip(self.columns, shap_values[0]))

        human_phrases = {
            "dti": "how your monthly obligations balance against your take-home pay",
            "annual_inc": "your overall annual earning potential",
            "loan_amnt": "the specific size of the loan you've requested",
            "revol_util": "the current debt levels on your revolving credit lines",
            "installment": "the projected monthly repayment amount"
        }

        is_negative_outcome = status in ["REJECTED", "MANUAL REVIEW"]

        if is_negative_outcome:
            # Find the feature that increased the risk the most
            top_feature = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[0][0]
            sentiment = "is currently outside our standard risk tolerance"
        else:
            # Find the feature that lowered the risk the most
            top_feature = sorted(feature_importance.items(), key=lambda x: x[1])[0][0]
            sentiment = "shows excellent financial stability"

        feature_name = human_phrases.get(top_feature, top_feature.replace("_", " "))
        return f"{feature_name} {sentiment}"

    def evaluate_applicant(self, applicant_data, policy_violation=None):
        # 1. Prepare Data and Scale
        df = pd.DataFrame([applicant_data])
        model_df = df.reindex(columns=self.columns, fill_value=0)
        scaled_input = self.scaler.transform(model_df)

        # 2. Initial AI Risk Assessment (Base quantitative ML score)
        base_risk_prob = self.model.predict_proba(scaled_input)[0][1]
        
        # 3. Hybrid Heuristic Overlay (Behavioral & Stability Rules)
        utilization = applicant_data.get('revol_util', 0)
        years_left = float(applicant_data.get('years_left', 20))
        job_sec = applicant_data.get('job_security', 'medium')
        
        penalty = 0
        
        # High revolving debt penalty
        if utilization > 80: penalty += 0.30
        
        # The Retirement Bypass logic
        if years_left == 0 and job_sec == 'high':
            penalty -= 0.10  # Bonus for guaranteed pension income
        else:
            # Active Employee Logic
            if years_left < 5: penalty += 0.25
            elif years_left < 10: penalty += 0.10
            
            # Job Volatility
            if job_sec == 'low': penalty += 0.15
            elif job_sec == 'high': penalty -= 0.10 

        final_risk_prob = base_risk_prob + penalty
        
        # 4. Hard Policy Penalty
        if policy_violation:
            final_risk_prob += 0.40

        # Ensure probability stays cleanly bounded between 0 and 1
        final_risk_prob = max(0.0, min(final_risk_prob, 1.0))

        # 5. Determine Status
        if final_risk_prob >= self.reject_threshold:
            status, color = "REJECTED", "#ef4444"
        elif final_risk_prob >= self.review_threshold:
            status, color = "MANUAL REVIEW", "#f59e0b"
        else:
            status, color = "APPROVED", "#10b981"

        reason = self.get_human_explanation(scaled_input, status)
        risk_score = round(final_risk_prob * 100, 2)

        # 6. Recommendation Phrasing
        if status == "REJECTED":
            recommendation = f"I've reviewed your profile, but we cannot proceed. The primary factor is that {reason}. " + \
                             (f"Additionally, {policy_violation}." if policy_violation else "Current job stability or retirement horizons also add significant risk.")
        elif status == "MANUAL REVIEW":
            recommendation = f"Your application is in a gray area. While {reason}, your career runway and employment sector require a manual second-look."
        else:
            if years_left == 0:
                recommendation = f"Congratulations! Your profile stands out because {reason}. Your pension income is verified as secure."
            else:
                recommendation = f"Congratulations! Your profile stands out because {reason}. You have a stable career runway for this loan term."

        return {"status": status, "risk_score": risk_score, "recommendation": recommendation, "color": color}