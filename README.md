# 🤖 Autonomous AI Agent for Credit Risk Assessment

### **An End-to-End Intelligent Loan Underwriting Prototype**

This project is a sophisticated financial AI system developed as part of the **Mumbai University Semester 6 Mini Project (Group A5)**. It transitions from a standard machine learning model to a functional **AI Agent** by combining probabilistic risk prediction with deterministic banking policy enforcement.

---

## 🌟 Key Features

- **Hybrid Decision Engine:** Integrates an **XGBoost Ensemble Model** for pattern recognition with a **Heuristic Reasoning Layer** for institutional safety.
- **Autonomous Underwriting:** Automatically categorizes applicants into `APPROVED`, `REJECTED`, or `MANUAL REVIEW` based on real-time risk calculations.
- **Policy Guardrails:** Implements hard stops for high Credit Utilization (>85%) and Debt-to-Income (DTI) ratios, ensuring the AI respects banking regulations regardless of income levels.
- **Explainable AI (XAI):** Generates a professional, dynamic **Risk Dossier (PDF)**, providing transparency and an "Advisory Roadmap" for every applicant.
- **Advanced Data Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance and ensure robust prediction accuracy.

---

## 🏗️ System Architecture

The system is divided into four distinct modules as per the 12-week roadmap:

1.  **Perception (Frontend):** Responsive UI built with **Tailwind CSS** handling data entry and live formatting.
2.  **Inference (ML Core):** An **XGBoost Classifier** that predicts the mathematical probability of default.
3.  **Reasoning (AI Agent):** A Python-based logic controller that weighs ML scores against career runway, job security, and leverage thresholds.
4.  **Action (Reporting):** A real-time feedback loop delivering a verdict and a downloadable improvement roadmap via `jsPDF`.

---

## 📁 Project Structure

```text
├── models/               # Serialized ML artifacts (.pkl)
│   ├── final_ensemble_model.pkl   # The trained XGBoost Brain
│   ├── scaler.pkl                 # Feature scaling weights
│   └── model_columns.pkl          # Feature alignment metadata
├── reports/              # Visual Analytics & EDA
│   └── figures/                   # Correlation heatmaps, DTI distributions
├── src/                  # Source Code
│   ├── main.py                    # FastAPI Backend (The API Gateway)
│   ├── agent_logic.py             # The AI Agent Reasoning Logic
│   ├── ensemble_model.py          # Training & SMOTE implementation
│   ├── preprocessing.py           # Feature engineering pipeline
│   └── index.html                 # AI Agent Web Interface
├── requirements.txt      # Project dependencies
└── .gitignore            # Excludes environment and local data

🚀 Installation & Setup
1. Clone the Repository
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)

    cd YOUR_REPO_NAME

2. Set Up Environment
    pip install -r requirements.txt

3. Run the Agent
Launch the FastAPI server:
    python src/main.py
The server will start at http://127.0.0.1:8000. You can then open src/index.html in your browser to interact with the agent.

📊 Evaluation Metrics
Algorithm: XGBoost (Gradient Boosting Decision Trees)

Baseline Models: Logistic Regression, Random Forest

Optimization: SMOTE for minority class oversampling

Status Thresholds:-
    < 35% Risk: Automated Approval

    35% - 65% Risk: Manual Review Trigger

    > 65% Risk: Automated Rejection

🛠️ Technology Stack
    Layer - Tools
    Language - Python 3.13
    ML/AI - Scikit-Learn, XGBoost, Pandas
    Backend - FastAPI, Uvicorn
    Frontend - Tailwind CSS, JavaScript
    Reporting - jsPDF-AutoTable
```
