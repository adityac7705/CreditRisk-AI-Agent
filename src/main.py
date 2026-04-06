from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from agent_logic import CreditAgent

app = FastAPI(title="CreditAgent AI Advisor")
agent = CreditAgent()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"],
)

class ApplicantData(BaseModel):
    loan_amnt: float
    annual_inc: float
    dti: float
    revol_util: float
    years_left: float       
    job_security: str       
    loan_purpose: str 

@app.get("/")
async def root():
    return {"status": "online"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.post("/predict")
async def predict(data: ApplicantData):
    try:
        applicant_dict = data.model_dump()
        
        # Hard Banking Policy Ceilings based on Loan Purpose
        limits = {"personal": 0.5, "home": 0.8, "education": 1.0}
        ceiling = limits.get(data.loan_purpose, 0.5)
        
        policy_violation = None
        if data.loan_amnt > (data.annual_inc * ceiling):
            policy_violation = f"the loan amount requested exceeds our standard {int(ceiling*100)}% income limit for a {data.loan_purpose} loan"

        ai_result = agent.evaluate_applicant(applicant_dict, policy_violation)
        return ai_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)