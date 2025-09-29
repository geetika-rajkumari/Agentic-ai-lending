from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Risk Prediction")

class PersonalInfo(BaseModel):
    name: str
    age: int
    employment_type: str
    monthly_income: float
    cibil_score: int
    existing_emi: float
    bank_balance: float
    loan_amount: float
    default_in_past: bool
    
class CreditCheckResponse(BaseModel):
    status: str

def evaluate_credit_risk(personal_info: PersonalInfo) -> CreditCheckResponse:
    if personal_info.age < 18 or personal_info.age > 60:
        return CreditCheckResponse(status="Rejected")
    if personal_info.employment_type not in ["Salaried", "Self-Employed"]:
        return CreditCheckResponse(status="Rejected")
    if personal_info.monthly_income < 15000:
        return CreditCheckResponse(status="Rejected")
    if personal_info.cibil_score < 700:
        return CreditCheckResponse(status="Rejected")
    if personal_info.bank_balance < 100000:
        return CreditCheckResponse(status="Rejected")
    foir = personal_info.existing_emi / personal_info.monthly_income
    if foir > 0.4:
        return CreditCheckResponse(status="Rejected")
    if personal_info.loan_amount > 20 * personal_info.monthly_income:
        return CreditCheckResponse(status="Rejected")
    if personal_info.default_in_past:
        return CreditCheckResponse(status="Rejected")


    return CreditCheckResponse(status="Approved")


@app.post("/api/v1/credit-check", response_model=CreditCheckResponse)
def check_credit_risk(payload: PersonalInfo):
    status = evaluate_credit_risk(payload)
    return CreditCheckResponse(status=status.status)
