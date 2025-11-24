import os
import io
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from PIL import Image
from pypdf import PdfReader
import httpx
import easyocr


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MIN_INCOME_GUIDELINE = 2000


if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Gemini API calls will fail.")


try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None


class BankStatementExtractionResponse(BaseModel):
    raw_text: str

class PayslipExtractionResponse(BaseModel):
    net_income: Optional[float]
    raw_text: str

class ComplianceReport(BaseModel):
    is_compliant: bool
    kyc_match: str
    document_status: str
    income_check: str
    flags: List[str]

class FileResult(BaseModel):
    filename: str
    type: str
    raw_text: str
    analysis: dict

class WorkflowResponse(BaseModel):
    status: str
    applicant_id: str
    files: List[FileResult]
    compliance_report: ComplianceReport


async def extract_text_from_image(file_bytes: bytes) -> str:
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR engine unavailable.")
    image = Image.open(io.BytesIO(file_bytes))
    result = await asyncio.to_thread(reader.readtext, image)
    raw_text = ' '.join([text for (_, text, _) in result])
    return raw_text

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader_pdf = PdfReader(io.BytesIO(file_bytes))
        raw_text = ""
        for page in reader_pdf.pages:
            raw_text += page.extract_text() or ""
        return raw_text
    except Exception as e:
        print(f"PDF parsing error: {e}")
        return ""

async def call_gemini(prompt: str, schema: dict) -> dict:
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": "You are a compliance officer bot."}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.0
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()
                candidate = result.get("candidates", [{}])[0]
                json_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "{}")
                return json.loads(json_text)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")
    return {}


COMPLIANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_compliant": {"type": "boolean"},
        "kyc_match": {"type": "string"},
        "document_status": {"type": "string"},
        "income_check": {"type": "string"},
        "flags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["is_compliant", "kyc_match", "document_status", "income_check", "flags"]
}

async def compliance_agent_gemini(bank_data: Optional[BankStatementExtractionResponse],
                                  payslip_data: Optional[PayslipExtractionResponse],
                                  uploaded_file_types: List[str]) -> ComplianceReport:
    bank_text = bank_data.raw_text if bank_data else ""
    payslip_text = payslip_data.raw_text if payslip_data else ""
    net_income = str(payslip_data.net_income) if payslip_data and payslip_data.net_income else "0"
    files_text = ", ".join(uploaded_file_types)

    prompt = (
        f"Applicant Documents: {files_text}\n"
        f"Bank Statement Text: {bank_text}\n"
        f"Payslip Text: {payslip_text}\n"
        f"Payslip Net Income: {net_income}\n"
        f"RBI Minimum Income Guideline: {MIN_INCOME_GUIDELINE}\n"
        "Check for KYC match, document completeness, and income guideline compliance. "
        "Return the result in JSON format matching the schema."
    )

    result = await call_gemini(prompt, COMPLIANCE_SCHEMA)
    return ComplianceReport(**result)


async def bank_analyzer_agent(file_bytes: bytes) -> BankStatementExtractionResponse:
    if file_bytes[:4] == b"%PDF":
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = await extract_text_from_image(file_bytes)
    return BankStatementExtractionResponse(raw_text=raw_text)

async def payslip_analyzer_agent(file_bytes: bytes) -> PayslipExtractionResponse:
    raw_text = await extract_text_from_image(file_bytes)
    # Simple net income extraction: look for a number in text (mock logic)
    import re
    match = re.findall(r"\b\d{3,6}\b", raw_text)
    net_income = float(match[0]) if match else None
    return PayslipExtractionResponse(net_income=net_income, raw_text=raw_text)


app = FastAPI(title="Applicant Workflow with OCR + Gemini Compliance")

@app.post("/process_application", response_model=WorkflowResponse)
async def process_application(applicant_id: str = Query(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    bank_data = None
    payslip_data = None
    uploaded_file_types = []

    for file in files:
        file_bytes = await file.read()
        filename_lower = file.filename.lower()
        try:
            if "bank" in filename_lower:
                bank_data = await bank_analyzer_agent(file_bytes)
                uploaded_file_types.append("bank_statement")
                results.append(FileResult(filename=file.filename, type="bank", raw_text=bank_data.raw_text,
                                          analysis={"raw_text": bank_data.raw_text}))
            elif "payslip" in filename_lower:
                payslip_data = await payslip_analyzer_agent(file_bytes)
                uploaded_file_types.append("payslip")
                results.append(FileResult(filename=file.filename, type="payslip", raw_text=payslip_data.raw_text,
                                          analysis={"net_income": payslip_data.net_income}))
            else:
                raw_text = await extract_text_from_image(file_bytes)
                results.append(FileResult(filename=file.filename, type="unknown", raw_text=raw_text, analysis={}))
        except Exception as e:
            results.append(FileResult(filename=file.filename, type="error", raw_text="", analysis={"error": str(e)}))

    compliance_report = await compliance_agent_gemini(bank_data, payslip_data, uploaded_file_types)

    return WorkflowResponse(
        status="success",
        applicant_id=applicant_id,
        files=results,
        compliance_report=compliance_report
    )
