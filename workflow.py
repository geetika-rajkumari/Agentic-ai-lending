from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional
import asyncio
import os
import io
from PIL import Image
from pypdf import PdfReader
import json
import httpx
import easyocr


from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MAX_RETRIES = 3
BASE_DELAY = 1

if not API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Gemini API calls will fail.")


app = FastAPI(title="Applicant Workflow")

try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None

class Transaction(BaseModel):
    date: Optional[str]
    description: Optional[str]
    amount: Optional[str]
    type: Optional[str]
    category: Optional[str]

class BankStatementExtractionResponse(BaseModel):
    transactions: List[Transaction]
    raw_text: str

class PayslipExtractionResponse(BaseModel):
    employer: Optional[str]
    net_income: Optional[float]
    deductions: Optional[float]
    raw_text: str

class FileResult(BaseModel):
    filename: str
    type: str
    raw_text: str
    analysis: Dict[str, Any]

class WorkflowResponse(BaseModel):
    status: str
    applicant_id: str
    files: List[FileResult]
    summary: Dict[str, Any]

async def call_gemini(raw_text: str, schema: Dict[str, Any], system_prompt: str) -> Any:
    payload = {
        "contents": [{"parts": [{"text": raw_text}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0.0
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(f"{API_URL}?key={API_KEY}",
                                             headers={'Content-Type': 'application/json'},
                                             data=json.dumps(payload))
                response.raise_for_status()
                result = response.json()
                candidate = result.get('candidates', [{}])[0]
                json_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '[]')
                return json.loads(json_text)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BASE_DELAY * (2 ** attempt))
                    continue
                print(f"Gemini API failed: {e}")
                return {}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
        return raw_text
    except Exception as e:
        print(f"PDF parsing error: {e}")
        return ""

async def extract_text_from_image(file_bytes: bytes) -> str:
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR engine unavailable.")
    image = Image.open(io.BytesIO(file_bytes))
    result = await asyncio.to_thread(reader.readtext, image)
    raw_text = ' '.join([text for (_, text, _) in result])
    return raw_text

BANK_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "date": {"type": "STRING"},
            "description": {"type": "STRING"},
            "amount": {"type": "STRING"},
            "type": {"type": "STRING"},
            "category": {"type": "STRING"}
        },
        "required": ["description", "amount"]
    }
}

async def bank_analyzer_agent(file_bytes: bytes) -> BankStatementExtractionResponse:
    # Extract text
    raw_text = extract_text_from_pdf(file_bytes)
    # Gemini prompt
    system_prompt = "Extract structured transactions from the bank statement text."
    extracted = await call_gemini(raw_text, BANK_SCHEMA, system_prompt)
    transactions = []
    for item in extracted:
        try:
            transactions.append(Transaction(**item))
        except ValidationError:
            continue
    return BankStatementExtractionResponse(transactions=transactions, raw_text=raw_text)

PAYSLIP_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "employer": {"type": "STRING"},
        "net_income": {"type": "NUMBER"},
        "deductions": {"type": "NUMBER"}
    },
    "required": ["net_income"]
}

async def payslip_analyzer_agent(file_bytes: bytes) -> PayslipExtractionResponse:
    # Extract text
    raw_text = await extract_text_from_image(file_bytes)
    # Gemini prompt
    system_prompt = "Extract employer, net income, and deductions from payslip text."
    extracted = await call_gemini(raw_text, PAYSLIP_SCHEMA, system_prompt)
    return PayslipExtractionResponse(
        employer=extracted.get("employer"),
        net_income=extracted.get("net_income"),
        deductions=extracted.get("deductions"),
        raw_text=raw_text
    )

def pre_screener_summary(bank_data: Optional[BankStatementExtractionResponse],
                         payslip_data: Optional[PayslipExtractionResponse]) -> Dict[str, Any]:
    summary_text = "Applicant has stable income." if payslip_data else "No payslip data."
    risk_score = 0.5
    return {"summary": summary_text, "risk_score": risk_score}


@app.post("/process_application", response_model=WorkflowResponse)
async def process_application(applicant_id: str = Query(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    bank_data = None
    payslip_data = None

    for file in files:
        file_bytes = await file.read()
        filename_lower = file.filename.lower()

        try:
            if "bank" in filename_lower:
                bank_data = await bank_analyzer_agent(file_bytes)
                results.append(FileResult(filename=file.filename, type="bank", raw_text=bank_data.raw_text,
                                          analysis={"transactions": [t.dict() for t in bank_data.transactions]}))
            elif "payslip" in filename_lower:
                payslip_data = await payslip_analyzer_agent(file_bytes)
                results.append(FileResult(filename=file.filename, type="payslip", raw_text=payslip_data.raw_text,
                                          analysis=payslip_data.dict()))
            else:
                # Default OCR for unknown file types
                raw_text = await extract_text_from_image(file_bytes)
                results.append(FileResult(filename=file.filename, type="unknown", raw_text=raw_text, analysis={}))
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append(FileResult(filename=file.filename, type="error", raw_text="", analysis={"error": str(e)}))

    summary = pre_screener_summary(bank_data, payslip_data)

    return WorkflowResponse(status="success", applicant_id=applicant_id, files=results, summary=summary)
