import os
import json
import base64
import time
from typing import List, Optional, Dict, Any, Tuple
from io import BytesIO

# Third-party libraries
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field, conlist
import httpx 
from PIL import Image

# NOTE: If you install the system dependency 'poppler-utils' (e.g., sudo apt install poppler-utils), 
# uncomment the line below to enable PDF processing:
# from pdf2image import convert_from_bytes 

# --- Configuration ---
# !!! IMPORTANT !!!
# Set your actual Gemini API Key here. For production, load this from environment variables.
GEMINI_API_KEY = "AIzaSyCY-lOPhvg38eZmWBT2j9pbP7jOj4RWdGU" 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MAX_PROCESS_TIME_SECONDS = 55 

# --- Pydantic Schemas for Agents and Summary ---

class Component(BaseModel):
    """Schema for individual earning or deduction components."""
    # FINAL FIX: Reverting to 'name' because the API output has consistently provided 'name' or 'type' 
    # and Pydantic validation now shows it is receiving 'name' from the last successful API call.
    name: str = Field(description="The name of the component (e.g., 'Basic', 'PF', 'HRA').")
    amount: float = Field(description="The monetary value of the component, as a floating-point number.")

class PayslipOutput(BaseModel):
    """Structured schema for the extracted salary slip data."""
    employeeName: str = Field(description="The full name of the employee.")
    payPeriod: str = Field(description="The payment period covered, e.g., 'October 2025'.")
    grossPay: float = Field(description="The total earnings before any deductions.")
    netPay: float = Field(description="The final take-home pay after all deductions.")
    earningsComponents: conlist(Component, min_length=1) = Field(description="Earning components.")
    deductionsComponents: List[Component] = Field(description="Deduction components.")

class BankAnalyzerOutput(BaseModel):
    """Mock structured schema for extracted bank statement data."""
    bankName: str
    accountHolder: str
    averageMonthlyBalance: float
    totalCreditsLast3Months: float
    analysisDate: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d"))

class ApplicationSummary(BaseModel):
    """The final consolidated output schema for the dashboard."""
    overallStatus: str = Field(description="Overall status (e.g., 'APPROVED', 'REJECTED', 'PENDING_REVIEW').")
    preScreeningNotes: str = Field(description="A brief summary of findings and decision rationale.")
    totalProcessingTimeSeconds: float = Field(description="The actual time taken for the entire workflow.")
    
    payslipData: Optional[PayslipOutput] = Field(None, description="Structured data from the payslip agent.")
    payslipError: Optional[str] = Field(None, description="Error message if payslip parsing failed.")

    bankData: Optional[BankAnalyzerOutput] = Field(None, description="Structured data from the bank analyzer agent.")
    bankError: Optional[str] = Field(None, description="Error message if bank analysis failed.")


# --- Helper Function for Schema Conversion ---

def get_gemini_schema(pydantic_model: BaseModel) -> Dict[str, Any]:
    """
    Converts a Pydantic model's schema into the format required by the Gemini API, 
    aggressively stripping unsupported metadata like '$defs', 'title', and 'description'.
    """
    raw_schema = pydantic_model.model_json_schema()
    STRIP_KEYS = ('title', 'default', 'description', '$defs', '$ref', 'definitions')
    
    def convert_schema(s):
        if isinstance(s, dict):
            new_s = {}
            for k, v in s.items():
                if k == 'type':
                    new_s[k] = v.upper()
                elif k in ('properties', 'items'):
                    new_s[k] = convert_schema(v)
                elif k not in STRIP_KEYS:
                    new_s[k] = convert_schema(v)
            return new_s
        elif isinstance(s, list):
            return [convert_schema(item) for item in s]
        else:
            return s
            
    cleaned_schema = convert_schema(raw_schema)
    
    final_schema = {
        k: v for k, v in cleaned_schema.items() if k not in ('$defs', '$ref')
    }
    
    return final_schema


# --- Document Processing Helpers ---

async def convert_to_base64_image(file_bytes: bytes, mime_type: str) -> Tuple[str, str]:
    """
    Converts a file (Image or PDF) to a Base64 encoded PNG image string.
    """
    
    if mime_type == 'application/pdf':
        try:
            raise RuntimeError("PDF conversion simulated to fail. Install 'poppler-utils' and uncomment conversion code for live PDF support.")
        except Exception as e:
            raise ValueError(f"PDF Conversion Failed: {str(e)}")
    
    elif mime_type.startswith('image/'):
        try:
            img = Image.open(BytesIO(file_bytes))
            
            if img.size[0] > 2000 or img.size[1] > 2000:
                 img.thumbnail((2000, 2000))
                 
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return base64_data, 'image/png'
        except Exception as e:
            raise ValueError(f"Image Loading/Encoding Failed: {str(e)}")
    
    raise ValueError(f"Unsupported file type for conversion: {mime_type}. Must be image or PDF.")


# --- Agent Functions ---

async def _call_gemini_agent(
    agent_name: str, 
    prompt: str, 
    b64_image: str, 
    mime_type: str, 
    response_schema: BaseModel
) -> Dict[str, Any]:
    """Generic function to call a Gemini agent with structured output and robust handling."""
    
    system_prompt = (
        f"You are the {agent_name} agent. Your task is to analyze the provided document image and extract ALL requested data "
        f"into the specified JSON schema. All monetary values MUST be numeric. Be highly accurate."
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": b64_image
                    }
                }
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": get_gemini_schema(response_schema), 
        },
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            response.raise_for_status() 
        except httpx.HTTPStatusError as e:
            api_error_detail = response.text 
            raise RuntimeError(f"API Rejected Payload: Status {e.response.status_code}. Detail: {api_error_detail}") from e
        
        result = response.json()
        json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')

        if not json_text:
            raise ValueError(f"{agent_name} Agent: API returned no content.")
            
        extracted_data = json.loads(json_text)
        
        # Pydantic validation (Robust Error Handling)
        validated_output = response_schema(**extracted_data)
        return validated_output.model_dump()


async def _run_payslip_parser(b64_image: str, mime_type: str) -> PayslipOutput:
    """Agent 1: Parses structured data from the payslip."""
    prompt = "Extract all fields from the payslip: employee name, pay period, gross pay, net pay, and lists of earnings and deductions."
    
    data = await _call_gemini_agent(
        agent_name="Payslip Parser", 
        prompt=prompt, 
        b64_image=b64_image, 
        mime_type=mime_type, 
        response_schema=PayslipOutput
    )
    return PayslipOutput(**data)


async def _run_bank_analyzer(b64_image: str, mime_type: str) -> BankAnalyzerOutput:
    """Agent 2: Parses and summarizes data from the bank statement."""
    prompt = (
        "Analyze the provided bank statement (past 3 months). "
        "Identify the account holder, bank name, calculate the average monthly balance, "
        "and determine the total credited amount over the period. "
        "Report 0 for any missing monetary metrics."
    )
    
    data = await _call_gemini_agent(
        agent_name="Bank Analyzer", 
        prompt=prompt, 
        b64_image=b64_image, 
        mime_type=mime_type, 
        response_schema=BankAnalyzerOutput
    )
    return BankAnalyzerOutput(**data)


async def _run_pre_screener(payslip_data: Optional[PayslipOutput], bank_data: Optional[BankAnalyzerOutput]) -> Tuple[str, str]:
    """Agent 3: Analyzes combined structured data and generates a decision summary."""
    print("--- Running Pre-screener Agent ---")
    
    notes = []
    status = "PENDING_REVIEW"
    
    # Logic for decision-making based on available data
    if payslip_data and payslip_data.netPay > 4000:
        notes.append("Payslip shows strong net income above lending threshold.")
    elif payslip_data:
        notes.append("Payslip net income is confirmed but below strong lending threshold.")
    
    if bank_data and bank_data.averageMonthlyBalance > 1500 and bank_data.totalCreditsLast3Months > 12000:
        notes.append("Bank analysis indicates stable average balance and high credit activity.")
    elif bank_data:
        notes.append("Bank stability metrics were marginal or just met minimum requirements.")
        
    # Determine Overall Status based on data availability (Robust Error Handling)
    if payslip_data and bank_data and payslip_data.netPay > 5000 and bank_data.averageMonthlyBalance > 2000:
        status = "APPROVED"
        notes.append("Criteria met for high confidence approval.")
    elif (payslip_data and not bank_data) or (not payslip_data and bank_data):
        status = "PENDING_REVIEW"
        notes.append("Insufficient data: Review required as one document failed to process. Check error fields.")
    elif not payslip_data and not bank_data:
        status = "REJECTED"
        notes.append("REJECTED: No valid financial documents could be processed for verification.")
        
    return status, " | ".join(notes)


# --- FastAPI App Initialization and Endpoint ---

app = FastAPI(
    title="Financial Application Workflow",
    description="Sequential processing of financial documents with robust error handling.",
)


@app.post("/process_application", response_model=ApplicationSummary)
async def process_application(
    payslip: UploadFile = File(..., description="The applicant's salary slip document (image or PDF)."),
    bank_statement: UploadFile = File(..., description="The applicant's bank statement document (image or PDF).")
):
    """
    Runs the sequential workflow: Document Parsing -> Bank Analysis -> Payslip Analysis -> Pre-screener Summary.
    NOTE: The Bank Analysis runs before Payslip Analysis for debugging purposes.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY not configured. Please set the key in main.py."
        )
        
    start_time = time.time()
    
    # Initialize result container
    payslip_data, payslip_error = None, None
    bank_data, bank_error = None, None
    
    # 1. --- Bank Statement Analysis Agent (Fail Gracefully) ---
    try:
        bank_bytes = await bank_statement.read()
        bank_b64, bank_mime = await convert_to_base64_image(bank_bytes, bank_statement.content_type)
        
        bank_data = await _run_bank_analyzer(bank_b64, bank_mime)
        
    except (ValueError, httpx.HTTPError, json.JSONDecodeError, RuntimeError) as e:
        bank_error = f"Bank Agent Failed ({type(e).__name__}). Document may be unreadable or in a new format: {e}"
        
    # 2. --- Payslip Parsing Agent (Fail Gracefully) ---
    try:
        payslip_bytes = await payslip.read()
        payslip_b64, payslip_mime = await convert_to_base64_image(payslip_bytes, payslip.content_type)
        
        payslip_data = await _run_payslip_parser(payslip_b64, payslip_mime)
        
    except (ValueError, httpx.HTTPError, json.JSONDecodeError, RuntimeError) as e:
        payslip_error = f"Payslip Agent Failed ({type(e).__name__}): {e}"
    
    # 3. --- Pre-screener Agent (Handles Null Data) ---
    overall_status, pre_screening_notes = await _run_pre_screener(payslip_data, bank_data)

    # --- Final Summary and Latency Check ---
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    if total_time > MAX_PROCESS_TIME_SECONDS:
        pre_screening_notes += f" | WARNING: Process exceeded target time of {MAX_PROCESS_TIME_SECONDS}s."
    
    # Construct the consolidated dashboard response
    summary = ApplicationSummary(
        overallStatus=overall_status,
        preScreeningNotes=pre_screening_notes,
        totalProcessingTimeSeconds=total_time,
        payslipData=payslip_data,
        payslipError=payslip_error,
        bankData=bank_data,
        bankError=bank_error
    )
    
    return summary