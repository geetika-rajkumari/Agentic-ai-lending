from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError
import easyocr
import io
from PIL import Image
import json
import os
import httpx
import asyncio
from typing import Optional, List, Dict, Any
from io import StringIO
import time 
from pypdf import PdfReader 


app = FastAPI(title="Bank Statement Extractor")


try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None


API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MAX_RETRIES = 3 
BASE_DELAY = 1

if not API_KEY:
    print("WARNING: GEMINI_API_KEY not set.")


class Transaction(BaseModel):
    date: Optional[str]
    description: Optional[str]
    amount: Optional[str]
    type: Optional[str]  
    category: Optional[str]  

class BankStatementExtractionResponse(BaseModel):
    status: str
    transactions: List[Transaction]
    raw_text: str
    message: str


BANK_SCHEMA: Dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "date": {"type": "STRING", "description": "Transaction date"},
            "description": {"type": "STRING", "description": "Narration of the transaction"},
            "amount": {"type": "STRING", "description": "Transaction amount"},
            "type": {"type": "STRING", "description": "Credit or Debit"},
            "category": {
                "type": "STRING",
                "description": "Transaction category: Salary, EMI, Anomaly, or Other"
            }
        },
        "required": ["description", "amount"]
    }
}

async def call_gemini_for_transactions_async(raw_text: str) -> List[Dict[str, Any]]:
    if not API_KEY:
        print("ERROR: Gemini API key missing. Cannot make external API call.")
        return []

    system_prompt = (
        "You are a highly accurate financial document parser. Your job is to extract structured transaction data from the provided bank statement text. "
        "Pay special attention to classifying **Salary Credits** and identifying **Negative Behaviors/Irregularities** like 'EMI bounce', 'penalty fees', or 'NSF charges', categorizing these as 'Anomaly'. "
        "Categorize each transaction as Salary, EMI, Anomaly, or Other."
    )

    user_query = f"Extract transactions from the following bank statement text:\n\n---START---\n{raw_text}\n---END---"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": BANK_SCHEMA,
            "temperature": 0.0
        }
    }
    
    
    async with httpx.AsyncClient(timeout=60.0) as client: 
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    f"{API_URL}?key={API_KEY}",
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()
                
                
                candidate = result.get('candidates', [{}])[0]
                json_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '[]')
                
                
                return json.loads(json_text)

            except httpx.HTTPStatusError as e:
                
                if e.response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt)
        
                    await asyncio.sleep(delay)
                    continue 
                
                
                print(f"Gemini API HTTP Error (Attempt {attempt + 1}): {e.response.status_code}")
                return []
                
            except httpx.RequestError as e:
                
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                
                print(f"Gemini API Request Error (Attempt {attempt + 1}): {type(e).__name__} - {e}")
                return []
                
            except json.JSONDecodeError:
                
                print(f"Gemini API JSON Decode Error: Model returned non-JSON data.")
                return []
        
        
        print("Gemini API Request failed after all retries.")
        return []

@app.post("/api/v1/extract_bank_statement", response_model=BankStatementExtractionResponse)
async def extract_bank_statement(file: UploadFile = File(...)):
    """
    Accepts a bank statement (Image, Text, CSV, or PDF) and extracts structured transaction info using Gemini.
    """
    
    file_type = file.content_type
    file_extension = os.path.splitext(file.filename)[1].lower()
    raw_text = ""
    
    try:
        if file_type in ["image/jpeg", "image/png"]:
            if reader is None:
                raise HTTPException(status_code=503, detail="OCR engine unavailable.")
                
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

        
            result = await asyncio.to_thread(reader.readtext, image)
            raw_text = ' '.join([text for (_, text, _) in result])
            
        elif file_extension in ['.csv', '.txt'] or file_type in ["text/csv", "text/plain"]:
            
            content = await file.read()
            raw_text = content.decode('utf-8')
            
        elif file_type == "application/pdf" or file_extension == '.pdf':
        
            contents = await file.read()
            pdf_reader = PdfReader(io.BytesIO(contents))
            raw_text = ""
            for page in pdf_reader.pages:
                raw_text += page.extract_text() or ""
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only JPEG, PNG, PDF, CSV, and plain text files are allowed.")


        extracted = await call_gemini_for_transactions_async(raw_text)
        
        
        transactions = []
        for item in extracted:
            try:
                transactions.append(Transaction(**item))
            except ValidationError as e:
                print(f"Validation error on extracted item (skipping): {e}")
                continue

        message = f"{len(transactions)} transactions extracted."
        return BankStatementExtractionResponse(
            status="Success",
            transactions=transactions,
            raw_text=raw_text,
            message=message
        )

    except HTTPException:
        
        raise
    except Exception as e:
        
        print(f"Overall Processing Error during file read/OCR/PDF: {e}") 
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
