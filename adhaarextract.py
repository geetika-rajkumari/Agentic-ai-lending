from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import easyocr
import io
from PIL import Image
import json
import os 
from typing import Optional, Dict, Any

import requests 

app = FastAPI(title="Aadhaar OCR Extractor (Gemini LLM Prompted)")

try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None
    
API_KEY = os.getenv("GEMINI_API_KEY", "") 
if not API_KEY:
    print("WARNING: API_KEY is not set. Requests will only succeed in environments where the key is automatically injected.")
    
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

class AadhaarData(BaseModel):
    """Structured data extracted from the Aadhaar card."""
    name: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    aadhaar_number: Optional[str] = None
   

class ExtractionResponse(BaseModel):
    """API response structure."""
    status: str
    data: AadhaarData
    raw_text: str
    message: str

AADHAAR_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "name": {"type": "STRING", "description": "The full name of the cardholder."},
        "dob": {"type": "STRING", "description": "The Date of Birth, preferably in DD/MM/YYYY or YYYY format."},
        "gender": {"type": "STRING", "description": "The gender (MALE or FEMALE)."},
        "aadhaar_number": {"type": "STRING", "description": "The 12-digit Aadhaar number, without any spaces."},
        
    },
    "required": ["aadhaar_number", "name"]
}


def call_gemini_api_sync(raw_text: str) -> Dict[str, Any]:
    """
    Synchronously calls the Gemini API to extract structured data.
    
    """
    if not API_KEY:
        
        print("ERROR: Gemini API Key is missing. Cannot make external API call.")
        return {}
        
    
    system_prompt = (
        "You are an expert document parser. Your task is to extract structured information "
        "from the provided raw text, which was OCR'd from an Indian Aadhaar card. "
        "Strictly adhere to the required JSON schema. Only use data present in the text."
    )
    
    
    user_query = f"Extract the required fields from the following raw OCR text: \n\n---START OF TEXT---\n{raw_text}\n---END OF TEXT---\n"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": AADHAAR_SCHEMA,
            "temperature": 0.0 
        }
    }

    try:
        url_with_key = f"{API_URL}?key={API_KEY}"
        
        response = requests.post(
            url_with_key,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status() 
        
        result = response.json()
        
        
        candidate = result.get('candidates', [{}])[0]
        json_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '{}')
        
       
        return json.loads(json_text)
        
    except requests.exceptions.RequestException as e:
        print(f"Gemini API Request Error: {e}")
        
        return {}
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from Gemini response: {e}")
        return {}

def extract_aadhaar_data_via_gemini(raw_text: str) -> AadhaarData:
    """
    Main function to orchestrate the LLM-based extraction.
    """
    extracted_dict = call_gemini_api_sync(raw_text)
    
    
    try:
        return AadhaarData(**extracted_dict)
    except ValidationError:
        
        return AadhaarData() 

@app.post("/api/v1/extract_aadhaar", response_model=ExtractionResponse)
async def extract_aadhaar(file: UploadFile = File(...)):
    """
    Accepts an Aadhaar image (JPEG/PNG), performs OCR, and sends the raw text 
    to the Gemini LLM for structured data extraction.
    """
    if reader is None:
        raise HTTPException(
            status_code=503, 
            detail="OCR service is unavailable. EasyOCR failed to initialize."
        )
    
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only JPEG and PNG images are accepted."
        )

    try:
    
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
      
        result = reader.readtext(image)
        
        
        raw_text_list = [text for (bbox, text, conf) in result]
        raw_text = ' '.join(raw_text_list)
        
        
        aadhaar_data = extract_aadhaar_data_via_gemini(raw_text)

        
        if aadhaar_data.aadhaar_number:
            status_msg = "LLM extraction successful. Aadhaar number found."
        else:
            status_msg = "LLM extraction completed, but the primary fields (Aadhaar number/Name) could not be reliably found."

        return ExtractionResponse(
            status="Success",
            data=aadhaar_data,
            raw_text=raw_text,
            message=status_msg
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Extraction Error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during processing: {str(e)}"
        )