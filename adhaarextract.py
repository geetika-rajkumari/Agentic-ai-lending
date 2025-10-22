from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError
import easyocr
import io
from PIL import Image
import json
import os 
import re # Added for text cleaning
from typing import Optional, Dict, Any

import httpx # Changed from 'requests' for async network calls
import asyncio # Added for concurrency


app = FastAPI(title="Aadhaar OCR Extractor")


try:
    # EasyOCR reader remains a synchronous, blocking object
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None
    
API_KEY = os.getenv("GEMINI_API_KEY", "") 
if not API_KEY:
    print("WARNING: API_KEY is not set. Requests will only succeed in environments where the key is automatically injected.")
    
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Pydantic Models ---
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

# --- Gemini Schema ---
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


# --- Asynchronous Gemini API Call ---
async def call_gemini_api_async(raw_text: str) -> Dict[str, Any]:
    """
    Asynchronously calls the Gemini API to extract structured data using httpx.
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
    
    # Use httpx.AsyncClient for non-blocking HTTP requests
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url_with_key = f"{API_URL}?key={API_KEY}"

            response = await client.post(
                url_with_key,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() 

            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            json_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '{}')
            
            return json.loads(json_text)
            
        except httpx.RequestError as e:
            print(f"Gemini API Request Error: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from Gemini response: {e}")
            return {}

async def extract_aadhaar_data_via_gemini(raw_text: str) -> AadhaarData:
    """
    Main function to orchestrate the LLM-based extraction.
    """
    extracted_dict = await call_gemini_api_async(raw_text)
    
    try:
        return AadhaarData(**extracted_dict)
    except ValidationError:
        return AadhaarData() 

# --- FastAPI Endpoint ---

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
      
        # CRITICAL FIX: Run the blocking EasyOCR call in a separate thread for concurrency
        result = await asyncio.to_thread(reader.readtext, image)
        
        raw_text_list = [text for (bbox, text, conf) in result]
        raw_text = ' '.join(raw_text_list)
        
        # --- ROBUSTNESS FIX: Clean the raw text before sending to LLM ---
        cleaned_raw_text = raw_text
        
        # Regex finds the Aadhaar number (12 digits, potentially spaced) and removes spaces.
        # This addresses the null output by guaranteeing the LLM sees the correct format.
        # Example: '1234 5678 9012' becomes '123456789012'
        cleaned_raw_text = re.sub(r'(\d{4})\s(\d{4})\s(\d{4})', r'\1\2\3', cleaned_raw_text)
        # --- END ROBUSTNESS FIX ---

        # Call the asynchronous Gemini extraction function with the CLEANED text
        aadhaar_data = await extract_aadhaar_data_via_gemini(cleaned_raw_text)

        
        if aadhaar_data.aadhaar_number and aadhaar_data.name:
            status_msg = "LLM extraction successful. Primary fields found."
        else:
            print(f"DEBUG: Failed extraction on cleaned text: {cleaned_raw_text}")
            status_msg = "LLM extraction completed, but the primary fields (Aadhaar number/Name) could not be reliably found."

        return ExtractionResponse(
            status="Success",
            data=aadhaar_data,
            raw_text=raw_text, # Return original raw text for debugging
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
