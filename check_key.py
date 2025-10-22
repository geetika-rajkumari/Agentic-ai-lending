from dotenv import load_dotenv
import os
import requests
import json

# Load environment variables (including GEMINI_API_KEY)
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

print("--- Gemini API Key Check ---")

if not API_KEY:
    print("❌ FAILURE: GEMINI_API_KEY is not loaded from the .env file.")
    print("   Please check that the key is set correctly in .env, and the file is in the same folder.")
    exit()

print(f"✅ SUCCESS: API Key found in environment variables. Starting connection test...")

try:
    # Minimal payload for a simple text generation request
    payload = {
        "contents": [{"parts": [{"text": "Generate a single word response: 'OK'"}]}]
    }

    # Use a synchronous requests call since this is a simple, one-off test
    response = requests.post(
        f"{API_URL}?key={API_KEY}",
        headers={'Content-Type': 'application/json'},
        data=json.dumps(payload),
        timeout=10 # Set a reasonable timeout
    )

    # Check for HTTP errors (4xx or 5xx)
    response.raise_for_status()

    result = response.json()

    # Check if the model returned content
    model_response = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()

    if "OK" in model_response:
        print("🎉 **VALID KEY & WORKING CONNECTION:**")
        print("   The key is correct, and the model responded successfully.")
    else:
        print("⚠️ **MODEL RESPONSE ISSUE:**")
        print(f"   The connection was successful, but the model gave an unexpected response: '{model_response}'.")
        print("   The key is likely valid, but check the model's output for API limits or content blocks.")


except requests.exceptions.HTTPError as errh:
    print(f"❌ **INVALID KEY/ACCESS DENIED (HTTP Error {errh.response.status_code}):**")
    if errh.response.status_code == 400:
        print("   This often means the API Key is invalid, has expired, or the endpoint URL is wrong.")
    elif errh.response.status_code == 429:
        print("   You may be hitting a quota limit (Too Many Requests).")
    else:
        print(f"   HTTP Error: {errh}")

except requests.exceptions.ConnectionError as errc:
    print(f"❌ **CONNECTION FAILED:** Check your internet connection or firewall rules.")

except requests.exceptions.Timeout as errt:
    print(f"❌ **REQUEST TIMEOUT:** Connection was too slow. Check your network.")

except Exception as e:
    print(f"❌ **AN UNEXPECTED ERROR OCCURRED:** {e}")
