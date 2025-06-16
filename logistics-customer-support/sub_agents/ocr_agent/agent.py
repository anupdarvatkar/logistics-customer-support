import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from google.adk import Agent

MODEL = "gemini-2.0-flash"

PAN_EXTRACTION_INSTRUCTION = """
You are an expert information extraction agent. Your task is to extract details from Indian PAN card text with maximum accuracy and completeness.

Instructions:
- Carefully analyze the input text, which may contain PAN card details in any order or format.
- Extract and return a JSON object with these fields:
  - pan_number: The 10-character alphanumeric Permanent Account Number (e.g., ABCDE1234F)
  - name: The full name as printed on the PAN card
  - father_name: The full father's name as printed on the PAN card
  - dob: Date of birth in DD/MM/YYYY format
  - gender: Gender as printed (MALE, FEMALE, or OTHER)

Guidelines:
- Do not infer or guess values; only extract what is present in the text.
- If a field is missing or unclear, set its value to null.
- Do not truncate names or other fields; return the full value as printed.
- Ignore any unrelated text or noise.
- Return only the JSON object, with no extra explanation or formatting.

Example output:
{
  "pan_number": "ABCDE1234F",
  "name": "RAVI KUMAR",
  "father_name": "SURESH KUMAR",
  "dob": "12/05/1985",
  "gender": "MALE"
}
"""

# Create the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the agent
ocr_extract_agent = Agent(
    model=MODEL,
    name="pan_extractor_agent",
    instruction=PAN_EXTRACTION_INSTRUCTION,
)

class PanRequest(BaseModel):
    text: str

def extract_pan_json(text: str) -> dict:
    """
    Passes the input text to Gemini LLM via ocr_extract_agent and returns the extracted PAN card details as a JSON object.
    """
    if not text or not isinstance(text, str):
        return {"error": "No text provided or invalid text format."}
    
    if len(text.strip()) < 10:
        return {"error": "Text is too short to contain valid PAN card details."}
    
    try:
        # Call the ocr_extract_agent to process the text
        response = ocr_extract_agent.invoke(text)
        
        # Validate the response
        if not response:
            return {"error": "No data returned from extraction agent."}
            
        print(f"Successfully extracted PAN card details from text.")
        return response
        
    except Exception as e:
        error_message = f"Error extracting PAN card details: {str(e)}"
        print(error_message)
        return {"error": error_message}

@app.post("/extract_pan")
async def extract_pan(request: PanRequest):
    """
    Endpoint to extract PAN card details from text.
    """
    result = extract_pan_json(request.text)
    return {"extracted_pan": result}

# Run the FastAPI server when this script is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)