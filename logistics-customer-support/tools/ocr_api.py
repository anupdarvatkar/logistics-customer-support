import os
import uuid
import posixpath
import re
import requests
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision, storage
import logging
from google.adk import Agent

# Service URL from environment variable
MODEL = "gemini-2.0-flash"

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "logistics_customer_support_bucket")
IMAGE_TEMP_FOLDER = "image_temp"

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

# Create the agent (do this once at module level)
ocr_extract_agent = Agent(
    model=MODEL,
    name="pan_extractor_agent",
    instruction=PAN_EXTRACTION_INSTRUCTION,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def upload_to_gcs_from_bytes(file_bytes: bytes, filename: str = None, bucket_name: str = GCS_BUCKET_NAME) -> str:
    if not filename:
        filename = f"upload_{uuid.uuid4().hex}.jpg"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    unique_name = posixpath.join(IMAGE_TEMP_FOLDER, f"{uuid.uuid4().hex}_{filename}")
    blob = bucket.blob(unique_name)
    blob.upload_from_string(file_bytes)
    return f"gs://{bucket_name}/{unique_name}"

def download_gcs_blob_as_bytes(gcs_uri: str) -> bytes:
    assert gcs_uri.startswith("gs://")
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def extract_id_details_from_bytes(image_bytes: bytes) -> dict:
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(
            image=image,
            image_context={
                "language_hints": ["en", "hi", "ta"],
                "text_detection_params": {"enable_text_detection_confidence_score": True}
            }
        )
        if response.error.message:
            return {"error": response.error.message}
        extracted_text = ""
        if response.full_text_annotation and response.full_text_annotation.text:
            extracted_text = response.full_text_annotation.text
        elif response.text_annotations:
            extracted_text = response.text_annotations[0].description
        else:
            return {"error": "No text found in image."}
        return {"full_text": extracted_text.strip()}
    except Exception as e:
        return {"error": str(e)}

def extract_id_details_from_gcs(gcs_uri: str) -> dict:
    image_bytes = download_gcs_blob_as_bytes(gcs_uri)
    result = extract_id_details_from_bytes(image_bytes)
    result["gcs_uri"] = gcs_uri
    return result

# --- Tool functions for MCP agent ---

def tool_upload_and_extract(file_bytes: bytes, filename: str = None):
    """
    Uploads an image file to Google Cloud Storage and extracts text details using OCR.
    
    Args:
        file_bytes (bytes): The binary content of the image file to upload.
        filename (str, optional): The name of the file. If not provided, a random name will be generated.
        
    Returns:
        dict: A dictionary containing:
            {
                "gcs_uri": str,  # The Google Cloud Storage URI where the file was uploaded
                "ocr_result": {
                    "full_text": str,  # The full text extracted from the image
                    # Additional extracted fields if available
                }
            }
            
            If an error occurs, returns: {"error": error_message}
    """
    if not file_bytes or not isinstance(file_bytes, bytes):
        return {"error": "No file data provided or invalid file format."}
    
    try:
        # Upload the file to GCS
        gcs_uri = upload_to_gcs_from_bytes(file_bytes, filename)
        if not gcs_uri:
            return {"error": "Failed to upload file to Google Cloud Storage."}
            
        print(f"Successfully uploaded file to: {gcs_uri}")
        
        # Extract text using OCR
        ocr_result = extract_id_details_from_gcs(gcs_uri)
        if not ocr_result or "error" in ocr_result:
            error_msg = ocr_result.get("error", "Unknown error during OCR processing.")
            return {"error": error_msg, "gcs_uri": gcs_uri}
            
        print(f"Successfully extracted text from image at {gcs_uri}")
        return {"gcs_uri": gcs_uri, "ocr_result": ocr_result}
        
    except Exception as e:
        error_message = f"Error in upload and extract process: {str(e)}"
        print(error_message)
        return {"error": error_message}

def tool_upload_file(file_bytes: bytes, filename: str = None):
    """
    Uploads an image file to Google Cloud Storage.
    
    Args:
        file_bytes (bytes): The binary content of the file to upload.
        filename (str, optional): The name of the file. If not provided, a random name will be generated.
        
    Returns:
        dict: A dictionary containing:
            {
                "gcs_uri": str  # The Google Cloud Storage URI where the file was uploaded
            }
            
            If an error occurs, returns: {"error": error_message}
    """
    if not file_bytes or not isinstance(file_bytes, bytes):
        return {"error": "No file data provided or invalid file format."}
    
    try:
        # Upload the file to GCS
        gcs_uri = upload_to_gcs_from_bytes(file_bytes, filename)
        if not gcs_uri:
            return {"error": "Failed to upload file to Google Cloud Storage."}
            
        print(f"Successfully uploaded file to: {gcs_uri}")
        return {"gcs_uri": gcs_uri}
        
    except Exception as e:
        error_message = f"Error uploading file: {str(e)}"
        print(error_message)
        return {"error": error_message}

def tool_extract_pan(text: str) -> dict:
    """
    Extract PAN card details from text using the integrated LLM agent logic.
    This replaces the need to call the FastAPI endpoint and resolves cyclic dependency.
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
            
        logging.info("Successfully extracted PAN card details from text.")
        return response
        
    except Exception as e:
        error_message = f"Error extracting PAN card details: {str(e)}"
        logging.error(error_message)
        return {"error": error_message}
# --- FastAPI endpoints ---

@app.post("/upload_and_extract/")
async def upload_and_extract(file: UploadFile = File(...)):
    file_bytes = await file.read()
    return tool_upload_and_extract(file_bytes, file.filename)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    return tool_upload_file(file_bytes, file.filename)

@app.post("/extract/")
async def extract_pan(file: UploadFile = File(None), text: str = Form(None)):
    if file:
        file_bytes = await file.read()
        ocr_result = extract_id_details_from_bytes(file_bytes)
        text_to_extract = ocr_result.get("full_text", "")
    elif text:
        text_to_extract = text
    else:
        return {"error": "No file or text provided."}
    return tool_extract_pan(text_to_extract)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)