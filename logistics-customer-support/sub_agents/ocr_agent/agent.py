import os
import uuid
import posixpath
import re
import threading
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision, storage
from google.adk import Agent

MODEL = "gemini-2.5-pro-preview-05-06"
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "logistics_customer_support_bucket")
IMAGE_TEMP_FOLDER = "image_temp"

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
        patterns = {
            "name": r"([A-Za-z\s]+)\nBhaskar Subbian",
            "dob": r"DOB:\s*(\d{2}/\d{2}/\d{4})",
            "gender": r"Gender:\s*(MALE|FEMALE)",
            "id_number": r"(\d{4}\s\d{4}\s\d{4})",
            "vid_number": r"VID\s*:\s*(\d{4}\s\d{4}\s\d{4}\s\d{4})",
            "issue_date": r"Issue\s*Date:\s*(\d{2}/\d{2}/\d{4})",
            "address_en": r"Address\s*:\s*((?:.|\n)+?)(?=\d{4}\s*\d{4}\s*\d{4}|VID\s*:)",
            "address_ta": r"முகவரி\s*:\s*((?:.|\n)+?)(?=Address\s*:|\d{4}\s*\d{4}\s*\d{4}|VID\s*:)",
            "postal_code": r"(\d{6})"
        }
        extracted_data = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, extracted_text, re.IGNORECASE)
            if match:
                value = match.group(1)
                value = value.replace('\n', ' ').strip()
                value = re.sub(r'\s+', ' ', value)
                extracted_data[field] = value
            else:
                extracted_data[field] = "Not Found"
        extracted_data["full_text"] = extracted_text.strip()
        return extracted_data
    except Exception as e:
        return {"error": str(e)}

def upload_id_image(file_bytes: bytes, filename: str = None) -> str:
    return upload_to_gcs_from_bytes(file_bytes, filename)

def extract_id_details_from_gcs(gcs_uri: str) -> dict:
    image_bytes = download_gcs_blob_as_bytes(gcs_uri)
    result = extract_id_details_from_bytes(image_bytes)
    result["gcs_uri"] = gcs_uri
    return result

OCR_AGENT_INSTRUCTIONS = """
You are an agent that can:
1. Upload an image to GCS from provided file bytes.
2. Extract ID details from an image in GCS using Google Cloud Vision OCR.
"""

root_agent = Agent(
    model=MODEL,
    name="ocr_gcs_vision_agent",
    tools=[upload_id_image, extract_id_details_from_gcs],
    instruction=OCR_AGENT_INSTRUCTIONS,
)

# Start ADK MCP agent server in a separate thread
def serve_mcp():
    # You can customize host/port for MCP as needed
    root_agent.serve_as_mcp(host="0.0.0.0", port=int(os.environ.get("MCP_PORT", 5005)))

# FastAPI REST API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.post("/upload_and_extract/")
async def upload_and_extract(file: UploadFile = File(...)):
    file_bytes = await file.read()
    gcs_uri = upload_id_image(file_bytes, file.filename)
    ocr_result = extract_id_details_from_gcs(gcs_uri)
    return {"gcs_uri": gcs_uri, "ocr_result": ocr_result}

if __name__ == "__main__":
    # Start MCP agent server in a thread
    mcp_thread = threading.Thread(target=serve_mcp, daemon=True)
    mcp_thread.start()
    # Start FastAPI (Uvicorn) server
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)