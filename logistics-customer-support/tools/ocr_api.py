import os
import uvicorn
import uuid
import posixpath
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision, storage

# Correct import for ocr_extract_agent
from sub_agents.ocr_agent.agent import ocr_extract_agent

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "logistics_customer_support_bucket")
IMAGE_TEMP_FOLDER = "image_temp"

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

def upload_id_image(file_bytes: bytes, filename: str = None) -> str:
    return upload_to_gcs_from_bytes(file_bytes, filename)

def extract_id_details_from_gcs(gcs_uri: str) -> dict:
    image_bytes = download_gcs_blob_as_bytes(gcs_uri)
    result = extract_id_details_from_bytes(image_bytes)
    result["gcs_uri"] = gcs_uri
    return result

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
    

def tool_upload_file(file_bytes: bytes, filename: str = None):
    gcs_uri = upload_to_gcs_from_bytes(file_bytes, filename)
    return {"gcs_uri": gcs_uri}


def tool_upload_and_extract(file_bytes: bytes, filename: str = None):
    gcs_uri = upload_id_image(file_bytes, filename)
    ocr_result = extract_id_details_from_gcs(gcs_uri)
    return {"gcs_uri": gcs_uri, "ocr_result": ocr_result}

def tool_extract_pan(text: str):
    pan_json = extract_pan_json(text)
    return {"extracted_pan": pan_json}


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

def extract_pan_json(text: str) -> dict:
    """
    Passes the input text to Gemini LLM via ocr_extract_agent and returns the extracted PAN card details as a JSON object.
    """
    response = ocr_extract_agent.invoke(text)
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)