import os
import json
import base64
import logging
from dotenv import load_dotenv
from vertexai import agent_engines

load_dotenv()
logging.basicConfig(level=logging.INFO)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("AGENT_LOCATION", "us-central1")
AGENT_ENGINE_ID = os.environ.get("ORCHESTRATE_AGENT_ID")

try:
    agent_engine = agent_engines.get_agent_engine(AGENT_ENGINE_ID)
    logging.info(f"Connected to Agent Engine: {AGENT_ENGINE_ID}")
except Exception as e:
    logging.error(f"Failed to connect to Agent Engine: {e}")
    agent_engine = None

def stream_booking_request(origin: str, destination: str, user_id: str):
    """
    Streams the process of booking a shipment via the orchestrate agent.
    """
    if not agent_engine:
        yield {"type": "error", "data": "Agent Engine client is not initialized."}
        return

    yield {"type": "thought", "data": "--- Initiating Booking Request ---"}
    yield {"type": "thought", "data": f"User ID: {user_id}"}
    yield {"type": "thought", "data": f"Booking from '{origin}' to '{destination}'."}

    prompt = (
        f"A user wants to book a shipment.\n"
        f"- Collection address: \"{origin}\"\n"
        f"- Destination address: \"{destination}\"\n\n"
        "Your task is to initiate this booking process. "
        "Acknowledge the request and ask for the next piece of information required, "
        "which is the package dimensions and weight."
    )

    try:
        for event in agent_engine.stream_query(prompt):
            # Pass through all events as-is
            yield event
    except Exception as e:
        logging.error(f"Error during booking interaction: {e}")
        yield {"type": "error", "data": f"Error communicating with the agent: {e}"}

def stream_id_verification(file_data: dict, user_id: str):
    """
    Streams the process of verifying a user's ID via OCR using the orchestrate agent.
    """
    if not agent_engine:
        yield {"type": "error", "data": "Agent Engine client is not initialized."}
        return

    yield {"type": "thought", "data": "--- Initiating ID Verification ---"}
    yield {"type": "thought", "data": f"User ID: {user_id}"}
    yield {"type": "thought", "data": f"Verifying ID card: '{file_data.get('filename', 'unknown')}'."}

    prompt = (
        "The user has uploaded their government-issued ID card for verification.\n"
        "Your task is to analyze the attached document and extract the following details:\n"
        "- Full Name\n"
        "- PAN Number (Permanent Account Number)\n"
        "- Date of Birth\n\n"
        "Use the specialized OCR agent to perform the extraction.\n"
        "Return the extracted information in a single, complete JSON object.\n\n"
        "Example JSON format:\n"
        "{\n"
        '  "fullName": "string",\n'
        '  "panNumber": "string",\n'
        '  "dateOfBirth": "YYYY-MM-DD"\n'
        "}\n\n"
        "CRITICAL: Your final response must be ONLY the JSON object. Do not include any conversational text or markdown formatting."
    )

    try:
        file_bytes = base64.b64decode(file_data["base64_content"])
        # The agent_engines API expects a dict for files, not a Part object
        file_part = {
            "data": file_bytes,
            "mime_type": file_data["content_type"],
            "filename": file_data.get("filename", "id_card")
        }

        # The query is a list: [prompt, file_part]
        for event in agent_engine.stream_query([prompt, file_part]):
            yield event

    except Exception as e:
        logging.error(f"Error during agent interaction for ID validation: {e}")
        yield {"type": "error", "data": f"Error communicating with the agent: {e}"}