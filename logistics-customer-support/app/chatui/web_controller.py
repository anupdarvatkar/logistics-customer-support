import os
import json
import base64
import logging
import pprint
from dotenv import load_dotenv
from vertexai import init as vertex_init
from vertexai.generative_models import Part, GenerativeModel
from vertexai.preview.generative_models import Tool

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("AGENT_LOCATION", "us-central1")
# **IMPORTANT**: Set this environment variable to your deployed Agent Engine ID
AGENT_ENGINE_ID = os.environ.get("ORCHESTRATE_AGENT_ID") 

# --- Vertex AI Initialization ---
try:
    vertex_init(project=PROJECT_ID, location=LOCATION)
    if not AGENT_ENGINE_ID:
        raise ValueError("ORCHESTRATE_AGENT_ID environment variable is not set.")
    # Initialize the remote agent engine
    orchestrate_agent_engine = GenerativeModel(
        model_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ENGINE_ID}",
        # The 'tools' parameter is often used for client-side tools, 
        # but the agent itself has tools defined in host_agent.py
        tools=[] 
    )
    logging.info(f"Successfully connected to Agent Engine: {AGENT_ENGINE_ID}")
except Exception as e:
    logging.error(f"Failed to initialize Vertex AI or connect to Agent Engine: {e}")
    orchestrate_agent_engine = None

def start_shipment_booking(origin: str, destination: str, user_id: str):
    """
    Calls the remote orchestrate_agent to begin the shipment booking process.
    Yields thought events for streaming to the UI.
    """
    if not orchestrate_agent_engine:
        yield {"type": "error", "data": "Agent Engine client is not initialized."}
        return

    yield {"type": "thought", "data": "--- Calling Orchestrate Agent to Start Booking ---"}
    yield {"type": "thought", "data": f"User ID: {user_id}"}
    yield {"type": "thought", "data": f"Task: Start shipment from '{origin}' to '{destination}'."}

    # This prompt instructs the orchestrator on its task. The orchestrator will then
    # use its own tools (e.g., send_task to a booking agent) to proceed.
    prompt = f"""
    A user wants to book a shipment.
    - The collection address is: "{origin}"
    - The destination address is: "{destination}"

    Your task is to initiate this booking process. Acknowledge the request and ask for the next piece of information required, which is the package dimensions and weight.
    """

    try:
        # Start a new chat session with the agent engine
        chat = orchestrate_agent_engine.start_chat()
        yield {"type": "thought", "data": "Sending booking request to agent..."}
        
        response = chat.send_message(prompt, stream=True)
        
        full_response_text = ""
        for chunk in response:
            if chunk.text:
                yield {"type": "agent_message_chunk", "data": chunk.text}
                full_response_text += chunk.text
        
        yield {"type": "thought", "data": f"Agent's full response: {full_response_text}"}
        yield {"type": "booking_started", "data": {"response": full_response_text}}

    except Exception as e:
        logging.error(f"Error during agent interaction for booking: {e}")
        yield {"type": "error", "data": f"An error occurred while communicating with the agent: {e}"}

def validate_id_with_agent(file_data: dict, user_id: str):
    """
    Calls the remote orchestrate_agent to validate an ID by sending file data.
    The orchestrator is expected to delegate this to an OCR agent.
    Yields thought and result events.
    """
    if not orchestrate_agent_engine:
        yield {"type": "error", "data": "Agent Engine client is not initialized."}
        return

    yield {"type": "thought", "data": "--- Calling Orchestrate Agent for ID Validation ---"}
    yield {"type": "thought", "data": f"User ID: {user_id}"}
    yield {"type": "thought", "data": f"Task: Extract details from uploaded ID card '{file_data['filename']}'."}

    # The prompt instructs the orchestrator on its high-level goal.
    # The orchestrator's internal logic (from host_agent.py) will guide it
    # to use its `send_task` tool to pass the file to the OCR agent.
    prompt = f"""
    The user has uploaded their government-issued ID card for verification.
    Your task is to analyze the attached document and extract the following details:
    - Full Name
    - PAN Number (Permanent Account Number)
    - Date of Birth

    Use the specialized OCR agent to perform the extraction.
    Return the extracted information in a single, complete JSON object.
    
    Example JSON format:
    {{
      "fullName": "string",
      "panNumber": "string",
      "dateOfBirth": "YYYY-MM-DD"
    }}
    
    CRITICAL: Your final response must be ONLY the JSON object. Do not include any conversational text or markdown formatting.
    """

    try:
        # Decode the base64 string back to bytes for the API call
        file_bytes = base64.b64decode(file_data["base64_content"])
        
        # Prepare the file part for the agent engine
        id_card_part = Part.from_data(
            data=file_bytes,
            mime_type=file_data["content_type"]
        )

        yield {"type": "thought", "data": "Sending ID document and prompt to agent..."}
        
        # Start a new chat session and send the message with the file attached
        chat = orchestrate_agent_engine.start_chat()
        response = chat.send_message([prompt, id_card_part], stream=True)

        accumulated_json_str = ""
        for chunk in response:
            # Log tool calls and other agent "thoughts"
            if chunk.function_calls:
                 yield {"type": "thought", "data": f"Agent is using a tool: {chunk.function_calls[0].name}"}
            if chunk.text:
                yield {"type": "agent_message_chunk", "data": chunk.text}
                accumulated_json_str += chunk.text
        
        yield {"type": "thought", "data": f"Agent's raw final output: {accumulated_json_str}"}

        # --- JSON Parsing Logic (from introvertally.py) ---
        json_to_parse = accumulated_json_str.strip()
        if "```json" in json_to_parse:
            try:
                json_to_parse = json_to_parse.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                yield {"type": "thought", "data": f"Extracted JSON block: {json_to_parse}"}
            except IndexError:
                yield {"type": "thought", "data": "Could not extract JSON from markdown block, attempting to parse as is."}

        if json_to_parse:
            try:
                final_result = json.loads(json_to_parse)
                yield {"type": "validation_complete", "data": final_result}
            except json.JSONDecodeError as e:
                error_message = f"Failed to parse agent's output as JSON. Error: {e}"
                yield {"type": "thought", "data": error_message}
                yield {"type": "error", "data": {"message": error_message, "raw_output": json_to_parse}}
        else:
            yield {"type": "error", "data": {"message": "Agent returned no content.", "raw_output": ""}}

    except Exception as e:
        logging.error(f"Error during agent interaction for ID validation: {e}")
        yield {"type": "error", "data": f"An error occurred while communicating with the agent: {e}"}