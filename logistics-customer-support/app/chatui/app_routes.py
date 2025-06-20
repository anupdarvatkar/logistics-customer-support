import json
import base64
import logging
import traceback
from flask import Blueprint, request, Response, stream_with_context, session

# Import the controller functions that interact with the remote Agent Engine
from web_controller import start_shipment_booking, validate_id_with_agent

# It's good practice to use a Blueprint for organizing routes
chat_bp = Blueprint('chat', __name__)

# --- API Routes for Chat Interaction ---

@chat_bp.route('/api/chat/start_booking', methods=['POST'])
def handle_start_booking():
    """
    Handles the initial request to start a shipment booking.
    This endpoint streams the agent's response.
    """
    data = request.get_json()
    if not data or 'origin' not in data or 'destination' not in data:
        return Response(json.dumps({"error": "Missing origin or destination"}), status=400, mimetype='application/json')

    # Use a unique user ID for the session, or create one if it doesn't exist.
    if 'user_id' not in session:
        session['user_id'] = 'user_' + os.urandom(8).hex()
    
    user_id = session['user_id']
    origin = data['origin']
    destination = data['destination']

    def generate_stream():
        """Generator function to stream agent responses."""
        logging.info(f"--- SSE: Starting booking stream for user '{user_id}' ---")
        try:
            # Call the controller function which yields events
            for event in start_shipment_booking(origin, destination, user_id):
                event_type = event.get("type", "thought")
                data_payload = json.dumps(event.get("data"))
                sse_message = f"event: {event_type}\ndata: {data_payload}\n\n"
                yield sse_message
            
            # Signal the end of the stream
            yield f"event: stream_end\ndata: {json.dumps({'message': 'Booking stream finished.'})}\n\n"
            logging.info(f"--- SSE: Booking stream finished for user '{user_id}' ---")

        except Exception as e:
            logging.error(f"!!! SSE: EXCEPTION during booking stream: {e} !!!")
            traceback.print_exc()
            error_payload = json.dumps({"message": f"Server error during agent communication: {e}"})
            yield f"event: error\ndata: {error_payload}\n\n"

    # Return a streaming response
    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')


@chat_bp.route('/api/chat/upload_id', methods=['POST'])
def handle_id_upload():
    """
    Handles the file upload for ID validation and streams the agent's response.
    """
    if 'file' not in request.files:
        return Response(json.dumps({"error": "No file part in the request"}), status=400, mimetype='application/json')
    
    file = request.files['file']
    if file.filename == '':
        return Response(json.dumps({"error": "No file selected"}), status=400, mimetype='application/json')

    if 'user_id' not in session:
        session['user_id'] = 'user_' + os.urandom(8).hex()
    
    user_id = session['user_id']

    # Prepare file data for the controller
    file_content_bytes = file.read()
    base64_content = base64.b64encode(file_content_bytes).decode('utf-8')
    file_data_for_agent = {
        "filename": file.filename,
        "content_type": file.mimetype,
        "base64_content": base64_content
    }

    def generate_validation_stream():
        """Generator function to stream agent's validation responses."""
        logging.info(f"--- SSE: Starting ID validation stream for user '{user_id}' ---")
        try:
            # Call the controller function which yields events
            for event in validate_id_with_agent(file_data_for_agent, user_id):
                event_type = event.get("type", "thought")
                data_payload = json.dumps(event.get("data"))
                sse_message = f"event: {event_type}\ndata: {data_payload}\n\n"
                yield sse_message

            # Signal the end of the stream
            yield f"event: stream_end\ndata: {json.dumps({'message': 'Validation stream finished.'})}\n\n"
            logging.info(f"--- SSE: Validation stream finished for user '{user_id}' ---")

        except Exception as e:
            logging.error(f"!!! SSE: EXCEPTION during validation stream: {e} !!!")
            traceback.print_exc()
            error_payload = json.dumps({"message": f"Server error during agent communication: {e}"})
            yield f"event: error\ndata: {error_payload}\n\n"

    # Return a streaming response
    return Response(stream_with_context(generate_validation_stream()), mimetype='text/event-stream')