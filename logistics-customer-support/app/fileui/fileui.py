import streamlit as st
import requests
import logging
import io
import sys

# Configure logging to capture messages in a buffer
log_buffer = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(log_buffer)]
)

# Function to display logs in Streamlit
def display_logs():
    logs = log_buffer.getvalue()
    if logs:
        st.text_area("Logs:", logs, height=200)


TOOLS_API_URL = "https://mcp-tool-server-service-203057862897.us-central1.run.app/upload_and_extract/"
OCR_AGENT_API_URL = "https://ocr-agent-service-203057862897.us-central1.run.app/extract_pan/"
st.title("ID Card OCR Uploader")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Upload and Extract"):
        with st.spinner("Uploading and extracting..."):
            file_bytes = uploaded_file.read()
            if not file_bytes:
                st.error("File could not be read or is empty.")
            else:
                files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                try:
                    response = requests.post(TOOLS_API_URL, files=files)
                    if response.ok:
                        data = response.json()
                        logging.info(f"API response data: {data}") # Logging the response data
                        st.success("Extraction successful!")
                        st.write("**GCS URI:**", data.get("gcs_uri", "Not Available"))
                        ocr = data.get("ocr_result", {})
                        if "error" in ocr:
                            st.error(f"OCR Error: {ocr['error']}")
                        else:
                            st.write("**Full Text:**")
                            st.code(ocr.get("full_text", ""), language="text")

                        if "pan_details" in data and "error" not in data["pan_details"]:
                            pan_details = data["pan_details"]
                            st.success("PAN Details Extracted!")
                            st.write("**PAN Number:**", pan_details.get("pan_number", "Not Available"))
                            st.write("**Name:**", pan_details.get("name", "Not Available"))
                            st.write("**Father's Name:**", pan_details.get("father_name", "Not Available"))
                            st.write("**Date of Birth:**", pan_details.get("dob", "Not Available"))
                            st.write("**Gender:**", pan_details.get("gender", "Not Available"))
                            st.markdown("**Full Text:**")
                    else:
                        st.error(f"Server returned error code: {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to connect or process: {e}")