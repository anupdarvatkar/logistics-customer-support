import streamlit as st
import requests

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
                        st.success("Extraction successful!")
                        st.write("**GCS URI:**", data.get("gcs_uri", "Not Available"))
                        ocr = data.get("ocr_result", {})
                        if "error" in ocr:
                            st.error(f"OCR Error: {ocr['error']}")
                        else:
                            st.write("**Name:**", ocr.get("name"))
                            st.write("**ID Number:**", ocr.get("id_number"))
                            st.write("**Date of Birth:**", ocr.get("dob"))
                            st.markdown("**Full Text:**")
                            st.code(ocr.get("full_text", ""), language="text")
                    else:
                        st.error(f"Server returned error code: {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to connect or process: {e}")