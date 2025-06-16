import os
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


ocr_extract_agent = Agent(
        model=MODEL,
        name="pan_extractor_agent",
        instruction=PAN_EXTRACTION_INSTRUCTION,)