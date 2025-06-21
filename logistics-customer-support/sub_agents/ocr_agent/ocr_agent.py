import os
import sys
import asyncio
import logging
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MCP_SERVER_URL = "https://mcp-tool-server-service-203057862897.us-central1.run.app/sse"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Tool and Agent Initialization ---

async def get_tools_async():
    """
    Get tools from MCP server.
    Note: This function expects the MCP server to return proper ADK tool objects.
    """
    try:
        logger.info(f"Connecting to MCP server at {MCP_SERVER_URL} to load tools...")
        
        # IMPORTANT: Make sure any local tools you add are decorated with @tool
        tools, exit_stack = await MCPToolset.from_server(
            connection_params=SseServerParams(url=MCP_SERVER_URL, headers={})
        )
        
        logger.info(f"Loaded tools: {[tool.name for tool in tools]}")
        return tools, exit_stack
    except Exception as e:
        logger.error(f"Failed to load tools from MCP server: {e}")
        raise

async def get_agent_async():
    """
    Create and return the agent with tools from MCP server.
    """
    tools, exit_stack = await get_tools_async()
    try:
        root_agent = LlmAgent(
            model='gemini-2.0-flash',
            name='logistics_ocr_agent',
            instruction="""You are an expert information extraction agent. Your task is to extract details from Indian PAN card text with maximum accuracy and completeness.

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
""",
            tools=tools  # These must be proper ADK tool objects, not plain functions
        )
        logger.info(f"Agent initialized: {root_agent.name}")
        return root_agent, exit_stack
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

async def initialize():
    """
    Initialize the application state with the agent and exit stack.
    """
    try:
        root_agent, exit_stack = await get_agent_async()
        app.state.root_agent = root_agent
        app.state.exit_stack = exit_stack
        logger.info("OCR Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Unexpected error during module level initialization: {e}")
        raise

# --- PAN Extraction Logic ---
# This is a utility function, NOT a tool. Don't decorate it with @tool
def extract_pan_json(agent: LlmAgent, text: str) -> Dict[str, Any]:
    """
    Passes the input text to the LLM agent and returns the extracted PAN card details as a JSON object.
    This is not a tool, just a helper function.
    """
    if not text or not isinstance(text, str):
        return {"error": "No text provided or invalid text format."}
    if len(text.strip()) < 10:
        return {"error": "Text is too short to contain valid PAN card details."}
    try:
        response = agent.invoke(text)
        if not response:
            return {"error": "No data returned from extraction agent."}
        return response
    except Exception as e:
        error_message = f"Error extracting PAN card details: {str(e)}"
        logger.error(error_message)
        return {"error": error_message}

# --- FastAPI Endpoints ---

class PanExtractRequest(BaseModel):
    text: str

@app.post("/extract_pan")
async def extract_pan(request: PanExtractRequest):
    """
    API endpoint for PAN extraction.
    """
    agent: LlmAgent = app.state.root_agent
    try:
        result = extract_pan_json(agent, request.text)
        return result
    except Exception as e:
        logger.error(f"PAN extraction failed: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

# --- Main Entrypoint ---

if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    nest_asyncio.apply()
    try:
        asyncio.run(initialize())
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

