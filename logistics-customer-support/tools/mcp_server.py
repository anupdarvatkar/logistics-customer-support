# adk_mcp_server.py

import asyncio
import json
import uvicorn
import os
import sys
import logging
import base64
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from dotenv import load_dotenv
from mcp import types as mcp_types 
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# Import the original tool functions
from ocr_api import tool_upload_and_extract, tool_upload_file, tool_extract_pan

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", 8080))

# --- Create wrapped versions of the tool functions with string parameters ---

def wrapped_upload_file(file_bytes: str) -> Dict:
    """Upload a file and return its ID.
    
    Args:
        file_bytes: The file content as a base64-encoded string
        
    Returns:
        Dict: Information about the uploaded file
    """
    # Convert base64 string to bytes inside the wrapper
    try:
        binary_data = base64.b64decode(file_bytes)
        return tool_upload_file(binary_data)
    except Exception as e:
        logger.error(f"Error decoding base64 in wrapped_upload_file: {e}")
        return {"error": f"Failed to decode base64: {str(e)}"}

def wrapped_upload_and_extract(file_bytes: str) -> Dict:
    """Upload a file and extract text from it.
    
    Args:
        file_bytes: The file content as a base64-encoded string
        
    Returns:
        Dict: Extracted text and file information
    """
    # Convert base64 string to bytes inside the wrapper
    try:
        binary_data = base64.b64decode(file_bytes)
        return tool_upload_and_extract(binary_data)
    except Exception as e:
        logger.error(f"Error decoding base64 in wrapped_upload_and_extract: {e}")
        return {"error": f"Failed to decode base64: {str(e)}"}

def wrapped_extract_pan(text: str) -> Dict:
    """Extract PAN card details from text.
    
    Args:
        text: The text to extract PAN details from
        
    Returns:
        Dict: Extracted PAN card details
    """
    return tool_extract_pan(text)

# Create FunctionTool objects
upload_file_tool = FunctionTool(wrapped_upload_file)
upload_and_extract_tool = FunctionTool(wrapped_upload_and_extract)
extract_pan_tool = FunctionTool(wrapped_extract_pan)

# Use the wrapped tools
tool_objects = [
    upload_and_extract_tool,
    upload_file_tool,
    extract_pan_tool
]

# Create a dictionary for easier tool lookup by name
available_tools = {
    tool.name: tool for tool in tool_objects
}

logger.info(f"Initialized tools: {list(available_tools.keys())}")

# Create a named MCP Server instance
app = Server("adk-tool-mcp-server")
sse = SseServerTransport("/messages/")

@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    """MCP handler to list available tools."""
    mcp_tools = []
    
    for tool in tool_objects:
        try:
            mcp_tool = adk_to_mcp_tool_type(tool)
            mcp_tools.append(mcp_tool)
            logger.info(f"Advertising tool: {mcp_tool.name}")
        except Exception as e:
            logger.error(f"Failed to convert tool {tool.name}: {e}")
    
    return mcp_tools

@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource]:
    """MCP handler to execute a tool call."""
    logger.info(f"Received call_tool request for '{name}'")

    # Look up the tool by name in our dictionary
    tool_to_call = available_tools.get(name)
    
    if tool_to_call:
        try:
            # No need to convert base64 here - it's handled in the wrapper functions
            adk_response = await tool_to_call.run_async(
                args=arguments,
                tool_context=None,
            )
            logger.info(f"ADK tool '{name}' executed successfully")

            response_text = json.dumps(adk_response, indent=2)
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            logger.error(f"Error executing ADK tool '{name}': {e}")
            error_text = json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        # Handle calls to unknown tools
        logger.warning(f"Tool '{name}' not found. Available tools: {list(available_tools.keys())}")
        error_text = json.dumps({"error": f"Tool '{name}' not implemented."})
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Remote Server ---
async def handle_sse(request):
    """Runs the MCP server over SSE."""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(
            streams[0], streams[1], app.create_initialization_options()
        )

# Create FastAPI app for OCR endpoint
ocr_api_app = FastAPI()

@ocr_api_app.post("/upload_and_extract/")
async def upload_and_extract(file: UploadFile = File(...)):
    file_bytes = await file.read()
    return tool_upload_and_extract(file_bytes, file.filename)


starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
        Mount("/", app=ocr_api_app),  # Mounts FastAPI app at root
    ],
)

if __name__ == "__main__":
    logger.info(f"Launching MCP Server on {APP_HOST}:{APP_PORT} exposing ADK tools...")
    try:
        uvicorn.run(starlette_app, host=APP_HOST, port=APP_PORT)
    except KeyboardInterrupt:
        logger.info("MCP Server stopped by user")
    except Exception as e:
        logger.error(f"MCP Server encountered an error: {e}")
    finally:
        logger.info("MCP Server process exiting")
# --- End MCP Server ---