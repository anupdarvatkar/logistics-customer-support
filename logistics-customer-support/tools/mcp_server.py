# adk_mcp_server.py
import asyncio
import json
import uvicorn
import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from dotenv import load_dotenv
from mcp import types as mcp_types 
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from ocr_api import tool_upload_and_extract, tool_upload_file, tool_extract_pan
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", 8080))

# Create FunctionTool instances for each tool
tool_objects = [
    FunctionTool(tool_upload_and_extract),
    FunctionTool(tool_upload_file),
    FunctionTool(tool_extract_pan),
]

# Create a dictionary for easier tool lookup by name
# CRITICAL FIX: This allows proper tool lookup by name
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
    # Convert all ADK tools to MCP format
    mcp_tools = []
    
    # FIXED: Convert each tool properly and add to the list
    for func in [tool_upload_and_extract, tool_upload_file, tool_extract_pan]:
        try:
            mcp_tool = adk_to_mcp_tool_type(func)
            mcp_tools.append(mcp_tool)
            logger.info(f"Advertising tool: {mcp_tool.name}")
        except Exception as e:
            logger.error(f"Failed to convert tool {func.__name__}: {e}")
    
    return mcp_tools

@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource]:
    """MCP handler to execute a tool call."""
    logger.info(f"Received call_tool request for '{name}' with args: {arguments}")

    # Look up the tool by name in our dictionary
    # FIXED: Proper dictionary lookup
    tool_to_call = available_tools.get(name)
    
    if tool_to_call:
        try:
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
    """Runs the MCP server over standard input/output."""
    # Use the stdio_server context manager from the MCP library
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(
            streams[0], streams[1], app.create_initialization_options()
        )

starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

if __name__ == "__main__":
    logger.info(f"Launching MCP Server on {APP_HOST}:{APP_PORT} exposing ADK tools...")
    try:
        # Fixed: Use proper asyncio.run pattern
        uvicorn.run(starlette_app, host=APP_HOST, port=APP_PORT)
    except KeyboardInterrupt:
        logger.info("MCP Server stopped by user")
    except Exception as e:
        logger.error(f"MCP Server encountered an error: {e}")
    finally:
        logger.info("MCP Server process exiting")
# --- End MCP Server ---