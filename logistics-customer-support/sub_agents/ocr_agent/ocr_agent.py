import asyncio
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
import logging
import os
import nest_asyncio
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Any, AsyncIterable, Dict, Optional
from google.adk.agents import LoopAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from common.task_manager import AgentWithTaskManager



# Load environment variables
load_dotenv()
MCP_SERVER_URL = "https://mcp-too-server-service-203057862897.us-central1.run.app/sse"

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Global variables ---
root_agent: LlmAgent | None = None
exit_stack: AsyncExitStack | None = None


class PanRequest(BaseModel):
    text: str

async def get_tools_async():
    """
    Asynchronously creates an MCP Toolset connected to the MCP server.
    """


    print("Attempting to connect to MCP Filesystem server...")
    tools, exit_stack = await MCPToolset.from_server(
      connection_params=SseServerParams(url=MCP_SERVER_URL, headers={})
  )
    
    return tools, exit_stack

async def get_agent_async():
    """
    Asynchronously creates the MCP Toolset and the LlmAgent.
    """
    tools, exit_stack = await get_tools_async()

    root_agent = LlmAgent(
        model='gemini-2.0-flash',
        name='logistics_ocr_agent',
        instruction="""
        You are a specialized OCR and document processing agent for logistics customer support.
        
        You can perform the following tasks:
        1. Upload files to Google Cloud Storage
        2. Extract text from images using OCR
        3. Extract structured PAN card details from text
        
        Guidelines:
        - For PAN card extraction, you need text containing PAN card information
        - The extraction will identify fields like PAN number, name, father's name, DOB, and gender
        - Always verify the inputs are valid before calling tools
        """,
        tools=tools
    )
    log.info("LlmAgent created with MCP tools.")

    # Return both the agent and the exit_stack needed for cleanup
    return root_agent, exit_stack

async def initialize():
    """Initializes the global root_agent and exit_stack."""
    global root_agent, exit_stack
    if root_agent is None:
        log.info("Initializing agent...")
        root_agent, exit_stack = await get_agent_async()
        if root_agent:
            log.info("Agent initialized successfully.")
        else:
            log.error("Agent initialization failed.")
    else:
        log.info("Agent already initialized.")

async def extract_pan_json(text: str) -> dict:
    """
    Passes the input text to the agent to extract PAN card details.
    """
    global root_agent
    
    if not root_agent:
        await initialize()
    
    if not text or not isinstance(text, str):
        return {"error": "No text provided or invalid text format."}
    
    if len(text.strip()) < 10:
        return {"error": "Text is too short to contain valid PAN card details."}
    
    try:
        # Construct a prompt for the agent to extract PAN details
        prompt = f"""
        Extract all PAN card details from the following text. Return only the structured data:
        
        {text}
        """
        
        # Call the agent with the prompt
        response = await root_agent.generate_content(prompt)
        
        # Process the response to extract the PAN details
        # The agent will call the tool_extract_pan via MCP
        result = response.text
        
        return result
    except Exception as e:
        error_message = f"Error extracting PAN card details: {str(e)}"
        log.error(error_message)
        return {"error": error_message}

def _cleanup_sync():
    """Synchronous wrapper to attempt async cleanup."""
    if exit_stack:
        log.info("Attempting to close MCP connection via atexit...")
        try:
            asyncio.run(exit_stack.aclose())
            log.info("MCP connection closed via atexit.")
        except Exception as e:
            log.error(f"Error during atexit cleanup: {e}", exc_info=True)


nest_asyncio.apply()

log.info("Running agent initialization at module level using asyncio.run()...")
try:
    asyncio.run(initialize())
    log.info("Module level asyncio.run(initialize()) completed.")
except RuntimeError as e:
    log.error(f"RuntimeError during module level initialization (likely nested loops): {e}", exc_info=True)
except Exception as e:
    log.error(f"Unexpected error during module level initialization: {e}", exc_info=True)

