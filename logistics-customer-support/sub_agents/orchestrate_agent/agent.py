
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from host_agent import HostAgent
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import BaseAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
import logging 
import nest_asyncio # Import nest_asyncio
import atexit


load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
# --- Global variables ---
# Define them first, initialize as None

# --- Configuration ---
# It's better to get this from environment variables or a config file
# Defaulting to empty list if not set. Adjust as needed.
REMOTE_AGENT_ADDRESSES_STR = "https://ocr-agent-service-203057862897.us-central1.run.app"
log.info(f"Remote Agent Addresses String: {REMOTE_AGENT_ADDRESSES_STR}")
REMOTE_AGENT_ADDRESSES = [addr.strip() for addr in REMOTE_AGENT_ADDRESSES_STR.split(',') if addr.strip()]
log.info(f"Remote Agent Addresses: {REMOTE_AGENT_ADDRESSES}")

# --- Agent Initialization ---
# Instantiate the HostAgent logic class
# You might want to add a task_callback here if needed, similar to run_orchestrator.py
host_agent_logic = HostAgent(remote_agent_addresses=REMOTE_AGENT_ADDRESSES)

# Create the actual ADK Agent instance
root_agent: BaseAgent = host_agent_logic.create_agent()
log.info(f"Orchestrator root agent '{root_agent.name}' created.")