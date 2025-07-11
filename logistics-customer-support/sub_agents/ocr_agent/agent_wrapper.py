import os
import sys
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
# Add the project root to Python's path to help with imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
import logistics_ocr_agent

from google.adk.agents import LoopAgent
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from common.task_manager import AgentWithTaskManager



class OcrAgent(AgentWithTaskManager):
    """
    An agent wrapper for OCR and document processing services.
    Provides a standardized interface for PAN card extraction and other OCR tasks.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "image/jpeg", "image/png", "application/pdf"]

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "ocr_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return "Processing your document and extracting information..."

    def _build_agent(self) -> LoopAgent:
        """
        Returns the pre-initialized root_agent from ocr_agent module.
        This agent already has MCP tools configured.
        """
        return logistics_ocr_agent.root_agent
