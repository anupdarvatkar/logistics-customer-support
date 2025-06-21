import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to Python's path to help with imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)



from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill
from common.task_manager import AgentTaskManager

# Import the OcrAgent from local agent_wrapper file
from agent_wrapper import OcrAgent

  # Direct import from same folder

# Configuration
host = os.environ.get("A2A_HOST", "0.0.0.0")
port = int(os.environ.get("A2A_PORT", 8080))
PUBLIC_URL = os.environ.get("PUBLIC_URL", f"http://localhost:{port}")

def main():
    try:
        # Define agent capabilities
        capabilities = AgentCapabilities(streaming=True)
        
        # Define OCR skill
        skill = AgentSkill(
            id="document_processor",
            name="Document Processor",
            description="""
            This agent extracts structured information from various document types using OCR technology.
            It can process PAN cards, invoices, receipts, and other structured documents to extract
            relevant data fields in a structured JSON format.
            """,
            tags=["ocr", "document-processing"],
            examples=["Extract information from this PAN card", "Process this invoice"],
        )
        
        # Define agent card
        agent_card = AgentCard(
            name="OCR Document Processing Agent",
            description="""
            This agent extracts structured information from various document types using OCR technology.
            It can process PAN cards, invoices, receipts, and other structured documents to extract
            relevant data fields in a structured JSON format.
            """,
            url=f"{PUBLIC_URL}",
            version="1.0.0",
            defaultInputModes=OcrAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=["text/plain", "application/json"],
            capabilities=capabilities,
            skills=[skill],
        )
        
        # Create OcrAgent instance
        ocr_agent = OcrAgent()
        
        # Create and start the A2A server
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=ocr_agent),
            host=host,
            port=port,
        )
        
        logger.info(f"Starting OCR Agent server with Agent Card: {agent_card.name}")
        logger.info(f"Listening on {host}:{port}")
        
        server.start()
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
