# Import the OcrAgent class from agent_wrapper
from .agent_wrapper import OcrAgent

# Import the ocr_agent module to make root_agent accessible
from . import logistics_ocr_agent

# Export these at the package level
__all__ = ['OcrAgent', 'ocr_agent']