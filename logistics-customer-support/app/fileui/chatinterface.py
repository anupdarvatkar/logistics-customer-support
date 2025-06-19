"""
Defines FastAPI app and conversational endpoints for logistics support agents.
"""

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid

# --- Agent Interfaces ---

class Message(BaseModel):
    user_id: str
    text: Optional[str] = None
    intent: Optional[str] = None
    file_id: Optional[str] = None

class AgentResponse(BaseModel):
    agent: str
    text: str
    data: Optional[Dict[str, Any]] = None
    next_action: Optional[str] = None

class FAQAgent:
    def handle(self, message: Message) -> AgentResponse:
        return AgentResponse(agent="FAQAgent", text=f"FAQ: Answering '{message.text}'")

class BookingAgent:
    def handle(self, message: Message) -> AgentResponse:
        return AgentResponse(agent="BookingAgent", text="Booking: Please provide booking details.")

class TrackingAgent:
    def handle(self, message: Message) -> AgentResponse:
        return AgentResponse(agent="TrackingAgent", text="Tracking: Please provide your tracking number.")

class OnboardingAgent:
    def handle(self, message: Message) -> AgentResponse:
        return AgentResponse(agent="OnboardingAgent", text="Onboarding: Please provide your information.")

class IDValidationAgent:
    def handle(self, message: Message) -> AgentResponse:
        if message.file_id:
            # Simulate extraction for demonstration
            return AgentResponse(agent="IDValidationAgent",
                                 text="ID Extracted: Name: John Doe, ID#: 12345678. Is this correct?",
                                 next_action="await_confirmation")
        else:
            return AgentResponse(agent="IDValidationAgent", text="Please upload your ID card image.",
                                 next_action="await_file_upload")

# --- Orchestrator Agent ---

class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            "faq": FAQAgent(),
            "booking": BookingAgent(),
            "tracking": TrackingAgent(),
            "onboarding": OnboardingAgent(),
            "id_validation": IDValidationAgent()
        }

    def detect_intent(self, message: str) -> str:
        msg = message.lower()
        if "book" in msg:
            return "booking"
        if "track" in msg:
            return "tracking"
        if "onboard" in msg or "register" in msg:
            return "onboarding"
        if "id" in msg or "identity" in msg or "upload" in msg:
            return "id_validation"
        return "faq"

    def handle(self, message: Message) -> AgentResponse:
        intent = message.intent or self.detect_intent(message.text or "")
        agent = self.agents.get(intent, self.agents["faq"])
        return agent.handle(message)

orchestrator = OrchestratorAgent()

# --- FastAPI App and Endpoints ---

app = FastAPI()

@app.post("/chat/", response_model=AgentResponse)
async def chat(
    user_id: str = Form(...),
    text: Optional[str] = Form(None),
    intent: Optional[str] = Form(None)
):
    msg = Message(user_id=user_id, text=text, intent=intent)
    response = orchestrator.handle(msg)
    return response

@app.post("/upload_id/", response_model=AgentResponse)
async def upload_id(user_id: str = Form(...), file: UploadFile = File(...)):
    # Save file to disk or cloud (simulate here)
    file_id = str(uuid.uuid4())
    msg = Message(user_id=user_id, file_id=file_id, intent="id_validation")
    response = orchestrator.handle(msg)
    return response