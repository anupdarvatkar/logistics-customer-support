#This is equivalent to service class

#Generic imports
import uuid
import asyncio
from dotenv import load_dotenv
import uvicorn
load_dotenv()

#Google Imports
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search

from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from fastapi import FastAPI, Request

from .util.util import load_instruction_from_file
#from sub_agents.tracking_agent import agent
from .sub_agents.tracking_agent import tracking_agent
from .sub_agents.booking_agent import booking_agent

# Instantiate constants
APP_NAME = "logistics-customer-support"
#MODEL="gemini-2.0-flash"
MODEl = "gemini-2.5-pro-preview-05-06"

logistics_coordinator_agent = None
try:
    logistics_coordinator_agent = LlmAgent(
        name="logistics_coordinator",
        model=MODEl,
        description=(
            "Main agent for ILOG, a Logistics company's customer support"
            "Handles customer interaction, delegates to agents"
        ),
        instruction=load_instruction_from_file("logistics_coordinator_instructions.txt"),
        tools=[
            AgentTool(agent=tracking_agent),
            AgentTool(agent=booking_agent)
        ]
        #sub_agents=[agent]
    )
except Exception as e:
    print(f"Failed to create Logistics Coordinator Agent. Error: {e}")    

root_agent = logistics_coordinator_agent

# --- FastAPI setup ---
app = FastAPI()

@app.post("/")
async def chat(request: Request):
    request_json = await request.json()
    query = request_json.get("query")
    user_id = request_json.get("user_id", "default_user")
    session_id = request_json.get("session_id")

    if not query:
        return {"error": "Missing query"}, 400

    response, new_session_id = await call_agent(query, user_id, session_id)
    return {"response": response, "session_id": new_session_id}

# --- Uvicorn server setup (for local testing) ---
if __name__ == "__main__":
    print("Starting FastAPI server for Logistics Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Or your preferred port



# Session
session_service = InMemorySessionService()

#Root function call - can be called from other services
async def call_agent(query, user_id, session_id):
    
    final_response = "Did not get response from Agent"

    #Handle Null Session ID
    if session_id is None:
        session_id = str(uuid.uuid4())
        #print(f"Session Not found. Created new with id {session_id}")
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    else:
        #print(f"Getting existing session with ID: {session_id}")
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        #print(f"Got session: {session.id}")

    
    runner = Runner(
        agent=logistics_coordinator_agent, app_name=APP_NAME, session_service=session_service
    )

    content = types.Content(role="user", parts=[types.Part(text=query)])
    #events = runner.run(user_id=user_id, session_id=session_id, new_message=content)
    events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            #print("Agent Response: ", final_response)

    return final_response, session_id

#Calling the Agent via command line  
async def main():
    """
    This main function continuously gets user input in a loop until the user types 'exit'.
    """
    print("Welcome to the interactive input program!")
    print("Type 'exit' at any time to quit.")
    USER_ID = "8700"
    #SESSION_ID = "8700_sessions_1"
    session_id = None

    while True:
        user_input = input("Please enter something: ")

        if user_input.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break  # Exit the loop

        #print(f"You entered: {user_input}")
        #print(f"Session ID: {session_id}")
        agent_response, session_id = await call_agent(user_input, USER_ID, session_id=session_id)
        print(f"Response: {agent_response}")


if __name__ == "__main__":
    # asyncio.run(main())  # Comment out the original CLI main
    pass # The server is started above