
from google.adk import Agent

MODEl = "gemini-2.5-pro-preview-05-06"


def tracking_tool(tracking_number: str) -> dict:
    """Retrieves tracking based on tracking number
    
        Args:
            tracking_number(str): Tracking number of the shipment. It is a 9 digit number

        Returns:
            dict: A dictionary containing the tracing details

    """

    mock_tracking_db = {
        "123456789" : {"status": "delivered", "delivery_date":"11-June-2025"},
        "987654321" : {"status": "delivered", "delivery_date":"12-June-2025"},
        "123459876" : {"status": "depart_hub", "hub_depart_date":"09-June-2025"},
    }

    if tracking_number in mock_tracking_db:
        data = mock_tracking_db[tracking_number]
        print(f"Got Tracking Data: {data}")
        return data
    else:
        print(f"No tracking data found for tracking number: {tracking_number}")
        return "No data found for tracking number {tracking_number}"

tracking_agent = None

TRACKING_AGENT_INSTRUCTIONS="""You are Tracking Sub-Agent. 
            Your role is to get tracking data using the tracking_tool based on the tracking number. 
            The tracking_number is a 9 digit number. e.g., 987612345
            Format the response based on the tools response in a user friendly format
            """

try:
    tracking_agent = Agent(
        model=MODEl,
        name="tracking_agent",
        instruction=TRACKING_AGENT_INSTRUCTIONS,
        tools=[tracking_tool]
    )
    print(f"Agent {tracking_agent.name} defined")
except Exception as e:
    print(f"Error in creating tracking_agent. Error: {e}")

