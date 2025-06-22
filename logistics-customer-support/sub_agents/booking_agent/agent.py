from google.adk import Agent
import random

MODEl = "gemini-2.5-pro-preview-05-06"

def booking_tool(collection_address: str, 
                 delivery_address: str, 
                 package_description: str,
                 package_weight:float,
                 service_level: str,
                 contact_email_address: str
                 ) -> dict:
    """Creates a new booking based on booking information shared

        Args:
            collection_address: Address from where the package needs to be collected
            delivery_address: Address to where the package neeeds to be delivered
            packagae_description: Brief description of the package
            package_weight: Weight of the package in kgs
            service_level: Choice between "ECONOMY" and "EXPRESS"
            contact_email_address: Valid Email address
        Returns:
            A dictionary indicating the status

        Example:
            >>>booking_tool(collection_address='W Cromwell Rd, London W14 8PB, United Kingdom', 
            delivery_address='Rimsky-Korssakovweg 9, 1323 LP Almere',
            package_description='Application documents', 
            package_weight='1',
            service_level='EXPRESS',
            contact_email_address='contact@gmail.com')
            {'status': 'success', 'booking_id':'10155'}
    """    

    print(f"Booking Details: \n collection_address: {collection_address} \n delivery_address: {delivery_address} \n  package_description:  {package_description} \n package_weight: {package_weight} \n" +
            f"service_level: {service_level} \n contact_email_address: {contact_email_address}")
    
    #Call the API to get the Booking Created
    booking_id = generate_booking_id()
    result = "{'status': 'success', 'booking_id':" + booking_id + "}"

    print(f"Result of Booking: {result}")
    
    return result


def generate_booking_id():
    last_two_digits = random.randint(0, 99)
    return f"101{last_two_digits:02d}"

booking_agent = None

BOOKING_AGENT_INSTRUCTIONS="""
        You are Booking Sub-Agent.
        Your role is to gather booking data from the users inputs
        
        Gather following information:
        - collection_address - Address from where the package needs to be collected
        - delivery_address - Address to which the package needs to be delivered
        - package_description - Brief description of the package
        - package_weight - Weight of the package in kg. Package weight cannot be greater than 45 kgs
        - service_level - Selection between "ECONOMY" and "EXPRESS"
        - contact_email_address - Valid email address 

        If any of the information is missing or incorrect, you must ask the user to provide the correct information.

        Do not proceed untill all the information is correct and complete
        Once all the information is gathered and validated, you must always ask for explicit confirmation from the user for every new booking request.
        This confirmation is a mandatory step.

        Tools:
        - booking_tool - Use this tool to create a new booking in the system

        Format the response from the tool  in a user friendly manner and share with the user. 
              
    """

try:
    booking_agent = Agent(
        model=MODEl,
        name="booking_agent",
        instruction=BOOKING_AGENT_INSTRUCTIONS,
        tools=[booking_tool]
    )
except Exception as e:
    print(f"Error in creating booking_agent. Error: {e}")