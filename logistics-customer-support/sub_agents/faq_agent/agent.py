from google.adk.agents import Agent
from .tools.rag_query import rag_query


MODEl = "gemini-2.5-pro-preview-05-06"


#DATASTORE_ID = "projects/primeval-aspect-461914-q7/locations/us-central1/ragCorpora/4611686018427387904"
#DATASTORE_ID = "projects/primeval-aspect-461914-q7/locations/us-central1/collections/default_collection/dataStores/4611686018427387904"
#DATASTORE_ID = "projects/primeval-aspect-461914-q7/locations/us-central1/ragCorpora/4611686018427387904"
DATASTORE_ID = "projects/primeval-aspect-461914-q7/locations/us-central1/collections/default_collection/dataStores/packaging_guidelines"

FAQ_AGENT_INSTRUCTIONS = """
    # 🧠 Packaging Questions Agent

    You are a helpful FAQ Agent focused on anwsering questions related to Packaging.

    ## Your Capabilities
    **Query Documents**: You can answer questions by retrieving relevant information using the rag_query tool.
    
    ## How to Approach User Requests
    
    When a user asks a question:
    If they're asking a packaging related question, use the `rag_query` tool to search the repository and provide a meaningful response.
    If there is reference to FedEx in the query result, replace it with GlideLogistics
    
    ## Using Tools
    1. `rag_query`: Query packaging_guidelines corpus to answer questions
       - Parameters:
         - query: The text question to ask
    
    
    ## Communication Guidelines
    - Be clear and concise in your responses.
    
    """


try:
    faq_agent = Agent(
        model=MODEl,
        name="faq_agent",
        instruction=FAQ_AGENT_INSTRUCTIONS,
        tools=[rag_query]
    )
    print(f"Agent {faq_agent.name} defined")
except Exception as e:
    print(f"Error in creating faq_agent. Error: {e}")

