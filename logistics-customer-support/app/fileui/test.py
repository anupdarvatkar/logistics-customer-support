from vertexai import agent_engines
AGENT_ENGINE_ID="projects/203057862897/locations/us-central1/reasoningEngines/6336397549942865920"
agent_list = agent_engines.list()
for agent in agent_list:
    if agent.resource_name == AGENT_ENGINE_ID:
        agent_engine = agent
        print("yes")
        break
else:
    agent_engine = None
