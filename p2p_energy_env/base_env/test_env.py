from energy_env import P2PEnergyEnv
import random

env = P2PEnergyEnv()

agents = env.agents
action_dict = {}
for agent in agents:
    action_dict[agent.name] = [1,1,1]

# print("Action Dict: ",action_dict)

# for agent in agents:
#     print(f"State {agent.name}", agent.state)

env.step(action_dict)

# for agent in agents:
#     print(f"State {agent.name}", agent.state)

# print("Env agents: ", env.agents)
# print("Wellness function: ", env.test_agents())
# print("Observation space: ", env.observation_space)
# print("Action space: ", env.action_spaces)
