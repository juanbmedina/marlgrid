import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from energy_env import P2PEnergyEnv


# policy mapping for MARLlib
policy_mapping_dict = {
    "P2PEnergyEnv": {
        "description": "peer-to-peer energy market with sellers and buyers",
        "team_prefix": ("agent_",),    # agents will be agent_0, agent_1, ...
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    }
}


class RLlibEnergyEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = P2PEnergyEnv()
        self.agents = [agent.name for agent in self.env.agents]
        self.num_agents = len(self.agents)

        # all agents share same spaces
        self.action_space = list(self.env.action_spaces.values())[0]
        self.observation_space = self.env.observation_space   # ✅ keep Dict, not Box

        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for agent in self.agents:
            obs[agent] = {"obs": np.array(original_obs[agent]["obs"], dtype=np.float64)}
        return obs

    def step(self, action_dict):
        # print("#### Acciones ####: ",action_dict)
        o, r, d, info = self.env.step(action_dict)
        obs = {}
        rewards = {}
        dones = {}
        for agent in self.agents:
            obs[agent] = {"obs": np.array(o[agent]["obs"], dtype=np.float64)}
            rewards[agent] = r[agent]
            dones[agent] = d[agent]
        dones["__all__"] = d["__all__"]
        return obs, rewards, dones, info


    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.max_steps,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info


# Register with MARLlib
ENV_REGISTRY["p2p_energy"] = RLlibEnergyEnv
COOP_ENV_REGISTRY["p2p_energy"] = RLlibEnergyEnv
