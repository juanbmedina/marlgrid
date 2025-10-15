import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from energy_env import P2PEnergyEnv


# Policy mapping for MARLlib
policy_mapping_dict = {
    "P2PEnergyEnv": {
        "description": "peer-to-peer energy market with sellers and buyers",
        "team_prefix": ("seller_", "buyer_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    }
}


class RLlibEnergyEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = P2PEnergyEnv()
        self.agents = [agent.group_name for agent in self.env.agents]
        self.num_agents = len(self.agents)

        # all agents share same spaces
        self.action_space = list(self.env.action_spaces.values())[0]
        self.observation_space = self.env.observation_space   # ✅ keep Dict, not Box

        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        self.agents = [agent.group_name for agent in self.env.agents]
        obs = {}
        for agent in self.agents:
            obs[agent] = {"obs": np.array(original_obs[agent]["obs"], dtype=np.float64)}
        return obs
    
    def step(self, action_dict):
        """One RLlib step for the MARLlib wrapper."""
        o, r, d, info = self.env.step(action_dict)
        self.agents = self.env.current_agents
        return o, r, d, info


    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def get_env_info(self):
        # ⚡ Return both groups’ spaces (like mate.py)
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "agents": self.agents,
            "episode_limit": self.env.max_steps,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info


# Register environment in MARLlib
ENV_REGISTRY["p2p_energy"] = RLlibEnergyEnv
COOP_ENV_REGISTRY["p2p_energy"] = RLlibEnergyEnv
