from gym.spaces import Discrete, Box
from gym.spaces import Dict as GymDict, Box
from pettingzoo import ParallelEnv
from typing import Optional
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime

import json

from energy_agent import EnergyAgent


class P2PEnergyEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "energy_market_v1"}

    def __init__(self):

        json_path = '/workspace/marlgrid/base_env/profiles/agents_profiles.json'

        with open(json_path, 'r') as f:
            data = json.load(f)
        self.n_agents = len(data.items()) # number of agents
        self.agents = []
        for name, profiles in data.items():
            agent = EnergyAgent(
                name=name,
                n_agents=self.n_agents,
                consumer_profile=profiles["consumer_profile"],
                generator_profile=profiles["generator_profile"]
            )
            self.agents.append(agent)

        # self.n_agents = len(self.agents) # number of agents

        self.observation_space = GymDict({"obs": Box(
            low=0.0,
            high=1.0,
            shape=(self.n_agents*self.n_agents, ), # Each agent has [value, profit]
            dtype=np.dtype("float64"))})
        
        # Action spaces: 3 discrete actions (decrease, stay, increase)
        self.action_spaces = {agent.name: Discrete(3) for agent in self.agents}

        # Action mappings
        self.action_step = 0.1
        self.action_to_delta = {
            0: -self.action_step,  # Decrease
            1: 0.0,                # Stay
            2: self.action_step    # Increase
        }

        self.t = 0
        self.max_steps = 1000
        self.current_step = 0


    def reset(self):
        self.current_step = 0
        t = 0
        global_state = []
        obs = {}

        for agent in self.agents:
            agent.state = np.zeros(self.n_agents)   
            global_state.append(agent.state)

        for agent in self.agents:
            obs[agent.name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}


        return obs



    def step(self, action_dict):
        """Execute one step in the environment"""
        t = 0
        global_state = []
        obs = {}
        rewards = {}

        for agent in self.agents:
            delta = self._action_to_delta(action_dict[agent.name])
            agent.state = agent.get_new_state(t, delta)
            global_state.append(agent.state)
            # print(f"Agent: {agent.name}, Role: {agent.rol[t]}, State: {agent.state}")

        
        # print(f"Global State: {global_state}")
        for agent in self.agents:
            rol = agent.rol[t]

            power = agent.state[:-1]
            price = agent.state[-1]

            others_power, others_price = self.get_others_power_price(agent, global_state)

            if rol == 'S':
                reward = agent.get_wellness(t,power,others_price,others_power,others_price)
            elif rol == 'B':
                reward = agent.get_wellness(t,power,price,others_power,others_price)

            obs[agent.name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}

            rewards[agent.name] = reward

            # print(f"Reward: {reward}")

                            # Handle termination
        done_flag = self.current_step >= self.max_steps
        dones = {agent: done_flag for agent in self.agents}
        dones["__all__"] = done_flag

        self.current_step += 1
        # print("Observation: ", obs)
    
        return obs, rewards, dones , {}


    def get_others_power_price(self, agent, global_state):
        gs = global_state.copy()
        del gs[agent.id-1]
        others_state = global_state
        others_power = np.concatenate([other_state[:-1] for other_state in others_state])
        others_price = np.array([other_state[-1] for other_state in others_state])

        return others_power, others_price

    
    def _action_to_delta(self, actions): 
        delta_actions = []
        for action in actions:
            delta = self.action_to_delta[action]
            delta_actions.append(delta)
        return delta_actions


