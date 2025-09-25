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
import itertools

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
                generator_profile=profiles["generator_profile"],
                cost_params=profiles["cost_params"]
            )
            self.agents.append(agent)

        self.sellers, self.buyers = self.split_agents(t=0)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        self.observation_space = GymDict({"obs": Box(
            low=0.0,
            high=10000,
            shape=((num_buyers+1)*self.n_agents, ), 
            dtype=np.dtype("float64"))})
        
        # Action mappings
        self.power_step = 0.01   # small step for power allocations
        self.price_step = 1    # larger step for price

        # Build action deltas: first num_buyers dims = power_step, last dim = price_step
        steps = [[-self.power_step, 0, self.power_step]] * num_buyers \
            + [[-self.price_step, 0, self.price_step]]

        # Cartesian product across dimensions
        all_combinations = list(itertools.product(*steps))

        # Map each index to a delta vector
        self.action_to_delta = {idx: list(combo) for idx, combo in enumerate(all_combinations)}

        self.iter_count = 0

        # Action spaces: 3 discrete actions (decrease, stay, increase)
        self.action_spaces = {agent.name: Discrete(len(self.action_to_delta)) for agent in self.agents}

        # print(self.action_to_delta)

        self.t = 0
        self.max_steps = 1000
        self.current_step = 0

        # Save printed states
        self.csv_initialized = False
        self.csv_file = f"market_log_all_episodes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"


    def reset(self):
        self.sellers, self.buyers = self.split_agents(t=0)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)
        self.current_step = 0
        global_state = []
        obs = {}

        self.iter_count += 1

        for seller in self.sellers:
            seller.state = np.array([random.uniform(0.1, 0.5), random.uniform(0.1, 0.5), 0]) 
            global_state.append(seller.state)


        for buyer in self.buyers:
            buyer.state = np.array([0.0, 0.0, random.uniform(50, 100)])   
            global_state.append(buyer.state)

        # for agent in self.agents:
        #     # agent.state = np.zeros(num_buyers+1)   
        #     agent.state = np.array([0.0, 0.0, 75])   
        #     global_state.append(agent.state)

        for agent in self.agents:
            obs[agent.name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}


        return obs



    def step(self, action_dict):
        """Execute one step in the environment"""
        t = 0
        global_state = []
        obs = {}
        rewards = {}

        self.sellers, self.buyers = self.split_agents(t)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)
        i = 0
        for agent in self.agents:
            if agent.rol[t] == 'B':
                agent.buyer_id = i
                i +=1
            delta = self._action_to_delta(action_dict[agent.name])
            agent.state = agent.get_new_state(t, delta, num_sellers, num_buyers)
            global_state.append(agent.state)
            # print(f"Agent: {agent.name}, Role: {agent.rol[t]}, State: {agent.state}")

        # self.enforce_constraints(t)
        # self.community_constrains(t)

        buyer_prices = self.get_buyers_price(t)
        for agent in self.agents:
            rol = agent.rol[t]

            power = agent.state[:-1]
            price = agent.state[-1]

            others_power, others_price = self.get_others_power_price(agent, global_state)

            if rol == 'S':
                reward = agent.get_wellness(t,power,buyer_prices,others_power,others_price)
            elif rol == 'B':
                reward = agent.get_wellness(t,power,price,others_power,others_price)

            obs[agent.name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}

            rewards[agent.name] = reward


                            # Handle termination
        done_flag = self.current_step >= self.max_steps
        dones = {agent.name: done_flag for agent in self.agents}
        dones["__all__"] = done_flag

        self.current_step += 1

        
    
        return obs, rewards, dones , {}
   

    def get_buyers_price(self, t):

        buyers_prices = []
        
        for agent in self.agents:
            if agent.rol[t]=='B':
                buyers_prices.append(agent.state[-1])
        
        return np.array(buyers_prices)

    def get_others_power_price(self, agent, global_state):
        gs = global_state.copy()
        del gs[agent.id-1]
        others_state = gs
        others_power = np.concatenate([other_state[:-1] for other_state in others_state])
        others_price = np.array([other_state[-1] for other_state in others_state])

        return others_power, others_price


    # Esto funciona mas o menos, revisar con test_env
    def enforce_constraints(self, t):
        """
        Enforce generation and demand constraints based on profiles.
        global_state: list of all agent states (for both sellers and buyers)
        """
        for seller in self.sellers:
            # Sum of power dispatched by seller j
            total_power = np.sum(seller.state[:-1])
            G_j = seller.net[t]   # generator capacity at time t

            if total_power > G_j:
                # Normalize proportions to fit capacity
                seller.state[:-1] = seller.state[:-1]
            seller.state[:-1] = np.clip(seller.state[:-1], 0, G_j)

        for buyer in self.buyers:
                D_i = buyer.net[t]
                total_incoming_power = 0

                for seller in self.sellers:
                    total_incoming_power += seller.state[buyer.buyer_id]

                if total_incoming_power > D_i:
                    for seller in self.sellers:
                        seller.state[seller.buyer_id] = seller.state[seller.buyer_id]

    def community_constrains(self, t):
        total_gen = 0
        total_con = 0

        for seller in self.sellers:
            total_gen += seller.net[t]
        for buyer in self.buyers:
            total_con += buyer.net[t]

        if total_con <= total_gen:
            pass

        print(f"Community constrains: Power: {total_gen}  Demand: {total_con}")
        
    def _action_to_delta(self, action):
        return self.action_to_delta[int(action)]
    
    def render(self, mode="human"):
        """Render the environment state"""
        if mode == "human":
            self.print_state(t=0)

    def print_state(self, t):
        """Print current state of the energy market and log to CSV"""

        # Initialize totals
        total_generation = 0.0
        total_demand = 0.0
        total_price = 0.0

        state_info = {
            'episode': self.iter_count,
            'step': self.current_step,
        }

        for agent in self.agents:
            rol = agent.rol[t]

            if rol == 'S':  # Seller
                gen_power = np.sum(agent.state[:-1])
                total_generation += gen_power
                state_info[f'{agent.name}_power'] = gen_power
                state_info[f'{agent.name}_capacity'] = agent.net[t]

            elif rol == 'B':  # Buyer
                price = agent.state[-1]
                demand = agent.net[t]
                total_price += price
                total_demand += demand
                state_info[f'{agent.name}_price'] = price
                state_info[f'{agent.name}_demand'] = demand

        # Add totals
        state_info['total_generation'] = total_generation
        state_info['total_demand'] = total_demand
        state_info['total_price'] = total_price
        
        # --- NEW: log rewards ---
        # compute rewards the same way as in step()
        buyer_prices = self.get_buyers_price(t)
        for agent in self.agents:
            rol = agent.rol[t]
            power = agent.state[:-1]
            price = agent.state[-1]
            others_power, others_price = self.get_others_power_price(agent, [a.state for a in self.agents])

            if rol == 'S':
                reward = agent.get_wellness(t, power, buyer_prices, others_power, others_price)
            elif rol == 'B':
                reward = agent.get_wellness(t, power, price, others_power, others_price)
            else:
                reward = 0.0

            state_info[f'{agent.name}_reward'] = reward

        # Write to CSV
        with open("market_log.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=state_info.keys())
            if f.tell() == 0:  # write header only once
                writer.writeheader()

            if self.current_step%1 == 0:
                writer.writerow(state_info)

    def split_agents(self, t=0):
        """
        Count how many sellers and buyers exist at time t.
        Returns (num_sellers, num_buyers).
        """
        sellers = []
        buyers = []

        for agent in self.agents:
            if agent.rol[t] == "S":
                sellers.append(agent)
            elif agent.rol[t] == "B":
                buyers.append(agent)

        return sellers, buyers




