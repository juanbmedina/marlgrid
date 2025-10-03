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
            low=-1000.0,
            high=10000,
            shape=(num_sellers*num_buyers+num_buyers, ), 
            dtype=np.dtype("float64"))})
        
        # Action mappings
        self.power_step = 0.1   # small step for power allocations
        self.price_step = 1.0    # larger step for price

        self.action_spaces = {}

        ############################ Sellers actions space ############################

        # Build action deltas: first num_buyers dims = power_step, last dim = price_step
        seller_steps = [[-self.power_step, 0, self.power_step]] * num_buyers

        # Cartesian product across dimensions
        all_seller_combinations = list(itertools.product(*seller_steps))

        self.action_to_delta_sellers = {idx: list(combo) for idx, combo in enumerate(all_seller_combinations)}

        # Action spaces: 3 discrete actions (decrease, stay, increase)
        for seller in self.sellers:
            self.action_spaces[seller.name] = Discrete(len(self.action_to_delta_sellers))


        ############################ Buyers actions space ############################

        self.action_to_delta_buyers = {0: -self.price_step,
                                       1: -self.price_step,
                                       2: -self.price_step,
                                       3: 0.0,
                                       4: 0.0,
                                       5: 0.0,
                                       6: self.price_step,
                                       7: self.price_step,
                                       8: self.price_step,}
                                       
                                       
        
        # Action spaces: 3 discrete actions (decrease, stay, increase)
        for buyer in self.buyers:
            self.action_spaces[buyer.name] = Discrete(len(self.action_to_delta_buyers))

        self.t = 0
        self.max_steps = 100
        self.current_step = 0
        self.iter_count = 0

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
            seller.state = np.array([random.uniform(0.2, 0.4), random.uniform(0.2, 0.4)]) 
            global_state = np.concatenate((global_state, seller.state), axis = 0)

        for buyer in self.buyers:
            buyer.state =  random.randint(60, 90)  
            global_state = np.append(global_state, buyer.state)


        for agent in self.agents:
            obs[agent.name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}


        return obs



    def step(self, action_dict):
        """Execute one step in the environment"""
        t = 0
        global_state = []
        obs = {}
        rewards = {}
        agents_rewards = {}
        dones = {}

        self.sellers, self.buyers = self.split_agents(t)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        i = 0

        any_done = False 

        buyer_prices = self.get_buyers_price(t)
        

        ################### TAKE ACTIONS ###################

        for seller in self.sellers:
            delta = self.action_to_delta_sellers[action_dict[seller.name]]
            done, seller.state = seller.get_new_state(t, delta, num_sellers, num_buyers)
            global_state = np.concatenate((global_state, seller.state), axis = 0)
            if done:   # if this agent is out of bounds
                any_done = True

        for buyer in self.buyers:
            delta = self.action_to_delta_buyers[action_dict[buyer.name]]
            done, buyer.state = buyer.get_new_state(t, delta, num_sellers, num_buyers)
            global_state = np.append(global_state, buyer.state)
            if done:   # if this agent is out of bounds
                any_done = True

        ################### OBSERVE ###################

        for agent in self.agents:
            obs[agent.name] = {"obs": global_state}

        ################### GET REWARDS ###################

        for seller in self.sellers:
            others_power,others_price = self.get_others_power_price(seller, global_state)
            reward = seller.get_wellness(t,seller.state,buyer_prices,others_power,others_price)
            agents_rewards[seller.name] = reward

        for buyer in self.buyers:
            seller_power = self.get_sellers_power(t, buyer)
            others_selers_power = self.get_others_sellers_power(t, buyer)
            others_power,others_price = self.get_others_power_price(seller, global_state)
            reward = buyer.get_wellness(t,seller_power,buyer.state,others_selers_power,others_price)
            agents_rewards[buyer.name] = reward


        ################### DONES ###################

        # set all agents' done flags the same
        # dones = {agent.name: any_done for agent in self.agents}

        # episode ends either by constraint violation OR max steps
        # done_flag = any_done or (self.current_step >= self.max_steps)
        # done_flag = self.current_step >= self.max_steps
        # dones = {agent.name: done_flag for agent in self.agents}
        # dones["__all__"] = done_flag

        self.current_step += 1

        # if any_done:
        #     for agent in self.agents:
        #         rewards[agent.name] = -1e6
        #     done_flag = True
        #     dones = {agent.name: done_flag for agent in self.agents}
        #     dones["__all__"] = done_flag
        # elif self.current_step >= self.max_steps:
        #     for agent in self.agents:
        #         rewards[agent.name] = 0

        #     done_flag = True
        #     dones = {agent.name: done_flag for agent in self.agents}
        #     dones["__all__"] = done_flag

        # else:
        #     for agent in self.agents:
        #         rewards = agents_rewards
        #     done_flag = False
        #     dones = {agent.name: done_flag for agent in self.agents}
        #     dones["__all__"] = done_flag

        ####### Agents are out of the bounds #######
        if any_done:
            for agent in self.agents:
                rewards[agent.name] = -1e6
            done_flag = True
            dones = {agent.name: done_flag for agent in self.agents}
            dones["__all__"] = done_flag

        ####### Finish episode at max steps #######
        elif self.current_step >= self.max_steps:
            for agent in self.agents:
                rewards[agent.name] = 0
            done_flag = True
            dones = {agent.name: done_flag for agent in self.agents}
            dones["__all__"] = done_flag

        ####### Finish episode at wellness threshold #######
        elif np.mean(list(agents_rewards.values())) >= -50:
            for agent in self.agents:
                rewards[agent.name] = 1e+6
            done_flag = True
            dones = {agent.name: done_flag for agent in self.agents}
            dones["__all__"] = done_flag

        else:
            for agent in self.agents:
                rewards[agent.name] = -10
            done_flag = False
            dones = {agent.name: done_flag for agent in self.agents}
            dones["__all__"] = done_flag



        
    
        return obs, rewards, dones , {}
   

    def get_buyers_price(self, t):

        buyers_prices = []
        
        for buyer in self.buyers:
            buyers_prices.append(buyer.state)
        
        return np.array(buyers_prices)
    
    def get_sellers_power(self, t, buyer):

        seller_power = []
        
        for seller in self.sellers:
            seller_power.append(seller.state[buyer.buyer_id])
        
        return np.array(seller_power)
    
    def get_others_sellers_power(self, t, buyer):

        other_seller_power = []
        
        for seller in self.sellers:
            seller_power = list(seller.state)
            del seller_power[buyer.buyer_id]
            other_seller_power = np.concatenate((other_seller_power,np.array(seller_power)), axis=0)
        
        return other_seller_power

    def get_others_power_price(self, agent, global_state):
        others_power = []
        others_price = []
        for seller in self.sellers:
            if seller.name != agent.name:
                others_power = np.concatenate((others_power, seller.state), axis=0)
        for buyer in self.buyers:
            if buyer.name != agent.name:
                others_price = np.append(others_price, buyer.state)

        return others_power, others_price
    
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

        global_state = []

        state_info = {
            'episode': self.iter_count,
            'step': self.current_step,
        }

        for agent in self.agents:
            rol = agent.rol[t]

            if rol == 'S':  # Seller
                gen_power = np.sum(agent.state)
                total_generation += gen_power
                state_info[f'{agent.name}_power'] = gen_power
                state_info[f'{agent.name}_capacity'] = agent.net[t]

            elif rol == 'B':  # Buyer
                price = agent.state
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

        for seller in self.sellers:
            global_state = np.concatenate((global_state, seller.state), axis = 0)

        for buyer in self.buyers:
            global_state = np.append(global_state, buyer.state)
        
        for seller in self.sellers:
            others_power,others_price = self.get_others_power_price(seller, global_state)
            reward = seller.get_wellness(t,seller.state,buyer_prices,others_power,others_price)
            state_info[f'{seller.name}_reward'] = reward

        for buyer in self.buyers:
            seller_power = self.get_sellers_power(t, buyer)
            others_selers_power = self.get_others_sellers_power(t, buyer)
            others_power,others_price = self.get_others_power_price(seller, global_state)
            reward = buyer.get_wellness(t,seller_power,buyer.state,others_selers_power,others_price)
            state_info[f'{buyer.name}_reward'] = reward



        # for agent in self.agents:
        #     rol = agent.rol[t]
        #     power = agent.state
        #     price = agent.state
        #     others_power, others_price = self.get_others_power_price(agent, [a.state for a in self.agents])

        #     if rol == 'S':
        #         reward = agent.get_wellness(t, power, buyer_prices, others_power, others_price)
        #     elif rol == 'B':
        #         reward = agent.get_wellness(t, power, price, others_power, others_price)
        #     else:
        #         reward = 0.0

        #     state_info[f'{agent.name}_reward'] = reward

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
        i = 0
        j = 0
        for agent in self.agents:
            if agent.rol[t] == "S":
                agent.seller_id = i
                sellers.append(agent)
                i += 1
            elif agent.rol[t] == "B":
                agent.buyer_id = j
                buyers.append(agent)
                j =+ 1

        return sellers, buyers




