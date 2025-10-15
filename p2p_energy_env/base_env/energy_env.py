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

        seller_id = 0
        for seller in self.sellers:
            seller.group_name = 'seller_' + str(seller_id)
            seller_id += 1
        
        buyer_id = 0 
        for buyer in self.buyers:
            buyer.group_name = 'buyer_' + str(buyer_id)
            buyer_id += 1

        self.observation_space = GymDict({"obs": Box(
            low=0,
            high=1000,
            shape=(num_sellers*num_buyers+num_buyers, ), 
            dtype=np.dtype("float64"))})
        
        # Action mappings
        self.power_step = 0.1   # small step for power allocations
        self.price_step = 1.0    # larger step for price

        self.action_spaces = {}

        self.seller_state = np.arange(0.2, 0.51, 0.1)
        self.buyer_state = np.arange(55,95, 1)

        ############################ Sellers actions space ############################

        # # Build action deltas: first num_buyers dims = power_step, last dim = price_step
        # seller_steps = [[-self.power_step, 0, self.power_step]] * num_buyers

        # # Cartesian product across dimensions
        # all_seller_combinations = list(itertools.product(*seller_steps))

        # self.action_to_delta_sellers = {idx: list(combo) for idx, combo in enumerate(all_seller_combinations)}

        # # Action spaces: 3 discrete actions (decrease, stay, increase)
        # for seller in self.sellers:
        #     self.action_spaces[seller.name] = Discrete(len(self.action_to_delta_sellers))

        for agent in self.agents:
            self.action_spaces[agent.name] = Box(
                                                low=-self.power_step,
                                                high=self.power_step,
                                                shape=(2,),
                                                dtype=np.float32
                                            )


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
        # for buyer in self.buyers:
        #     self.action_spaces[buyer.name] = Discrete(len(self.action_to_delta_buyers))


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

        self.dones = {a.group_name: False for a in self.agents}

        self.iter_count += 1

        for seller in self.sellers:
            s1 = round(np.random.choice(self.seller_state),1)
            s2 = round(np.random.choice(self.seller_state),1)
            seller.state = np.array([s1, s2]) 
            global_state = np.concatenate((global_state, seller.state), axis = 0)

        for buyer in self.buyers:
            buyer.state =  round(np.random.choice(self.buyer_state),0)
            global_state = np.append(global_state, buyer.state)


        for seller in self.sellers:
            obs[seller.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
        for buyer in self.buyers:
            obs[buyer.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}


        return obs



    def step(self, action_dict):
        """Execute one step in the environment"""
        t = 0
        global_state = []
        obs = {}
        rewards = {}
        agents_rewards = {}
        self.current_agents = []
        agent_out = {a.group_name: False for a in self.agents}

        self.sellers, self.buyers = self.split_agents(t)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        buyer_prices = self.get_buyers_price(t)
        

        ################### TAKE ACTIONS ###################
        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                delta = action_dict[seller.group_name]
                done, seller.state = seller.get_new_state(t, delta, num_sellers, num_buyers)
                # self.dones[seller.group_name] = done
                # agent_out[seller.group_name] = done
            global_state = np.concatenate((global_state, seller.state), axis=0)

        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                delta = action_dict[buyer.group_name]
                done, buyer.state = buyer.get_new_state(t, delta[0]*10, num_sellers, num_buyers)
                # self.dones[buyer.group_name] = done
                # agent_out[buyer.group_name] = done
            global_state = np.append(global_state, buyer.state)

        ################### REWARD AND OBSERVATIONS ###################
        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                obs[seller.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
                others_power,others_price = self.get_others_power_price(seller, global_state)
                reward = seller.get_wellness(t,seller.state,buyer_prices,others_power,others_price)
                # print(seller.group_name, cost_fairness_reward)
                agents_rewards[seller.group_name] = reward 
        # cost_fairness_penalty = self.evaluate_cost_fairness()
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                obs[buyer.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
                seller_power = self.get_sellers_power(t, buyer)
                others_selers_power = self.get_others_sellers_power(t, buyer)
                others_power,others_price = self.get_others_power_price(seller, global_state)
                reward = buyer.get_wellness(t,seller_power, float(buyer.state),others_selers_power,others_price)
                agents_rewards[buyer.group_name] = reward


        self.current_step += 1

        constraint_reward = self.evaluate_constraints(t)
        # print(constraint_reward)

        for agent in self.agents:
            if not self.dones[agent.group_name]:
                agents_rewards[agent.group_name] += 1000 * constraint_reward 

        ################### ACTIVE AGENTS ###################
        for agent in self.agents:
            if not self.dones[agent.group_name]:
                self.current_agents.append(agent)

        ################### DONE EPISODE ###################
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                self.dones[agent.group_name] = True
        # Global done
        self.dones["__all__"] = all(self.dones[a.group_name] for a in self.agents)


        rewards = agents_rewards

        dones = self.dones

        return obs, rewards, dones , {}
    
    def evaluate_constraints(self, t):
        total_demand = sum(b.net[t] for b in self.buyers)
        total_gen = sum(s.net[t] for s in self.sellers)
        total_power = sum(sum(s.state) for s in self.sellers)

        # print(total_demand, total_gen, total_power)

        # desired total = min(total_gen, total_demand)
        target = min(total_gen, total_demand)

        # Smooth penalty
        deviation = abs(total_power - target)
        # print(deviation)

        reward_signal = np.exp(-10*deviation)  # near 1 if very close, decays rapidly otherwise
        # print(reward_signal)
        return reward_signal

    def evaluate_cost_fairness(self):
        """
        Returns a continuous penalty (negative value) when generation cost exceeds total cost.
        0 means fair cost (Hg <= total_cost).
        """
        penalties = {}
        for seller in self.sellers:
            Hg = seller.get_generation_costs(seller.state)
            total_cost = 0.0
            for buyer in self.buyers:
                # buyer.state = price; seller.state[buyer.buyer_id] = power sold to buyer
                total_cost += buyer.state * seller.state[buyer.buyer_id]
            # print(f"{seller.group_name} total_cost: {total_cost}, Hg: {Hg}" )
            if Hg <= total_cost:
                penalty = np.exp(-1*(total_cost-Hg))
            else:
                penalty = 1

            penalties[seller.group_name] = penalty
        
        mean_penalties = sum(penalties.values())/len(penalties)

        return -mean_penalties



   

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




