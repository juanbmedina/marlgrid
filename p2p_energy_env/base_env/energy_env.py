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
            seller.lagrange = np.array([0.0, 0.0]) 
        
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

        for agent in self.agents:
            self.action_spaces[agent.name] = Box(
                                                low=-self.power_step,
                                                high=self.power_step,
                                                shape=(2,),
                                                dtype=np.float32
                                            )

        self.t = 0
        self.max_steps = 100
        self.current_step = 0
        self.iter_count = 0

        # === Dual ascent parameters ===
        self.lambda_alpha = 0.1          # learning rate para actualización dual
        self.lambda_clip_min = 0.0
        self.lambda_clip_max = 1000.0     # límite superior para evitar explosiones
        self.cost_threshold = 0.0         # d en J_C - d <= 0
        self.cost_ma_alpha = 0.05         # suavizado EMA para la violación


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
        self.previous_wellness = {}


        self.dones = {a.group_name: False for a in self.agents}

        self.iter_count += 1

        for seller in self.sellers:
            s1 = round(np.random.choice(self.seller_state),1)
            s2 = round(np.random.choice(self.seller_state),1)
            seller.state = np.array([s1, s2]) 
            seller.lagrange = np.array([0.0, 0.0, 0.0]) 
            seller.cost_ma = 0.0                    # NEW
            global_state = np.concatenate((global_state, seller.state), axis = 0)


        for buyer in self.buyers:
            buyer.state =  round(np.random.choice(self.buyer_state),0) 
            buyer.lagrange = np.array([0.0, 0.0])
            buyer.cost_ma = 0.0  
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
        self.current_agents = []
        agent_out = {a.group_name: False for a in self.agents}
        wellness = {}

        self.sellers, self.buyers = self.split_agents(t)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        P = np.zeros([num_sellers, num_buyers])
        C = np.zeros([num_buyers])

        ################### TAKE ACTIONS ###################
        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                delta = action_dict[seller.group_name]
                done, seller.state = seller.get_new_state(t, delta[:1])
                # self.dones[seller.group_name] = done
                # agent_out[seller.group_name] = done
            global_state = np.concatenate((global_state, seller.state), axis=0)
            P[seller.seller_id] = seller.state 

        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                delta = action_dict[buyer.group_name]
                done, buyer.state = buyer.get_new_state(t, delta[0]*10)
                # self.dones[buyer.group_name] = done
                # agent_out[buyer.group_name] = done
            global_state = np.append(global_state, buyer.state)
            C[buyer.buyer_id] = buyer.state
        
        comm_penalty = self.check_community_constraints(t)

        ################### REWARD AND OBSERVATIONS ###################
        for seller in self.sellers:
            penalty2 = 0
            if not self.dones[seller.group_name]:
                obs[seller.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
                penalty1 = self.check_power_constrain_seller(t,P,seller) 
                for buyer in self.buyers:
                    penalty2 += self.check_power_constrain_buyer(t, P, buyer)
                l1 = seller.lagrange[0]
                l2 = seller.lagrange[1]
                l3 = seller.lagrange[2]
                wellness[seller.group_name] = seller.get_wellness(t,P,C) - (l1*penalty1 + l2*penalty2 + l3*comm_penalty)
        
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                obs[buyer.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
                penalty3 = self.check_cost_constrain(P,C)
                l4 = buyer.lagrange[0]
                wellness[buyer.group_name] = buyer.get_wellness(t,P,C) - (l4*penalty3)
        
        # === DUAL ASCENT UPDATE (Lagrangian primal-dual RL) ===

        # --- Sellers ---
        for seller in self.sellers:

            # Actualización dual λ_power
            seller.lagrange[0] += self.lambda_alpha * (penalty1 - self.cost_threshold)
            seller.lagrange[1] += self.lambda_alpha * (penalty2 - self.cost_threshold)
            seller.lagrange[2] += self.lambda_alpha * (comm_penalty - self.cost_threshold)

            # Aplicar clipping
            seller.lagrange[0] = np.clip(
                seller.lagrange[0],
                self.lambda_clip_min,
                self.lambda_clip_max
            )
            
            seller.lagrange[1] = np.clip(
                seller.lagrange[1],
                self.lambda_clip_min,
                self.lambda_clip_max
            )

        # --- Buyers ---
        for buyer in self.buyers:

            buyer.lagrange[0] += self.lambda_alpha * (penalty3 - self.cost_threshold)

            buyer.lagrange = np.clip(
                buyer.lagrange,
                self.lambda_clip_min,
                self.lambda_clip_max
            )



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

        dones = self.dones

        self.current_step += 1

        return obs, wellness, dones , {}
    
    def check_power_constrain_seller(self, t, P, seller):
        penalty= sum(P[seller.seller_id])-seller.net[t]
        return penalty

    def check_power_constrain_buyer(self, t, P, buyer):
        penalty = sum(P[:,buyer.buyer_id])-buyer.net[t]
        return penalty
    
    def check_cost_constrain(self, P, C):
        penalty = 0 
        for seller in self.sellers:
            Hg = seller.get_generation_costs(seller.state)
            cost = 0
            for i in range(len(C)):
                cost += C[i]*P[seller.seller_id, i]
            
            penalty += Hg - cost

        return penalty
    
    def check_community_constraints(self, t):
        total_demand = sum(b.net[t] for b in self.buyers)
        total_gen = sum(s.net[t] for s in self.sellers)
        total_power = sum(sum(s.state) for s in self.sellers)

        # print(total_demand, total_gen, total_power)

        # desired total = min(total_gen, total_demand)
        target = min(total_gen, total_demand)

        # Smooth penalty
        deviation = total_power - target
        # print(deviation)

        c = 0.1

        reward_signal = np.exp(-(deviation)**2/(2*c**2)) -1 # near 1 if very close, decays rapidly otherwise
        # print(reward_signal)
        return deviation

    def evaluate_cost_fairness(self):
        """
        Returns a continuous penalty (negative value) when generation cost exceeds total cost.
        0 means fair cost (Hg <= total_cost).
        """
        penalties = {}
        c = 0.1
        for seller in self.sellers:
            Hg = seller.get_generation_costs(seller.state)
            total_cost = 0.0
            for buyer in self.buyers:
                # buyer.state = price; seller.state[buyer.buyer_id] = power sold to buyer
                total_cost += buyer.state * seller.state[buyer.buyer_id]
            # print(f"{seller.group_name} total_cost: {total_cost}, Hg: {Hg}" )
            if Hg <= total_cost:
                penalty = np.exp(-(total_cost-Hg)**2/(2*c**2))
            else:
                penalty = 1

            penalties[seller.group_name] = penalty
        
        mean_penalties = sum(penalties.values())/len(penalties)

        return -mean_penalties
    
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
        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        P = np.zeros([num_sellers, num_buyers])
        C = np.zeros([num_buyers])


        for seller in self.sellers:
            P[seller.seller_id] = seller.state 

        for buyer in self.buyers:
            C[buyer.buyer_id] = buyer.state


        comm_penalty = self.check_community_constraints(t)
    
        for seller in self.sellers:
            penalty2 = 0
            if not self.dones[seller.group_name]:
                penalty1 = self.check_power_constrain_seller(t,P,seller) 
                for buyer in self.buyers:
                    penalty2 += self.check_power_constrain_buyer(t, P, buyer)
                l1 = seller.lagrange[0]
                l2 = seller.lagrange[1]
                l3 = seller.lagrange[2]
                state_info[f'{seller.name}_reward']= seller.get_wellness(t,P,C) - (l1*penalty1 + l2*penalty2 + l3*comm_penalty)
                state_info[f'{seller.name}_l1']= l1
                state_info[f'{seller.name}_l2']= l2
                state_info[f'{seller.name}_l3']= l3
        
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                penalty3 = self.check_cost_constrain(P,C)
                l4 = buyer.lagrange[0]
                state_info[f'{buyer.name}_reward'] =  buyer.get_wellness(t,P,C) - (l4*penalty3)
                state_info[f'{buyer.name}_l1'] = l4
        
        # for seller in self.sellers:
        #     others_power,others_price = self.get_others_power_price(seller, global_state)
        #     reward = seller.get_wellness(t,seller.state,buyer_prices,others_power,others_price)
        #     state_info[f'{seller.name}_reward'] = reward

        # for buyer in self.buyers:
        #     seller_power = self.get_sellers_power(t, buyer)
        #     others_selers_power = self.get_others_sellers_power(t, buyer)
        #     others_power,others_price = self.get_others_power_price(seller, global_state)
        #     reward = buyer.get_wellness(t,seller_power,buyer.state,others_selers_power,others_price)
        #     state_info[f'{buyer.name}_reward'] = reward


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




