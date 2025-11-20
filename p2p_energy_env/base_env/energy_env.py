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

        # NOTE: current code used Box(shape=(2,)) for each agent - keep that mapping
        # but the code below will handle vector/array actions by broadcasting to seller.state shape.
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
                                       
                                       
        
        self.t = 0
        self.max_steps = 100
        self.current_step = 0
        self.iter_count = 0

        # Save printed states
        self.csv_initialized = False
        self.csv_file = f"market_log_all_episodes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"


    # ----------------------------
    # Projection and initialization helpers
    # ----------------------------
    def initial_project_to_feasible(self, seller_caps, buyer_dems):
        """
        Project Pprime to feasible set using proportional scaling.
        The projection implemented here returns the proportional allocation:
            P_feasible = T * outer(alpha, beta)
        where alpha is normalized seller capacities and beta is normalized buyer demands,
        and T = min(total_gen, total_dem).

        This ensures:
            - each seller's row sum <= seller_caps[i]
            - each buyer's col sum <= buyer_dems[j]
            - total sum = T
        """
        # Ensure numpy arrays
        seller_caps = np.array(seller_caps, dtype=float)
        buyer_dems = np.array(buyer_dems, dtype=float)

        total_gen = seller_caps.sum()
        total_dem = buyer_dems.sum()
        T = min(total_gen, total_dem)

        # Avoid division by zero
        eps = 1e-9
        seller_caps_safe = np.maximum(seller_caps, eps)
        buyer_dems_safe = np.maximum(buyer_dems, eps)

        #projection algorithm
        alpha = seller_caps_safe / seller_caps_safe.sum()
        beta = buyer_dems_safe / buyer_dems_safe.sum()

        P_feasible = T * np.outer(alpha, beta)

        return P_feasible
    
    def project_to_feasible(self, Pprime, seller_caps, buyer_dems):
        """
        Project Pprime to feasible region by minimal proportional scaling.

        Steps:
            1. Fix seller (row) violations
            2. Fix buyer (column) violations
            3. Fix global balance to T = min(total_gen, total_dem)
        """

        P = np.array(Pprime, dtype=float)   # start from the agent's tentative matrix

        # --- 1. Enforce seller capacity constraints ---
        row_sums = P.sum(axis=1)
        for i in range(P.shape[0]):
            if row_sums[i] > seller_caps[i]:
                if row_sums[i] > 0:
                    P[i, :] *= seller_caps[i] / row_sums[i]

        # --- 2. Enforce buyer demand constraints ---
        col_sums = P.sum(axis=0)
        for j in range(P.shape[1]):
            if col_sums[j] > buyer_dems[j]:
                if col_sums[j] > 0:
                    P[:, j] *= buyer_dems[j] / col_sums[j]

        # --- 3. Enforce global market constraint ---
        total_gen = seller_caps.sum()
        total_dem = buyer_dems.sum()
        T = min(total_gen, total_dem)

        total_power = P.sum()
        if total_power > T:
            P *= T / (total_power + 1e-9)

        return P


    def initialize_feasible_states(self, t):
        """
        Construct a feasible initial P matrix and assign to seller.state and buyer.state.
        Buyers' state are prices (kept randomly in a reasonable interval).
        """
        seller_caps = np.array([s.net[t] for s in self.sellers], dtype=float)
        buyer_dems = np.array([b.net[t] for b in self.buyers], dtype=float)

        # Guard in case there are zero sellers or buyers
        if len(seller_caps) == 0:
            return np.zeros((0, len(buyer_dems)))
        if len(buyer_dems) == 0:
            return np.zeros((len(seller_caps), 0))

        P = self.initial_project_to_feasible(seller_caps, buyer_dems)

        # assign seller states (row per seller)
        for i, seller in enumerate(self.sellers):
            seller.state = P[i].copy()

        # buyer states -> price (random initial price)
        for j, buyer in enumerate(self.buyers):
            buyer.state = random.uniform(40, 100)

        return P


    # ----------------------------
    # Reset
    # ----------------------------
    def reset(self):
        self.sellers, self.buyers = self.split_agents(t=0)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)
        self.current_step = 0
        obs = {}
        self.previous_wellness = {}

        self.dones = {a.group_name: False for a in self.agents}

        self.iter_count += 1

        t = 0

        # Initialize states in a feasible way
        global_state = []
        P = self.initialize_feasible_states(t)

        # Build global_state from assigned seller and buyer states
        global_state = np.array([], dtype=float)
        for seller in self.sellers:
            global_state = np.concatenate((global_state, seller.state), axis=0)
        for buyer in self.buyers:
            global_state = np.append(global_state, buyer.state)

        # Logging prints to help debugging (optional)
        print("################## Reset ##################")
        for seller in self.sellers:
            print(f"{seller.name}: Total power to sell: {sum(seller.state)} max generation: {seller.net[t]}")
           
        for buyer in self.buyers:
            seller_power = self.get_sellers_power(t, buyer)
            print(f"{buyer.name}: Total power bought (sample from sellers): {sum(seller_power)} demand: {buyer.net[t]}")

        # Populate observations
        for seller in self.sellers:
            obs[seller.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
        for buyer in self.buyers:
            obs[buyer.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}

        return obs


    # ----------------------------
    # Step
    # ----------------------------
    def step(self, action_dict):
        """Execute one step in the environment"""
        t = 0
        global_state = []
        obs = {}
        rewards = {}
        self.current_agents = []
        agent_out = {a.group_name: False for a in self.agents}
        wellness = {}

        # update sellers/buyers role distribution (if roles vary over time)
        self.sellers, self.buyers = self.split_agents(t)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        buyer_prices = self.get_buyers_price(t)

        ################### PROVISIONAL APPLY ACTIONS (build Pprime) ###################
        # We'll take seller.state (current) + delta (action) as tentative new values,
        # clip them to per-dimension bounds, then project whole matrix to feasible set.
        Pprime_rows = []
        # Build tentative seller rows
        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                delta = action_dict.get(seller.group_name, None)
                if delta is None:
                    # No action provided: assume zero delta
                    delta_arr = np.zeros_like(seller.state)
                else:
                    # Convert delta into an array that matches seller.state shape
                    delta_arr = np.array(delta, dtype=float)
                    # If shapes don't match, try to broadcast or tile
                    if delta_arr.size == 1:
                        delta_arr = np.full_like(seller.state, delta_arr.item())
                    elif delta_arr.size != seller.state.size:
                        # try broadcast along first elements (best-effort)
                        try:
                            delta_arr = np.resize(delta_arr, seller.state.shape)
                        except Exception:
                            delta_arr = np.zeros_like(seller.state)
                # Tentative new state
                tentative = seller.state + delta_arr
                # Clip by per-dimension min/max
                tentative = np.clip(tentative, seller.ind_power_min, seller.ind_power_max)
                Pprime_rows.append(tentative)
            else:
                Pprime_rows.append(seller.state.copy())

        if len(Pprime_rows) > 0:
            Pprime = np.array(Pprime_rows, dtype=float)
        else:
            Pprime = np.zeros((0, num_buyers), dtype=float)

        # If there are buyers but no sellers or viceversa handle gracefully
        seller_caps = np.array([s.net[t] for s in self.sellers], dtype=float) if len(self.sellers) > 0 else np.array([], dtype=float)
        buyer_dems = np.array([b.net[t] for b in self.buyers], dtype=float) if len(self.buyers) > 0 else np.array([], dtype=float)

        # Project the tentative matrix to feasible region
        if Pprime.size == 0:
            P = Pprime.copy()
        else:
            P = self.project_to_feasible(Pprime, seller_caps, buyer_dems)

        # Assign projected states back to sellers and build global_state
        global_state = np.array([], dtype=float)
        for i, seller in enumerate(self.sellers):
            seller.state = P[i].copy()
            global_state = np.concatenate((global_state, seller.state), axis=0)

        # Buyers: apply their actions (price changes) and clip
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                delta = action_dict[buyer.group_name]
                done, buyer.state = buyer.get_new_state(t, delta[0]*10, num_sellers, num_buyers)
                # self.dones[buyer.group_name] = done
                # agent_out[buyer.group_name] = done
            global_state = np.append(global_state, buyer.state)

        ################### REWARD AND OBSERVATIONS ###################
        # Build observations and compute wellness/rewards using the now-feasible state P
        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                obs[seller.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
                others_power,others_price = self.get_others_power_price(seller, global_state)
                wellness[seller.group_name] = seller.get_wellness(t,seller.state,buyer_prices,others_power,others_price)
        
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                obs[buyer.group_name] = {"obs": np.array(global_state, dtype=np.float64).flatten()}
                seller_power = self.get_sellers_power(t, buyer)
                others_selers_power = self.get_others_sellers_power(t, buyer)
                others_power,others_price = self.get_others_power_price(seller, global_state)
                wellness[buyer.group_name] = buyer.get_wellness(t,seller_power, float(buyer.state),others_selers_power,others_price)


        constraint_reward = self.evaluate_constraints(t)
        cost_fairness_penalty = self.evaluate_cost_fairness()

        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                wellness[seller.group_name] += 1000 * constraint_reward 
        
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                wellness[buyer.group_name] += 1000 * cost_fairness_penalty 

        if self.current_step ==  0:
            self.previous_wellness = wellness

        for seller in self.sellers:
            if not self.dones[seller.group_name]:
                rewards[seller.group_name] = (wellness[seller.group_name]**2 - self.previous_wellness[seller.group_name]**2)/100
        
        for buyer in self.buyers:
            if not self.dones[buyer.group_name]:
                rewards[buyer.group_name] = (wellness[buyer.group_name]**2 - self.previous_wellness[buyer.group_name]**2)/100


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

        # Logging prints to help debugging (optional)
        print("################## Step ##################")
        for seller in self.sellers:
            print(f"{seller.name}: Total power to sell: {sum(seller.state)} max generation: {seller.net[t]}")
           
        for buyer in self.buyers:
            seller_power = self.get_sellers_power(t, buyer)
            print(f"{buyer.name}: Total power bought (sample from sellers): {sum(seller_power)} demand: {buyer.net[t]}")


        # Note: the original step returned (obs, wellness, dones, {}) - keeping same contract
        # rewards dict is computed but not returned for RLlib wrapper compat (wrapper uses wellness)
        return obs, wellness, dones , {}
    
    def evaluate_constraints(self, t):
        total_demand = sum(b.net[t] for b in self.buyers)
        total_gen = sum(s.net[t] for s in self.sellers)
        total_power = sum(sum(s.state) for s in self.sellers)

        # desired total = min(total_gen, total_demand)
        target = min(total_gen, total_demand)

        # Smooth penalty
        deviation = abs(total_power - target)

        reward_signal = np.exp(-10*deviation)  # near 1 if very close, decays rapidly otherwise
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
            if Hg <= total_cost:
                penalty = np.exp(-10*(total_cost-Hg))
            else:
                penalty = 1

            penalties[seller.group_name] = penalty
        
        mean_penalties = sum(penalties.values())/len(penalties) if len(penalties)>0 else 0.0

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
            # remove the element corresponding to buyer.buyer_id
            if buyer is not None:
                del seller_power[buyer.buyer_id]
            other_seller_power = np.concatenate((other_seller_power,np.array(seller_power)), axis=0)
        
        return other_seller_power

    def get_others_power_price(self, agent, global_state):
        others_power = []
        others_price = []
        for seller in self.sellers:
            if agent is None or seller.name != agent.name:
                others_power = np.concatenate((others_power, seller.state), axis=0)
        for buyer in self.buyers:
            # buyers don't have group_name collision with sellers, so the if keeps consistent behavior
            if agent is None or buyer.name != getattr(agent, "name", None):
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
            others_power,others_price = self.get_others_power_price(self.sellers[0] if len(self.sellers)>0 else None, global_state)
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
        Returns (sellers_list, buyers_list).
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
                # j increment corrected (was j =+ 1 in original)
                j += 1

        return sellers, buyers
