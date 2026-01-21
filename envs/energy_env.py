from ray.rllib.env.multi_agent_env import MultiAgentEnv
from envs.energy_agent import EnergyAgent
from gymnasium.spaces import Discrete, Box
from gymnasium.spaces import Dict as GymDict

import gymnasium as gym


import json
import numpy as np
import csv
import os
from datetime import datetime



class P2PEnergyEnv(MultiAgentEnv):

    def __init__(self, config=None):
        super().__init__()

        ######## Agent creation ########
        json_path = 'profiles/agents_profiles.json'
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Required file not found: {json_path}")
        self.create_agents(json_path)
        self.sellers, self.buyers = self.split_agents(t=0)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        ######## Agent name ########
        seller_id = 0
        for seller in self.sellers:
            seller.group_name = 'seller_' + str(seller_id)
            seller_id += 1
        
        buyer_id = 0 
        for buyer in self.buyers:
            buyer.group_name = 'buyer_' + str(buyer_id)
            buyer_id += 1

        self.possible_agents = []
        for agent in self.env_agents:
            self.possible_agents.append(agent.group_name)
        
        self.agents = self.possible_agents

        ######## Init obs and action space ########
        self.observation_spaces = {}
        self.action_spaces = {}

        # Action mappings
        self.power_step = 0.1   # small step for power allocations
        self.price_step = 1.0    # larger step for price


        for seller in self.sellers:
            self.observation_spaces[seller.group_name] = gym.spaces.Box(low=-100,
                                                                        high=1000,
                                                                        shape=(num_sellers*num_buyers+num_buyers + 2, ), 
                                                                        dtype=np.float32)
            
            self.action_spaces[seller.group_name] = gym.spaces.Box(low=-self.power_step,
                                                                    high=self.power_step,
                                                                    shape=(2,),
                                                                    dtype=np.float32
                                                                )
        for buyer in self.buyers:
            self.observation_spaces[buyer.group_name] = gym.spaces.Box(low=-100,
                                                                        high=1000,
                                                                        shape=(num_sellers*num_buyers+num_buyers + 2, ), 
                                                                        dtype=np.float32)
            
            self.action_spaces[buyer.group_name] = gym.spaces.Box(low=-self.price_step,
                                                                    high=self.price_step,
                                                                    dtype=np.float32
                                                                )

        self.t = 0
        self.max_steps = 100
        self.current_step = 0
        self.iter_count = 0

        # === Dual ascent parameters ===
        self.lambda_alpha = 0.1          # learning rate para actualización dual
        self.lambda_clip_min = 0.0
        self.lambda_clip_max = 100.0     # límite superior para evitar explosiones
        self.cost_threshold = 0.0         # d en J_C - d <= 0
        self.cost_ma_alpha = 0.05         # suavizado EMA para la violación

        self.lagrange = np.array([0.0, 0.0, 0.0, 0.0]) 
        self.episode_id = 0


        # Save printed states
        self.csv_initialized = False
        self.csv_file = f"market_log_all_episodes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    def reset(self, *, seed=None, options=None):
        self.episode_id += 1
        infos = {}
        self.sellers, self.buyers = self.split_agents(t=0)
        self.current_step = 0
        global_state = []
        obs = {}
        self.previous_wellness = {}
        self.iter_count += 1


        for seller in self.sellers:
            s1 = 0.475
            s2 = 0.475
            seller.state = np.array([s1, s2]) 
            seller.cost_ma = 0.0                    # NEW
            global_state = np.concatenate((global_state, seller.state), axis = 0)

        for buyer in self.buyers:
            buyer.state =  50
            buyer.cost_ma = 0.0  
            global_state = np.append(global_state, buyer.state)
        
        global_state = np.concatenate((global_state, np.array([0, 0])), axis=0)

        for seller in self.sellers:
            obs[seller.group_name] = np.array(global_state, dtype=np.float32).flatten()
        for buyer in self.buyers:
            obs[buyer.group_name] = np.array(global_state, dtype=np.float32).flatten()

        for agent in self.env_agents:
            infos[agent.group_name] =   {}

        return obs, infos

    def step(self, action_dict):
        """Execute one step in the environment"""
        t = 0
        global_state = []
        obs = {}
        infos = {}
        penalty1 = {}
        penalty2 = {}
        penalty3 = {}
        wellness = {}
        terminateds = {}
        truncateds = {}

        terminateds = {agent.group_name: False for agent in self.env_agents}
        truncateds = {agent.group_name: False for agent in self.env_agents}

        self.sellers, self.buyers = self.split_agents(t)

        num_sellers = len(self.sellers)
        num_buyers = len(self.buyers)

        P = np.zeros([num_sellers, num_buyers])
        C = np.zeros([num_buyers])

        ################### TAKE ACTIONS ###################
        for seller in self.sellers:
            if not terminateds[seller.group_name]:
                delta = action_dict[seller.group_name]
                done, seller.state = seller.get_new_state(t, delta)
            P[seller.seller_id] = seller.state 

        for buyer in self.buyers:
            if not terminateds[buyer.group_name]:
                delta = action_dict[buyer.group_name]
                done, buyer.state = buyer.get_new_state(t, delta)
            C[buyer.buyer_id] = buyer.state

        P_flat = P.flatten()  # This gives num_sellers * num_buyers values
        global_state = np.concatenate([P_flat, C])
        
        penalty4 = self.check_community_constraints(t)

        global_state = np.concatenate((global_state, np.array([self.lagrange[3], penalty4])), axis=0)
        

        ################### REWARD AND OBSERVATIONS ###################
        for seller in self.sellers:
            if not terminateds[seller.group_name]:
                obs[seller.group_name] = np.array(global_state, dtype=np.float64).flatten()

                penalty1[seller.group_name] = self.check_power_constrain_seller(t,P,seller)  
                penalty2[seller.group_name] = self.check_cost_constrain(P,C, seller)   
                    
                wellness[seller.group_name] = seller.get_wellness(t,P,C)

                self.lagrange[0] += self.lambda_alpha * (penalty1[seller.group_name])
                self.lagrange[1] += self.lambda_alpha * (penalty2[seller.group_name])
                
        
        for buyer in self.buyers:
            if not terminateds[buyer.group_name]:
                obs[buyer.group_name] = np.array(global_state, dtype=np.float64).flatten()

                penalty3[buyer.group_name] = self.check_power_constrain_buyer(t, P, buyer)

                wellness[buyer.group_name] = buyer.get_wellness(t,P,C) 

                self.lagrange[2] += self.lambda_alpha * (penalty3[buyer.group_name])

        for seller in self.sellers:
            l1 = self.lagrange[0]
            l2 = self.lagrange[1]
            mu1 = self.lagrange[3]
            infos[seller.group_name] = {
                "mu1": mu1,
                "penalty4": penalty4
            }
            # wellness[seller.group_name] += - (l1*penalty1[seller.group_name]  + l2*penalty2[seller.group_name]) + mu1*penalty4
            wellness[seller.group_name] +=  -mu1*penalty4

        for buyer in self.buyers:
            l3 = self.lagrange[2]
            # mu1 = self.lagrange[3]
            # wellness[buyer.group_name] += - (l3*penalty3[buyer.group_name])  + mu1*penalty4
            # wellness[buyer.group_name] += mu1*penalty4


        self.lagrange[0] = np.clip(
                self.lagrange[0],
                self.lambda_clip_min,
                self.lambda_clip_max
            )
            
        self.lagrange[1] = np.clip(
                self.lagrange[1],
                self.lambda_clip_min,
                self.lambda_clip_max
            )
        
        self.lagrange[2] = np.clip(
                self.lagrange[2],
                self.lambda_clip_min,
                self.lambda_clip_max
            )
        
        self.lagrange[3] = np.clip(
                self.lagrange[3],
               -self.lambda_clip_max,
                self.lambda_clip_max
            )



        ################### ACTIVE AGENTS ###################
        for agent in self.env_agents:
            terminateds[agent.group_name] = False
            truncateds[agent.group_name] = False

        ################### DONE EPISODE ###################
        if self.current_step >= self.max_steps:
            self.lagrange[3] += self.lambda_alpha*penalty4
            # with open(self.log_path, "a", newline="") as f:
            #     writer = csv.writer(f)
            #     writer.writerow([
            #         self.episode_id,
            #         mu1,
            #         penalty4
            #     ])
            # print("mu: ", mu1, "Penalty: ", penalty4, "to wellness: ", -mu1*penalty4)
            for agent in self.env_agents:
                terminateds[agent.group_name] = True
        # Global done
        terminateds["__all__"] = all(terminateds[a.group_name] for a in self.env_agents)
        truncateds["__all__"] = False


        self.current_step += 1

        return obs, wellness, terminateds, truncateds, infos

    def check_power_constrain_seller(self, t, P, seller):
        penalty= sum(P[seller.seller_id])-seller.net[t]
        return penalty

    def check_power_constrain_buyer(self, t, P, buyer):
        penalty = sum(P[:,buyer.buyer_id])-buyer.net[t]
        return penalty

    def check_cost_constrain(self, P, C, seller):

        Hg = seller.get_generation_costs(seller.state)
        cost = 0
        for i in range(len(C)):
            cost += C[i]*P[seller.seller_id, i]
        penalty = Hg - cost

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


    def split_agents(self, t=0):
        """
        Count how many sellers and buyers exist at time t.
        Returns (num_sellers, num_buyers).
        """
        sellers = []
        buyers = []
        i = 0
        j = 0
        for agent in self.env_agents:
            if agent.rol[t] == "S":
                agent.seller_id = i
                sellers.append(agent)
                i += 1
            elif agent.rol[t] == "B":
                agent.buyer_id = j
                buyers.append(agent)
                j += 1

        return sellers, buyers
    
    def create_agents(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.n_agents = len(data.items()) # number of agents
        self.env_agents = []
        for name, profiles in data.items():
            agent = EnergyAgent(
                name=name,
                n_agents=self.n_agents,
                consumer_profile=profiles["consumer_profile"],
                generator_profile=profiles["generator_profile"],
                cost_params=profiles["cost_params"]
            )
            self.env_agents.append(agent)
