import numpy as np

class EnergyAgent:
    def __init__(self, name, n_agents, consumer_profile, generator_profile, cost_params):
        self.name = name
        self.agent_con = np.array(consumer_profile)
        self.agent_gen = np.array(generator_profile)

        self.id = int(name.split("_")[1])

        self.T = len(self.agent_con)

        self.n_agents = n_agents

        self.state = None
        self.lagrange = None

        self.lambda_factor =  10  
        self.theta_factor = 1/2
        self.beta_factor = 1

        self.group_name = None

        self.buyer_id = None
        self.seller_id = None

        self.a = cost_params[0]
        self.b = cost_params[1]
        self.c = cost_params[2]

        self.grid_sell_price = 100
        self.grid_buy_price = 50

        self.ind_power_min = 0.1
        self.ind_power_max = 2   # or based on generator capacity
        # self.price_min = 0.01
        # self.price_max = 1000.0


        self.calculate_agent_rol()
        self.calculate_net_profile() 

    def calculate_agent_rol(self):
        self.rol = []
        for t in range(self.T):
            gdr = self.agent_gen[t] / self.agent_con[t]
            self.rol.append(np.where(gdr > 1, 'S', np.where(gdr < 1, 'B', 'N')).tolist())

    def get_new_state(self, t, action):
        """
        Construct agent state dynamically based on role at time t.
        powers: list of all agents' powers
        prices: list of all agents' prices
        """
        if self.rol[t] == 'S':
            # Seller: state = [P_i,j ... , price_i] with mask [1,...,1,0]
            self.state += action
            done = self.agent_in_range(t, 10e6)

            # Apply boundaries to power states
            self.state = np.clip(self.state, self.ind_power_min, self.ind_power_max)

            return done, self.state
        
        elif self.rol[t] == 'B':
            # Buyer: state = [P_i,j ... , price_i] with mask [0,...,0,1]
            self.state += action

            # Apply boundaries to price state
            self.state = np.clip(self.state, self.grid_buy_price, self.grid_sell_price)

            done = self.agent_in_range(t, 10e6)

            return done, self.state


    def calculate_net_profile(self):
        self.net = []
        for t in range(self.T):
            if self.rol[t] == 'S':
                net = self.agent_gen[t] - self.agent_con[t]
            elif self.rol[t] == 'B':
                net = self.agent_con[t] - self.agent_gen[t]
            else:
                net = 0.0
            self.net.append(net)
        self.net = np.array(self.net)
    
    def get_wellness(self, t, P, C): 
        
        if self.rol[t] == 'S':
            reward = self.get_seller_reward(P[self.seller_id], C)
            Hg = self.get_generation_costs(P[self.seller_id])
            utility = self.calculate_utility(t)
            wellness = utility + reward - Hg
            return wellness

        elif self.rol[t] == 'B':
            reward = self.get_buyer_reward(P[:,self.buyer_id], C[self.buyer_id])
            utility = self.calculate_utility(t)
            P_others = np.delete(P, self.buyer_id, axis=1)
            C_others = np.delete(C, self.buyer_id)
            comp_resource = self.calculate_comp_resource(P_others, C_others)
            wellness = utility + reward - comp_resource
            return float(wellness)
        
    def agent_in_range(self, t, penalty):
        if self.rol[t] == 'S':
            # Seller: state = [P_i,j ... , price_i] with mask [1,...,1,0]

            in_range = (self.ind_power_min < self.state) & (self.state < self.ind_power_max)

            if np.all(in_range):
                return False
            else: 
                return True
        
        elif self.rol[t] == 'B':
            # Buyer: state = [P_i,j ... , price_i] with mask [0,...,0,1]
            in_range = (self.grid_buy_price < self.state) & (self.state < self.grid_sell_price)

            if np.all(in_range):
                return False
            else: 
                return True
            
    def calculate_utility(self, t):
        utility = self.lambda_factor*self.agent_con[t] - (self.theta_factor/2)*self.agent_con[t]**2
        return utility
    
    def get_seller_reward(self, power, price):
        if isinstance(price, np.ndarray):
            reward = 0
            for i in range(len(power)):
                reward += power[i]/np.log(1+1/1000*price[i]+0.001)
            return -reward 
        else:
            raise TypeError("The seller price must be a np.ndarray")
    
    def get_generation_costs(self, power):
        Hg = self.a*(sum(power))**2 + self.b*sum(power) + self.c
        return Hg
    
    def get_buyer_reward(self, power, price):
        if isinstance(price, float):   # más pythonic que type(price)==float
            reward = sum(power) / np.log(price + 1 +0.001)
            return reward
        else:
            raise TypeError("The seller price must be a float number")
    
    def calculate_comp_resource(self, others_power, others_price):
        cr = self.beta_factor*sum(others_price)*sum(others_power)
        return cr


    def __repr__(self):
        return (f"<Agent {self.name}:\n"
                f"  Consumer={self.agent_con},\n"
                f"  Generator={self.agent_gen},\n"
                # f"  Role={self.rol},\n"
                # f"  Net={self.net_profile},\n"
                # f"  Utility={self.utility}>\n"
                )
