import numpy as np

class EnergyAgent:
    def __init__(self, name, n_agents, consumer_profile, generator_profile):
        self.name = name
        self.agent_con = np.array(consumer_profile)
        self.agent_gen = np.array(generator_profile)

        self.id = int(name.split("_")[1])

        self.T = len(self.agent_con)

        self.n_agents = n_agents

        self.state = np.array([8.1, 2.2, 10])

        self.lambda_factor =  0.1
        self.theta_factor = 0.1
        self.beta_factor = 0.1

        self.a = 0.1
        self.b = 0.1
        self.c = 0 

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
            mask = np.append(np.ones(self.n_agents-1),0)
            self.state += mask*action
            return self.state
        
        elif self.rol[t] == 'B':
            # Buyer: state = [P_i,j ... , price_i] with mask [0,...,0,1]
            mask = np.append(np.zeros(self.n_agents-1),1)
            self.state += mask*action
            return self.state


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
    
    def get_wellness(self, t, power, price, others_power, others_price): 
        
        if self.rol[t] == 'S':
            reward = self.get_seller_reward(power, price)
            Hg = self.get_generation_costs(power)
            utility = self.calculate_utility(t)
            wellness = utility + reward - Hg
            return wellness 

        elif self.rol[t] == 'B':
            reward = self.get_buyer_reward(power, price)
            utility = self.calculate_utility(t)
            comp_resource = self.calculate_comp_resource(others_power, others_price)
            wellness = utility + reward - comp_resource
            return wellness 
        

    def calculate_utility(self, t):
        utility = self.lambda_factor*self.net[t] - (self.theta_factor/2)*self.net[t]**2
        return utility
    
    def get_seller_reward(self, power, price):
        if isinstance(price, np.ndarray):
            reward = 0
            for i in range(len(power)):
                reward += power[i]/np.log(1+price[i]+0.001)
            return -reward 
        else:
            raise TypeError("The seller price must be a np.ndarray")
    
    def get_generation_costs(self, power):
        Hg = self.a*(sum(power))**2 + self.b*sum(power) + self.c
        return Hg
    
    def get_buyer_reward(self, power, price):
        if isinstance(price, float):   # más pythonic que type(price)==float
            reward = sum(power) / np.log(price + 1)
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
