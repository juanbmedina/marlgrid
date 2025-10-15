import numpy as np
from energy_env import P2PEnergyEnv
import matplotlib.pyplot as plt
from energy_agent import EnergyAgent

def test_environment(num_steps=5):
    # Initialize environment
    env = P2PEnergyEnv()
    obs = env.reset()
    actions_dict = {}

    # print("=== ACTION SPACES ===")
    # print(env.action_spaces)

    print("=== INITIAL OBSERVATION ===")
    print(f"Global Obs: {obs}")

    print("\n=== RUNNING TEST EPISODE ===")
    seller_state = 0.7
    buyer_state = 50 

    seller_state = [[0.51, 0.24], [1.48, 0.35], [0.53, 0.24], [0.35, 0.20]]
    buyer_state = [60.22, 50.04]

    for i, seller in enumerate(env.sellers):
        seller.state = np.array([seller_state[i][0], seller_state[i][1]])
        actions_dict[seller.group_name] = [0.0, 0.0]

    for i, buyer in enumerate(env.buyers):
        buyer.state = buyer_state[i]
        actions_dict[buyer.group_name] =  [0.0, 0.0]

    o, r, d , _ = env.step(actions_dict)

    print("=== FINAL OBSERVATION ===")
    print(f"Global Obs: {o}")
    print(f"Reward: {r}")
    print(f"Reward: {d}")

    # env.evaluate_constraints(0)


    print("\n=== TEST FINISHED ===")



def plot_agents_heatmaps(num_points=50, filename="agents_reward_heatmaps.png"):
    env = P2PEnergyEnv()
    env.reset()

    seller_states = np.linspace(0.0, 10.0, num_points)
    buyer_states = np.linspace(50.0, 100.0, num_points)

    agents = [agent.name for agent in env.agents]
    n_agents = len(agents)

    # First pass: collect all rewards
    all_rewards = {agent: [] for agent in agents}
    for buyer_state in buyer_states:
        for seller_state in seller_states:
            actions_dict = {}
            for seller in env.sellers:
                seller.state = np.array([0.8*seller_state, 0.2*seller_state])
                actions_dict[seller.name] = 4
            for buyer in env.buyers:
                buyer.state = buyer_state
                actions_dict[buyer.name] = 1
            _, rewards, _, _ = env.step(actions_dict)
            for agent in agents:
                all_rewards[agent].append(rewards[agent])

    # Normalize globally
    all_values = np.concatenate([np.array(vals) for vals in all_rewards.values()])
    r_min, r_max = all_values.min(), all_values.max()

    cols = int(np.ceil(np.sqrt(n_agents)))
    rows = int(np.ceil(n_agents / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, agent_name in enumerate(agents):
        rewards_matrix = np.zeros((num_points, num_points))
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                val = all_rewards[agent_name][k]
                rewards_matrix[i, j] = (val - r_min) / (r_max - r_min + 1e-9)
                k += 1

        im = axes[idx].imshow(
            rewards_matrix,
            extent=[seller_states.min(), seller_states.max(),
                    buyer_states.min(), buyer_states.max()],
            origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1
        )
        axes[idx].set_title(agent_name)
        axes[idx].set_xlabel("Seller state")
        axes[idx].set_ylabel("Buyer state")

    # Remove unused subplots
    for ax in axes[n_agents:]:
        fig.delaxes(ax)

    # Shared colorbar on the right side
    cbar = fig.colorbar(im, ax=axes[:n_agents], fraction=0.046, pad=0.08, location='right')
    cbar.set_label("Normalized Reward")

    fig.suptitle("Normalized reward heatmaps per agent", fontsize=16)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Subplot heatmaps (normalized) saved to {filename}")



def manual_state_test():
    env = P2PEnergyEnv()
    env.reset()
    t = 0   # timestep to test

    print("\n=== MANUAL STATE TEST ===")

    global_state = []
    j = 0
    # Example: set custom states manually
    states = np.array([[0.1, 0.1], [0.3, 0.4, 0], [0.5, 0.6, 0], [0.0, 0.0, 61],  [0.7, 0.8, 0],  [0.0, 0.0, 62]])
    for i, agent in enumerate(env.agents):
        if agent.rol[t] == "S":
            # Seller with some power distribution
            agent.state = states[i]
        elif agent.rol[t] == "B":
            # Buyer with a chosen price
            agent.buyer_id = j
            agent.state = states[i]
            j +=1

        # Build global state
        global_state.append(agent.state)

    # Compute rewards manually
    buyer_prices = env.get_buyers_price(t)
    for agent in env.agents:
        role = agent.rol[t]
        power = agent.state[:-1]
        price = agent.state[-1]
        others_power, others_price = env.get_others_power_price(agent, global_state)
        others_selers_power = None

        # print(f"{agent.name} Others power: {others_power}, Others price: {others_price}, Buyers prices: {buyer_prices}")

        if role == "S":
            # Seller needs others_price (array)
            reward = agent.get_wellness(t, power, buyer_prices, None, None)
        elif role == "B":
            seller_power = env.get_sellers_power(t, agent)
            others_selers_power = env.get_others_sellers_power(t, agent)
            # print("Other sellers power for buyer: ", others_selers_power)
            # Buyer needs its own scalar price
            reward = agent.get_wellness(t, seller_power, price, others_selers_power, others_price)
        else:
            done = True
            reward = 0.0
            
        
        print(f"{agent.name} ({role})")
        print(f"   Others power: {others_selers_power, others_power}, Others price: {others_price}, Buyers prices: {buyer_prices}")
        print(f"   State: {agent.state}")
        print(f"   Reward: {reward}")
        # print(f"   Done: {done}")

def test_constraints():
    
    env = P2PEnergyEnv()
    env.reset()
    t = 0   # timestep to test

    print("\n=== MANUAL STATE TEST ===")

    # Example: set custom states manually
    states = np.array([[1.1,48], [0.0,40], [0.0,200], [0.0, 40], [0.0,54], [0.0,200]])
    for i, agent in enumerate(env.agents):
        if agent.rol[t] == "S":
            # Seller with some power distribution
            agent.state = states[i]
        elif agent.rol[t] == "B":
            # Buyer with a chosen price
            agent.state = states[i]

    # Build global state
    global_state = []

    sellers, buyers = env.split_agents(t)

    num_sellers = len(sellers)
    num_buyers = len(buyers)

    print("\n=== BEFORE ENFORCE CONSTRAINTS ===")

    i = 0

    for agent in env.agents:
  
        if agent.rol[t] == 'B':
            agent.buyer_id = i
            i +=1

        delta = [0, 0]
        done, agent.state = agent.get_new_state(t, delta, num_sellers, num_buyers)
        print(f"{agent.name}, Role: {agent.rol[t]} Net: {agent.net[t]} State: {agent.state} Done: {done}")
        global_state.append(agent.state)


    print("\n=== AFTER ENFORCE CONSTRAINTS ===")

    env.enforce_constraints(t)
    env.community_constrains(t)
        
    for agent in env.agents:    
        print(f"{agent.name}, Role: {agent.rol[t]} Net: {agent.net[t]} State: {agent.state}")

def test_rewards():
    # Simple profiles for testing
    consumer_profile = [20, 20, 20]
    generator_profile = [50, 50, 50]
    cost_params = [0.01, 2, 10]   # quadratic cost params (a,b,c)

    # Create one seller and one buyer
    seller = EnergyAgent("agent_1", 2, consumer_profile, generator_profile, cost_params)
    buyer = EnergyAgent("agent_2", 2, consumer_profile, generator_profile, cost_params)

    t = 0

    # Sweep ranges
    power_values = np.linspace(0.1, 0.5, 50)     # seller's power
    price_values = np.linspace(1, 100, 50)      # price

    # Matrices for wellness
    seller_wellness = np.zeros((len(power_values), len(price_values)))
    buyer_wellness = np.zeros((len(power_values), len(price_values)))

    seller_w = seller.get_wellness(
                t,
                power=np.array([1,2]),
                price=np.array([1]),
                others_power=np.array([0]),
                others_price=np.array([1])
            )

    # for i, p in enumerate(power_values):
    #     for j, price in enumerate(price_values):
    #         # Seller case: power as vector, price as np.ndarray
    #         seller_w = seller.get_wellness(
    #             t,
    #             power=np.array([p]),
    #             price=np.array([price]),
    #             others_power=np.array([0]),
    #             others_price=np.array([1])
    #         )
    #         seller_wellness[i, j] = seller_w

    #         # Buyer case: power as list, price as float
    #         buyer_w = buyer.get_wellness(
    #             t,
    #             power=[p],
    #             price=float(price),
    #             others_power=[0],
    #             others_price=[1]
    #         )
    #         buyer_wellness[i, j] = buyer_w

    # # --- Plotting ---
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # im1 = axs[0].imshow(seller_wellness, extent=[price_values[0], price_values[-1], power_values[0], power_values[-1]],
    #                     origin='lower', aspect='auto')
    # axs[0].set_title("Seller Wellness")
    # axs[0].set_xlabel("Price")
    # axs[0].set_ylabel("Power")
    # fig.colorbar(im1, ax=axs[0])

    # im2 = axs[1].imshow(buyer_wellness, extent=[price_values[0], price_values[-1], power_values[0], power_values[-1]],
    #                     origin='lower', aspect='auto')
    # axs[1].set_title("Buyer Wellness")
    # axs[1].set_xlabel("Price")
    # axs[1].set_ylabel("Power")
    # fig.colorbar(im2, ax=axs[1])

    # plt.tight_layout()
    # plt.savefig("wellness_sweep.png")
    # print("✅ Plot saved as wellness_sweep.png")




if __name__ == "__main__":
    test_environment(num_steps=3)
    # plot_agents_heatmaps(num_points=50, filename="all_agents_heatmaps.png")




    # test_rewards()
    # manual_state_test()
    # test_constraints()