import numpy as np
from energy_env import P2PEnergyEnv

def test_environment(num_steps=5):
    # Initialize environment
    env = P2PEnergyEnv()
    obs = env.reset()

    print("=== INITIAL OBSERVATION ===")
    for agent, o in obs.items():
        print(f"{agent}: {o['obs']}")

    print("\n=== RUNNING TEST EPISODE ===")
    for step in range(num_steps):
        print(f"\n--- Step {step} ---")

        actions = {}
        deltas = {}

        # Sample random actions for each agent
        for agent in env.agents:
            action_id = env.action_spaces[agent.name].sample()
            delta = env._action_to_delta(action_id)
            actions[agent.name] = action_id
            deltas[agent.name] = delta

        # Print actions and deltas
        for agent in env.agents:
            print(f"{agent.name} -> action_id={actions[agent.name]}, delta={deltas[agent.name]}")

        # Step environment
        obs, rewards, dones, infos = env.step(actions)

        # Print states, rewards, and done flags
        for agent in env.agents:
            name = agent.name
            print(f"{name}, {agent.rol[0]}:")
            print(f"   State: {agent.state}")
            print(f"   Reward: {rewards[name]:.4f}")
            print(f"   Done: {dones[name]}")

        if dones["__all__"]:
            print("\nEnvironment signaled termination.")
            break

    print("\n=== TEST FINISHED ===")

def manual_state_test():
    env = P2PEnergyEnv()
    env.reset()
    t = 0   # timestep to test

    print("\n=== MANUAL STATE TEST ===")

    # Example: set custom states manually
    states = np.array([[0.1,0.2,50], [0.3,0.4,51], [0.5,0.6,52], [0.7,0.8,53], [0.9,1.0,54], [1.1,1.2,55]])
    for i, agent in enumerate(env.agents):
        if agent.rol[t] == "S":
            # Seller with some power distribution
            agent.state = states[i]
        elif agent.rol[t] == "B":
            # Buyer with a chosen price
            agent.state = states[i]

    # Build global state
    global_state = [a.state for a in env.agents]

    # Compute rewards manually
    buyer_prices = env.get_buyers_price(t)
    for agent in env.agents:
        role = agent.rol[t]
        power = agent.state[:-1]
        price = agent.state[-1]
        others_power, others_price = env.get_others_power_price(agent, global_state)

        # print(f"{agent.name} Others power: {others_power}, Others price: {others_price}, Buyers prices: {buyer_prices}")

        if role == "S":
            # Seller needs others_price (array)
            reward = agent.get_wellness(t, power, others_price, others_power, others_price)
        elif role == "B":
            # Buyer needs its own scalar price
            reward = agent.get_wellness(t, power, price, others_power, others_price)
        else:
            reward = 0.0

        print(f"{agent.name} ({role})")
        print(f"   State: {agent.state}")
        print(f"   Reward: {reward:.4f}")

def test_constraints():
    
    env = P2PEnergyEnv()
    env.reset()
    t = 0   # timestep to test

    print("\n=== MANUAL STATE TEST ===")

    # Example: set custom states manually
    states = np.array([[0.5,0.2,50], [0.0,0.0,51], [0.0,0.0,52], [0.1,0.0, 53], [0.0,0.0,54], [0.1,0.0,55]])
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

        delta = [0, 0, 0]
        agent.state = agent.get_new_state(t, delta, num_sellers, num_buyers)
        print(f"{agent.name}, Role: {agent.rol[t]} Net: {agent.net[t]} State: {agent.state}")
        global_state.append(agent.state)


    print("\n=== AFTER ENFORCE CONSTRAINTS ===")

    env.enforce_constraints(t)
    env.community_constrains(t)
        
    for agent in env.agents:    
        print(f"{agent.name}, Role: {agent.rol[t]} Net: {agent.net[t]} State: {agent.state}")





if __name__ == "__main__":
    test_environment(num_steps=3)
    # manual_state_test()
    # test_constraints()