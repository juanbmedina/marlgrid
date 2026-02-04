import numpy as np
from envs.energy_env import P2PEnergyEnv
import matplotlib.pyplot as plt
from envs.energy_agent import EnergyAgent

def test_environment(num_steps=5):
    # Initialize environment
    env = P2PEnergyEnv()
    obs, infos = env.reset()
    actions_dict = {}

    print("\n=== RETURN RESET ===")
    print("--- obs ---")
    print(obs)

    seller_state = [[0.5, 0.5], [0.5, 0.5], [0.53, 0.24], [0.35, 0.20]]
    buyer_state = [60.22, 50.04]

    for i, seller in enumerate(env.sellers):
        # seller.state = np.array([0.4875*2, 0.4875*2])
        # print("seller net: ", seller.net[0])
        actions_dict[seller.group_name] = [0.1, 0.2]

    for i, buyer in enumerate(env.buyers):
        # buyer.state = 50
        # print("buyer buy: ", buyer.net[0])
        actions_dict[buyer.group_name] =  2.0

    o, r, d , _, _ = env.step(actions_dict)

    print("=== FINAL OBSERVATION ===")
    print(f"Global Obs: {o}")
    print(f"Reward: {r}")

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





if __name__ == "__main__":
    test_environment(num_steps=3)
    # plot_agents_heatmaps(num_points=50, filename="all_agents_heatmaps.png")