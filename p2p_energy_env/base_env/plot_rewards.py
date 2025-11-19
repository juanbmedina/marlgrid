import numpy as np
import matplotlib.pyplot as plt
from energy_env import P2PEnergyEnv


def plot_agents_heatmaps(num_points=50, filename="agents_reward_heatmaps.png"):
    """Normalized reward heatmaps for all agents (scaled to [0,1])."""
    env = P2PEnergyEnv()
    env.reset()

    seller_states = np.linspace(0.0, 2.0, num_points)
    buyer_states = np.linspace(50.0, 100.0, num_points)

    agents = [agent.group_name for agent in env.agents]
    n_agents = len(agents)
    all_rewards = {agent: [] for agent in agents}

    print(f"🌀 Sweeping reward space with {num_points**2} samples ...")

    for buyer_state in buyer_states:
        for seller_state in seller_states:
            actions_dict = {}
            env.dones = {a.group_name: False for a in env.agents}
            env.current_agents = env.agents

            for seller in env.sellers:
                seller.state = np.array([seller_state / 2, seller_state / 2])
                actions_dict[seller.group_name] = np.zeros_like(seller.state)
            for buyer in env.buyers:
                buyer.state = buyer_state
                actions_dict[buyer.group_name] = np.zeros((1,))

            _, rewards, _, _ = env.step(actions_dict)
            for agent in agents:
                all_rewards[agent].append(rewards.get(agent, 0.0))

    # Global normalization
    all_values = np.concatenate([np.array(vals) for vals in all_rewards.values()])
    r_min, r_max = all_values.min(), all_values.max()

    cols = int(np.ceil(np.sqrt(n_agents)))
    rows = int(np.ceil(n_agents / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, agent_name in enumerate(agents):
        rewards_matrix = np.zeros((num_points, num_points))
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                val = all_rewards[agent_name][k]
                rewards_matrix[i, j] = (val - r_min) / (r_max - r_min + 1e-9)
                k += 1

        # --- Role-dependent plotting ---
        if "buyer" in agent_name:
            rewards_matrix = rewards_matrix.T
            extent = [buyer_states.min(), buyer_states.max(),
                      seller_states.min(), seller_states.max()]
            x_label = "Buyer price"
            y_label = "Seller power level"
        else:
            extent = [seller_states.min(), seller_states.max(),
                      buyer_states.min(), buyer_states.max()]
            x_label = "Seller power level"
            y_label = "Buyer price"

        im = axes[idx].imshow(
            rewards_matrix, extent=extent, origin="lower",
            aspect="auto", cmap="viridis", vmin=0, vmax=1
        )
        axes[idx].set_title(agent_name)
        axes[idx].set_xlabel(x_label)
        axes[idx].set_ylabel(y_label)

        # Individual colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Reward")

    for ax in axes[n_agents:]:
        fig.delaxes(ax)

    fig.suptitle("Normalized Reward Heatmaps per Agent", fontsize=16)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Normalized subplot heatmaps saved to {filename}")


# ---------------------------------------------------------------------- #
# RAW VERSION (no normalization)
# ---------------------------------------------------------------------- #
def plot_agents_raw_heatmaps(num_points=50, filename="agents_reward_raw_heatmaps.png"):
    """Raw (non-normalized) reward heatmaps for all agents."""
    env = P2PEnergyEnv()
    env.reset()

    seller_states = np.linspace(0.0, 2.0, num_points)
    buyer_states = np.linspace(50.0, 100.0, num_points)

    agents = [agent.group_name for agent in env.agents]
    n_agents = len(agents)
    all_rewards = {agent: [] for agent in agents}

    print(f"🌀 Sweeping reward space with {num_points**2} samples ...")

    for buyer_state in buyer_states:
        for seller_state in seller_states:
            actions_dict = {}
            env.dones = {a.group_name: False for a in env.agents}
            env.current_agents = env.agents

            for seller in env.sellers:
                seller.state = np.array([seller_state / 2, seller_state / 2])
                actions_dict[seller.group_name] = np.zeros_like(seller.state)
            for buyer in env.buyers:
                buyer.state = buyer_state
                actions_dict[buyer.group_name] = np.zeros((1,))

            _, rewards, _, _ = env.step(actions_dict)
            for agent in agents:
                all_rewards[agent].append(rewards.get(agent, 0.0))

    cols = int(np.ceil(np.sqrt(n_agents)))
    rows = int(np.ceil(n_agents / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, agent_name in enumerate(agents):
        rewards_matrix = np.zeros((num_points, num_points))
        k = 0
        for i in range(num_points):
            for j in range(num_points):
                rewards_matrix[i, j] = all_rewards[agent_name][k]
                k += 1

        # --- Role-dependent plotting ---
        if "buyer" in agent_name:
            rewards_matrix = rewards_matrix.T
            extent = [buyer_states.min()*2, buyer_states.max()*2,
                      seller_states.min(), seller_states.max()]
            x_label = "Buyer price"
            y_label = "Seller power level"
        else:
            extent = [seller_states.min(), seller_states.max(),
                      buyer_states.min()*2, buyer_states.max()*2]
            x_label = "Seller power level"
            y_label = "Buyer price"

        im = axes[idx].imshow(
            rewards_matrix, extent=extent, origin="lower",
            aspect="auto", cmap="plasma"
        )
        axes[idx].set_title(agent_name)
        axes[idx].set_xlabel(x_label)
        axes[idx].set_ylabel(y_label)

        # Individual colorbar
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.set_label("Reward (raw scale)")

    for ax in axes[n_agents:]:
        fig.delaxes(ax)

    fig.suptitle("Raw Reward Heatmaps per Agent", fontsize=16)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Raw reward subplot heatmaps saved to {filename}")

def plot_total_reward_heatmap(num_points=50, filename="total_reward_heatmap.png", normalize=True):
    """
    Plots a single heatmap showing the sum of all agents' rewards
    across seller and buyer states.
    If normalize=True, scales the reward surface to [0,1].
    """
    env = P2PEnergyEnv()
    env.reset()

    seller_states = np.linspace(0.0, 2.0, num_points)
    buyer_states = np.linspace(50.0, 100.0, num_points)

    total_rewards = np.zeros((num_points, num_points))

    print(f"🌀 Sweeping total reward space with {num_points**2} samples ...")

    for i, buyer_state in enumerate(buyer_states):
        for j, seller_state in enumerate(seller_states):
            actions_dict = {}
            env.dones = {a.group_name: False for a in env.agents}
            env.current_agents = env.agents

            # Assign states and dummy actions
            for seller in env.sellers:
                seller.state = np.array([seller_state / 2, seller_state / 2])
                actions_dict[seller.group_name] = np.zeros_like(seller.state)
            for buyer in env.buyers:
                buyer.state = buyer_state
                actions_dict[buyer.group_name] = np.zeros((1,))

            _, rewards, _, _ = env.step(actions_dict)

            # Sum all agent rewards
            total_rewards[i, j] = sum(rewards.values())

    # Normalize if requested
    if normalize:
        r_min, r_max = total_rewards.min(), total_rewards.max()
        total_rewards = (total_rewards - r_min) / (r_max - r_min + 1e-9)
        cmap = "viridis"
        label = "Normalized total reward"
    else:
        cmap = "plasma"
        label = "Total reward (raw scale)"

    # Plot
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        total_rewards,  # transpose so x=seller, y=buyer
        extent=[2*seller_states.min(), 2*seller_states.max(),
                2*buyer_states.min(), 2*buyer_states.max()],
        origin="lower",
        aspect="auto",
        cmap=cmap
    )
    plt.colorbar(im, label=label)
    plt.xlabel("Seller power level")
    plt.ylabel("Buyer price")
    title = "Normalized Total Cooperative Reward" if normalize else "Total Cooperative Reward"
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Total reward heatmap saved to {filename}")


# ---------------------------------------------------------------------- #
# MAIN ENTRY POINT
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # plot_agents_heatmaps(num_points=30, filename="reward_space_normalized.png")
    # plot_agents_raw_heatmaps(num_points=30, filename="reward_space_raw.png")
    plot_total_reward_heatmap(num_points=30, filename="rewards_plots/total_reward_heatmap_normalized_2cons.png", normalize=True)
    plot_total_reward_heatmap(num_points=30, filename="rewards_plots/total_reward_heatmap_raw_2cons_2cons.png", normalize=False)
