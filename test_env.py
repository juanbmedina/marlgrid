import numpy as np
from envs.energy_env import P2PEnergyEnv
import matplotlib.pyplot as plt

def test_environment(num_steps=5):

    ENV_CONFIG = dict(
        enable_csv_log=False,
        max_steps=96,           # 24 horas * 4 pasos por hora
        steps_per_hour=4,
        hour_mode="hold_last",
        action_mode="absolute", # ya no delta
        pi_min=60.0,
        pi_max=100.0,
        lambda_sell=50,
        lambda_buy=110,
        lambda_u=110.0,         # 7.8 aquí sigue viéndose muy mal escalado para precios 60-110
        theta_u=1.0,
        reward_mode="payoff",
        training_mode="individual",   # o "group" con shared_policy
        alpha=0.1,
        beta=0.1,
        pair_pricing_rule="midpoint",
        agents_json_path="profiles/agents_profiles_24h.json",
        welfare_mode="gini"
    )
    # Initialize environment
    env = P2PEnergyEnv(ENV_CONFIG)
    obs, infos = env.reset()
    actions_dict = {}

    # print("\n=== RETURN RESET ===")
    # print("--- obs ---")
    # print(obs)

    seller_state = [[2, 60], [0.35, 60], [1.354, 60], [1.89, 60]]
    buyer_state = [[2.1, 100], [0.59, 100]]


    # for j, sid in enumerate(env.seller_ids):
    #     # env.P[j, :] = seller_state[j]
    #     # print("seller net: ", seller.net[0])

    #     env.offer_q[j] = seller_state[j][0]
    #     env.ask_p[j] = seller_state[j][1]
    #     actions_dict[sid] = [0.0, 0.0]

    # for i, bid in enumerate(env.buyer_ids):
    #     # env.pi[i] = buyer_state[i]
    #     # print("buyer buy: ", buyer.net[0])

    #     env.bid_q[i] = buyer_state[i][0]
    #     env.bid_p[i] = buyer_state[i][1]
    #     actions_dict[bid] =   [0.0, 0.0]

    o, r, d , _, _ = env.step(actions_dict)

    # print("=== FINAL OBSERVATION ===")
    # print(f"Global Obs: {o}")
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