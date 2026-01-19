from ray.rllib.algorithms.ppo import PPOConfig
from envs.register_env import register_energy_env
from training.policy_mapping import energy_policy_mapping_fn

# Register your custom environment
register_energy_env()

# Simple PPO configuration for multi-agent
config = (
    PPOConfig()
    .environment("energy_market_ma")
    .multi_agent(
        # All agents share the same policy (simplest approach)
        policies={"seller_policy", "buyer_policy"},
        policy_mapping_fn=energy_policy_mapping_fn,
    )
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size=4000,
        num_sgd_iter=10,
    )
    .resources(
        num_gpus=1
    )
)

# Build and train
algo = config.build()

# Training loop
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward_mean = {result['env_runners']['episode_return_mean']:.2f}")
    
    # Save checkpoint every 10 iterations
    if i % 100 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved at {checkpoint_dir}")

print("Training complete!")