from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.framework import try_import_torch
from ray import tune
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    EPISODE_RETURN_MAX,
    EPISODE_LEN_MEAN,
    NUM_EPISODES,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.tune.result import TRAINING_ITERATION

from envs.register_env import register_energy_env
from training.policy_mapping import energy_policy_mapping_fn

import os

torch, _ = try_import_torch()

register_energy_env()

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

my_multi_agent_progress_reporter = tune.CLIReporter(
    metric_columns={
        # **{
            TRAINING_ITERATION: "iter",
            NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "mean return",
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MIN}": "min return",
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MAX}": "max return",
            f"{ENV_RUNNER_RESULTS}/{EPISODE_LEN_MEAN}": "ep len mean",
            f"{ENV_RUNNER_RESULTS}/{NUM_EPISODES}": "num ep",
        # },
        # **{
        #     f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/{pid}": f"return {pid}"
        #     for pid in ["seller_policy", "buyer_policy"]
        # },
    },
    max_report_frequency = 30
)

config = (
    PPOConfig()
    .environment("energy_market_ma")
    .multi_agent(
        policies={"seller_policy", "buyer_policy"},
        policy_mapping_fn=energy_policy_mapping_fn,
        policies_to_train=["seller_policy", "buyer_policy"],
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "seller_policy": RLModuleSpec(),
                "buyer_policy": RLModuleSpec(),
            }
        )
    )
    .learners(
        num_learners = 2,
        num_gpus_per_learner=0,
    )
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size_per_learner=4096, 
        minibatch_size=512,
        num_epochs=10,
        
        # Additional optimizations
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
    )
    .env_runners(
        # Parallel environment rollouts on CPU
        num_env_runners=6,
        num_cpus_per_env_runner=1,
    )
)

# Use absolute path
storage_path = os.path.abspath("./exp_results")

# Train through Ray Tune
results = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=tune.RunConfig(
        stop={"num_env_steps_sampled_lifetime": 400000},
        storage_path=storage_path,
        name="energy_market_training",
        verbose = 1,
        progress_reporter=my_multi_agent_progress_reporter,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
            num_to_keep=5,
        )
    ),
).fit()

print("\nTraining completed!")  