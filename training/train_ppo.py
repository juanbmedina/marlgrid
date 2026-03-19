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
from training.policy_mapping import policy_mode
from training.rl_modules import EnergyRLModule
from envs.energy_env import P2PEnergyEnv

import os
import json
import shutil

torch, _ = try_import_torch()

register_energy_env()

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"


folder_path = "/workspace/exp_results"

for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.unlink(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

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

# ENV_CONFIG = dict(
#     enable_csv_log = False,
#     action_mode='delta',
#     max_steps=100,
#     power_step=0.1,
#     price_step=1,
#     pi_min=60.0,
#     pi_max=100.0,
#     lambda_sell=50,
#     lambda_buy=110,
#     lambda_u=7.8,
#     theta_u=1.0,
#     reward_mode="payoff",   # or "welfare"
#     training_mode="individual",
#     alpha=0.0,
#     beta=0.0,
#     pair_pricing_rule="midpoint"
# )

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
    welfare_mode="none"
)

energy_policy_mapping_fn = policy_mode(ENV_CONFIG)
# Use absolute path
storage_path = os.path.abspath("./exp_results")

os.makedirs(storage_path, exist_ok=True)
with open(os.path.join(storage_path, "env_config_used.json"), "w") as f:
    json.dump(ENV_CONFIG, f, indent=2, sort_keys=True)

model_config = {
                        "fcnet_hiddens": [128, 128],
                        "fcnet_activation": "relu",
                        "vf_share_layers": True,
                    }

env = P2PEnergyEnv()

rl_module_specs = {}

if ENV_CONFIG["training_mode"]=='individual':
    policy_ids = env.individual_policy_ids
else:
    policy_ids = env.group_policy_ids

for polname in policy_ids:
    rl_module_specs[polname] = RLModuleSpec(
                    model_config=model_config
                )

config = (
    PPOConfig()
    .environment("energy_market_ma",env_config=ENV_CONFIG)
    .multi_agent(
        policies={p for p in policy_ids},
        policy_mapping_fn=energy_policy_mapping_fn,
        policies_to_train=policy_ids,
    )
    .rl_module(
        rl_module_spec = MultiRLModuleSpec(
            rl_module_specs = rl_module_specs
    )
    )
    .learners(
        num_learners = 1,
        num_gpus_per_learner=1,
    )
    .training(
        gamma=0.95,
        lr=3e-4,
        train_batch_size_per_learner=1024,
        minibatch_size=128,
        num_epochs=6,
        entropy_coeff=0.001,
        # use_gae=True,
    )
    .env_runners(
        num_env_runners=8,
        num_cpus_per_env_runner=1,
        max_requests_in_flight_per_env_runner=1,
        batch_mode="complete_episodes", 
    )
)

# Train through Ray Tune
results = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=tune.RunConfig(
        stop={"num_env_steps_sampled_lifetime": 200000},
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