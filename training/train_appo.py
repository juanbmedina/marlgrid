from ray.rllib.algorithms.appo import APPOConfig
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
from envs.energy_env import P2PEnergyEnv

import os
import json
import shutil


torch, _ = try_import_torch()

register_energy_env()
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

# -------------------------------------------------------------------
# Housekeeping
# -------------------------------------------------------------------
CLEAR_RESULTS_DIR = True
folder_path = "/workspace/exp_results"
os.makedirs(folder_path, exist_ok=True)

if CLEAR_RESULTS_DIR:
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


my_multi_agent_progress_reporter = tune.CLIReporter(
    metric_columns={
        TRAINING_ITERATION: "iter",
        NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "mean return",
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MIN}": "min return",
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MAX}": "max return",
        f"{ENV_RUNNER_RESULTS}/{EPISODE_LEN_MEAN}": "ep len mean",
        f"{ENV_RUNNER_RESULTS}/{NUM_EPISODES}": "num ep",
    },
    max_report_frequency=30,
)


ENV_CONFIG = dict(
    enable_csv_log=False,
    action_mode="delta",
    max_steps=100,
    power_step=0.1,
    price_step=1,
    pi_min=60.0,
    pi_max=100.0,
    lambda_sell=50,
    lambda_buy=110,
    lambda_u=7.8,
    theta_u=1.0,
    reward_mode="payoff",   # or "welfare"
    training_mode="individual",
    beta=0.0,
    pair_pricing_rule="midpoint",
)

energy_policy_mapping_fn = policy_mode(ENV_CONFIG)

# Absolute path for Ray results.
storage_path = os.path.abspath("./exp_results")
exp_name = "energy_market_training"
os.makedirs(storage_path, exist_ok=True)

# NOTE:
# Your current evaluate scripts search env_config_used.json inside the TRIAL dir,
# not in storage_path. This file is still useful for bookkeeping, but evaluation may
# need a small fix if it only checks the checkpoint parent folder.
with open(os.path.join(storage_path, "env_config_used.json"), "w") as f:
    json.dump(ENV_CONFIG, f, indent=2, sort_keys=True)


model_config = {
    "fcnet_hiddens": [128, 128],
    "fcnet_activation": "relu",
    "vf_share_layers": True,
}

env = P2PEnergyEnv(ENV_CONFIG)

if ENV_CONFIG["training_mode"] == "individual":
    policy_ids = env.individual_policy_ids
else:
    policy_ids = env.group_policy_ids

rl_module_specs = {}
for polname in policy_ids:
    rl_module_specs[polname] = RLModuleSpec(model_config=model_config)


config = (
    APPOConfig()
    .framework("torch")
    .environment("energy_market_ma", env_config=ENV_CONFIG)
    .multi_agent(
        policies=set(policy_ids),
        policy_mapping_fn=energy_policy_mapping_fn,
        policies_to_train=policy_ids,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs=rl_module_specs,
        )
    )
    .training(
        # gamma=0.99,
        lr=5e-4,
        vtrace=True,
        # use_gae=True,
        # lambda_=1.0,
        # clip_param=0.3,
        # grad_clip=10.0,
        entropy_coeff=0.01,
        vf_loss_coeff=1.0,
        train_batch_size_per_learner=500,

        broadcast_interval=5,
        
        target_network_update_freq=2,
        circular_buffer_num_batches=4,
        circular_buffer_iterations_per_batch=2
    )
    .env_runners(
        num_env_runners=16,
        num_cpus_per_env_runner=1,
        batch_mode="complete_episodes",
        max_requests_in_flight_per_env_runner=1,
    )
    .learners(
        # For APPO/IMPALA with exactly one GPU, RLlib recommends a local learner
        # (num_learners=0) instead of one remote learner actor.
        num_learners=0,
        num_gpus_per_learner=1,
    )
)


results = tune.Tuner(
    "APPO",
    param_space=config,
    run_config=tune.RunConfig(
        stop={"num_env_steps_sampled_lifetime": 1_000_000},
        storage_path=storage_path,
        name=exp_name,
        verbose=1,
        progress_reporter=my_multi_agent_progress_reporter,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
            num_to_keep=5,
        ),
    ),
).fit()

print("\nTraining completed!")
