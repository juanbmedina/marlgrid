# =============================================================================
# Reproducibility preamble — MUST run before any import that might load torch
# (Ray imports torch transitively, so these env vars go on line 1).
# =============================================================================
import os

# Accept SEED from env var MARL_SEED (default 42). Allows launching with
# `MARL_SEED=42 python train_ppo.py`, `MARL_SEED=43 python train_ppo.py`, ...
SEED = int(os.environ.get("MARL_SEED", "42"))
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
from training.seed_callbacks import (
    SeedEverythingCallback,
    DeterministicPPOTorchRLModule,
)
from envs.energy_env import P2PEnergyEnv

import json
import shutil

torch, _ = try_import_torch()

# Force deterministic kernels now that torch is loaded. `warn_only=False`
# turns any non-deterministic op into a HARD ERROR with a clear message
# pointing at the op. Useful for diagnosing residual reproducibility
# leaks. Switch back to `warn_only=True` for production once identified.
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

register_energy_env()

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

# NOTE: we do NOT wipe /workspace/exp_results between runs anymore — doing so
# would destroy results from previously-run seeds. If you need a clean start,
# delete the folder manually before launching.
# folder_path = "/workspace/exp_results"
# for item in os.listdir(folder_path):
#     item_path = os.path.join(folder_path, item)
#     if os.path.isfile(item_path) or os.path.islink(item_path):
#         os.unlink(item_path)
#     elif os.path.isdir(item_path):
#         shutil.rmtree(item_path)

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
    training_mode="individual",   # o "group" con shared_policy
    pair_pricing_rule="midpoint",
    agents_json_path="profiles/agents_profiles_24h.json",
    welfare_mode="gini",
    obs_mode="local"
)


# ============================================================
# Experiment registry: optionally override ENV_CONFIG based on
# MARL_EXPERIMENT_NAME. Stays no-op if the env var is unset or "default",
# so `python3 -m training.train_ppo` keeps working as before without
# any wrapper.
# ============================================================
_exp_name = os.environ.get("MARL_EXPERIMENT_NAME", "default")
_env_overrides = {}
_exp_notes = ""

if _exp_name and _exp_name != "default":
    from training.experiments_registry import get_experiment
    _exp = get_experiment(_exp_name)
    _env_overrides = _exp.get("env_config", {})
    _exp_notes = _exp.get("notes", "")

    print("=" * 70)
    print(f"[train] Experiment: {_exp_name}")
    print(f"[train]   notes:    {_exp_notes or '(no notes)'}")
    print(f"[train]   ENV_CONFIG overrides ({len(_env_overrides)} keys):")
    for k, v in sorted(_env_overrides.items()):
        print(f"             {k} = {v!r}")
    print("=" * 70)

    ENV_CONFIG = {**ENV_CONFIG, **_env_overrides}
else:
    print(f"[train] No experiment override (MARL_EXPERIMENT_NAME='{_exp_name}')")


energy_policy_mapping_fn = policy_mode(ENV_CONFIG)
# Use absolute path
storage_path = os.path.abspath("./exp_results")

os.makedirs(storage_path, exist_ok=True)
with open(os.path.join(storage_path, "env_config_used.json"), "w") as f:
    json.dump(ENV_CONFIG, f, indent=2, sort_keys=True)

# Record seed and library versions so each run is fully identifiable later.
import ray as _ray_for_meta
with open(os.path.join(storage_path, "run_meta.json"), "w") as f:
    json.dump(
        {
            "seed": SEED,
            "experiment_name": _exp_name,
            "experiment_notes": _exp_notes,
            "experiment_env_overrides": _env_overrides,
            "ray_version": _ray_for_meta.__version__,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        f,
        indent=2,
        sort_keys=True,
    )

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

for pol_idx, polname in enumerate(policy_ids):
    # Distinct per-policy seed, deterministic function of master SEED.
    # Different policies => different weight init (healthy); same master SEED
    # across runs => same weights across runs (what we want).
    per_policy_seed = SEED * 1000 + pol_idx
    policy_model_config = {**model_config, "_deterministic_seed": per_policy_seed}
    rl_module_specs[polname] = RLModuleSpec(
        module_class=DeterministicPPOTorchRLModule,
        model_config=policy_model_config,
    )

config = (
    PPOConfig()
    .environment("energy_market_ma",env_config=ENV_CONFIG)
    .debugging(seed=SEED)
    # .callbacks(SeedEverythingCallback)
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
        num_gpus_per_learner=1,  # REPRODUCIBILITY MODE: CPU only.
                                 # Set back to 1 once reproducibility is confirmed.
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
        num_env_runners=8,  # REPRODUCIBILITY: single env_runner.
        num_cpus_per_env_runner=1,
        max_requests_in_flight_per_env_runner=1,
        batch_mode="complete_episodes",
    )
)

# Name the run folder. If MARL_RUN_TAG env var is set (e.g. by
# run_experiments_repro.sh), use it so that repeating the same seed
# creates separate folders (seed42_run1, seed42_run2, ...). Otherwise
# fall back to naming by seed.
_run_tag = os.environ.get("MARL_RUN_TAG", f"seed{SEED}")

# Train through Ray Tune
results = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=tune.RunConfig(
        stop={"num_env_steps_sampled_lifetime": 250000},
        storage_path=storage_path,
        name=f"energy_market_training_{_run_tag}",
        verbose = 1,
        progress_reporter=my_multi_agent_progress_reporter,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            num_to_keep=3,
        )
    ),
).fit()

print("\nTraining completed!")