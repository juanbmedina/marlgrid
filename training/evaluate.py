"""
Evaluation script for energy_env_v2 / SAC (NO argparse).

Features:
- register_energy_env() called in main
- auto-detect latest checkpoint in EXPERIMENT_DIR
- auto-load env_config_used.json from trial dir
- RLModule-based deterministic evaluation
- logs env state, infos, rewards, and agent states
- saves CSV next to checkpoint folder (trial dir)

Control via env vars (optional):
  EVAL_EXPERIMENT_DIR   default: ./exp_results/energy_market_training
  EVAL_NUM_EPISODES     default: 200
  EVAL_OUTPUT_CSV       default: evaluation_agent_states.csv
  EVAL_CHECKPOINT_PATH  (optional override, skips autodetect)
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy

from envs.register_env import register_energy_env
from envs.energy_env import P2PEnergyEnv
from training.policy_mapping import policy_mode


# -----------------------------
# CONFIG (edit or set env vars)
# -----------------------------
EXPERIMENT_DIR = os.environ.get("EVAL_EXPERIMENT_DIR", "./exp_results/energy_market_training")
NUM_EPISODES = int(os.environ.get("EVAL_NUM_EPISODES", "50"))
OUTPUT_CSV_NAME = os.environ.get("EVAL_OUTPUT_CSV", "evaluation_agent_states.csv")
CHECKPOINT_OVERRIDE = os.environ.get("EVAL_CHECKPOINT_PATH", "").strip()


def find_latest_checkpoint(experiment_dir: str) -> str:
    """
    Find latest trial directory by mtime, then latest checkpoint_* by index.
    No hardcoding of PPO/SAC trial name prefixes.
    """
    experiment_dir = Path(os.path.abspath(experiment_dir))
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment dir does not exist: {experiment_dir}")

    trial_dirs = [p for p in experiment_dir.iterdir() if p.is_dir()]
    if not trial_dirs:
        raise ValueError(f"No trial directories found in: {experiment_dir}")

    # Keep only trial dirs that actually contain checkpoints
    valid_trials = [p for p in trial_dirs if list(p.glob("checkpoint_*"))]
    if not valid_trials:
        raise ValueError(f"No checkpoint_* folders found under any trial dir in: {experiment_dir}")

    latest_trial = max(valid_trials, key=lambda p: p.stat().st_mtime)
    checkpoints = list(latest_trial.glob("checkpoint_*"))
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))

    return os.path.abspath(str(latest_checkpoint))


def load_env_config_used(checkpoint_path: str) -> dict:
    """
    Loads env_config_used.json from the trial dir (same level as checkpoint_*).
    Falls back to {} if missing.
    """
    ckpt = Path(checkpoint_path).resolve()
    trial_dir = ckpt.parent
    json_path = trial_dir / "env_config_used.json"

    if not json_path.exists():
        print(f"[eval] env_config_used.json not found at: {json_path}")
        print("[eval] Falling back to env defaults ({}).")
        return {}

    with json_path.open("r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"[eval] env_config_used.json is not a dict: {json_path}")

    print(f"[eval] Loaded env_config from: {json_path}")
    return cfg


def load_rlmodules(checkpoint_path: str, env_config: dict) -> dict:
    """
    Load RLModules for each policy directly from checkpoint:
      checkpoint_*/learner_group/learner/rl_module/{policy_id}
    """
    base = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    if not base.exists():
        raise FileNotFoundError(f"[eval] RLModule base path not found: {base}")

    env = P2PEnergyEnv(env_config)

    training_mode = env_config.get("training_mode", "group")
    if training_mode == "individual":
        policy_ids = env.individual_policy_ids
    else:
        policy_ids = env.group_policy_ids

    rlmodules = {}
    for policy_id in policy_ids:
        policy_path = base / policy_id
        if not policy_path.exists():
            raise ValueError(f"[eval] RLModule path not found for {policy_id}: {policy_path}")

        print(f"[eval] Loading RLModule {policy_id} from {policy_path}")
        m = RLModule.from_checkpoint(str(policy_path))
        m.eval()
        rlmodules[policy_id] = m

    return rlmodules


def deterministic_action(
    env,
    agent_id: str,
    policy_id: str,
    rl_module: RLModule,
    obs: np.ndarray,
):
    """
    Get deterministic action from RLModule inference output.
    Works for SAC more robustly than directly reading private distribution attrs.
    """
    del policy_id  # not needed, kept only for signature consistency

    obs = np.asarray(obs, dtype=np.float32)
    input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

    with torch.no_grad():
        out = rl_module.forward_inference(input_dict)

    # Preferred path: module already returns deterministic actions
    if Columns.ACTIONS in out:
        action_np = convert_to_numpy(out[Columns.ACTIONS])[0]
    else:
        # Fallback: build action distribution and force deterministic sample
        if Columns.ACTION_DIST_INPUTS not in out:
            raise KeyError(
                f"[eval] RLModule output has neither {Columns.ACTIONS} nor "
                f"{Columns.ACTION_DIST_INPUTS}. Keys: {list(out.keys())}"
            )

        dist_cls = rl_module.get_inference_action_dist_cls()
        action_dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
        action_np = convert_to_numpy(action_dist.to_deterministic().sample())[0]

    # Clip to action space
    space = env.action_spaces[agent_id]
    action_np = np.asarray(action_np, dtype=np.float32)
    action_np = np.clip(action_np, space.low, space.high)

    # Enforce expected dimensionality from env action space
    if hasattr(space, "shape") and space.shape is not None:
        expected_dim = int(np.prod(space.shape))
        action_np = action_np.reshape(-1)[:expected_dim]
        action_np = action_np.reshape(space.shape)

    return action_np


def evaluate_and_save_states(checkpoint_path: str, num_episodes: int, output_csv_name: str):
    print("\n" + "=" * 90)
    print(f"[eval] Using checkpoint: {checkpoint_path}")
    print("=" * 90 + "\n")

    env_config = load_env_config_used(checkpoint_path)
    env_config["phase"] = "eval"

    _scale = float(os.environ.get("MARL_REALITY_SCALE", "1.0"))
    if _scale != 1.0:
        for _k in ("sigma_reality_gen", "sigma_reality_dem"):
            _v = env_config.get(_k, 0.0)
            if isinstance(_v, dict):
                env_config[_k] = {a: x * _scale for a, x in _v.items()}
            elif isinstance(_v, (list, tuple)):
                env_config[_k] = [x * _scale for x in _v]
            else:
                env_config[_k] = float(_v) * _scale


    energy_policy_mapping_fn = policy_mode(env_config)

    env = P2PEnergyEnv(env_config)
    rlmodules = load_rlmodules(checkpoint_path, env_config)

    trial_dir = Path(checkpoint_path).resolve().parent
    output_path = trial_dir / output_csv_name
    print(f"[eval] Saving CSV to: {output_path}\n")

    rows = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        step = 0

        while not (terminateds.get("__all__", False) or truncateds.get("__all__", False)):
            actions = {}

            for agent_id, agent_obs in obs.items():
                policy_id = energy_policy_mapping_fn(agent_id, ep, None)
                rl_module = rlmodules[policy_id]
                actions[agent_id] = deterministic_action(
                    env=env,
                    agent_id=agent_id,
                    policy_id=policy_id,
                    rl_module=rl_module,
                    obs=agent_obs,
                )

            next_obs, _, terminateds, truncateds, infos = env.step(actions)

            row = {
                "episode": ep + 1,
                "step": step,
            }

            # Snapshot consistente de infos
            info_any = {}
            if isinstance(infos, dict) and len(infos) > 0:
                first_key = next(iter(infos))
                info_any = infos.get(first_key, {})

            # Volcar dinámicamente lo que venga del entorno
            if isinstance(info_any, dict):
                for k, v in info_any.items():
                    if isinstance(v, (list, tuple)):
                        row[k] = json.dumps(v)
                    elif isinstance(v, np.ndarray):
                        row[k] = json.dumps(v.tolist())
                    elif isinstance(v, (int, float, np.number)):
                        row[k] = float(v)
                    else:
                        row[k] = str(v)

            rewards = info_any['payoff']

            # Mean reward over agents
            reward_values = list(rewards.values()) if isinstance(rewards, dict) else []
            row["mean_reward"] = float(np.mean(reward_values)) if reward_values else 0.0

            for aid in env.possible_agents:
                # Reward
                row[f"{aid}_reward"] = float(rewards.get(aid, 0.0))

                # State
                state_val = env.state.get(aid, None)

                if state_val is None:
                    row[f"{aid}_state"] = np.nan
                    continue

                if np.isscalar(state_val):
                    row[f"{aid}_state"] = float(state_val)
                else:
                    state_array = np.asarray(state_val, dtype=np.float32).flatten()
                    if state_array.size == 1:
                        row[f"{aid}_state"] = float(state_array[0])
                    else:
                        for idx, value in enumerate(state_array):
                            row[f"{aid}_state_{idx}"] = float(value)

            rows.append(row)
            obs = next_obs
            step += 1

        print(f"[eval] Episode {ep + 1}/{num_episodes} finished in {step} steps")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 90)
    print(f"[eval] Saved: {output_path}")
    print(f"[eval] Rows: {len(df)} | Cols: {len(df.columns)}")
    print("=" * 90 + "\n")

    return output_path


def main():
    register_energy_env()

    if CHECKPOINT_OVERRIDE:
        ckpt = os.path.abspath(CHECKPOINT_OVERRIDE)
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"EVAL_CHECKPOINT_PATH does not exist: {ckpt}")
        checkpoint_path = ckpt
    else:
        checkpoint_path = find_latest_checkpoint(EXPERIMENT_DIR)

    evaluate_and_save_states(checkpoint_path, NUM_EPISODES, OUTPUT_CSV_NAME)


if __name__ == "__main__":
    main()