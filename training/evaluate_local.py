"""
Evaluation script for energy_env_v2 (NO argparse).

Features:
- register_energy_env() called in main
- auto-detect latest checkpoint in EXPERIMENT_DIR
- auto-load env_config_used.json from trial dir
- RLModule-based deterministic evaluation (no Algorithm.compute_single_action)
- logs env_v2 state: P matrix, pi vector, mu scalar, target, balance, rewards
- saves CSV next to checkpoint folder (trial dir)

Control via env vars (optional):
  EVAL_EXPERIMENT_DIR   default: ./exp_results/energy_market_training
  EVAL_NUM_EPISODES     default: 50
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

# Must exist in your project (you said you must call it)
from envs.register_env import register_energy_env

# Your env + mapping
from envs.energy_env import P2PEnergyEnv
from training.policy_mapping import policy_mode


# -----------------------------
# CONFIG (edit or set env vars)
# -----------------------------
EXPERIMENT_DIR = os.environ.get("EVAL_EXPERIMENT_DIR", "./exp_results/exp_results_2026-03-04_17-30-37/energy_market_training")
NUM_EPISODES = int(os.environ.get("EVAL_NUM_EPISODES", "200"))
OUTPUT_CSV_NAME = os.environ.get("EVAL_OUTPUT_CSV", "evaluation_agent_states.csv")

CHECKPOINT_OVERRIDE = os.environ.get("EVAL_CHECKPOINT_PATH", "").strip()


def find_latest_checkpoint(experiment_dir: str) -> str:
    """Find latest PPO trial by mtime, then latest checkpoint by index."""
    experiment_dir = os.path.abspath(experiment_dir)
    trial_dirs = list(Path(experiment_dir).glob("PPO_*"))
    if not trial_dirs:
        raise ValueError(f"No PPO_* trial directories found in: {experiment_dir}")

    latest_trial = max(trial_dirs, key=lambda p: p.stat().st_mtime)
    checkpoints = list(latest_trial.glob("checkpoint_*"))
    if not checkpoints:
        raise ValueError(f"No checkpoint_* folders found in: {latest_trial}")

    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))
    return os.path.abspath(str(latest_checkpoint))


def load_env_config_used(checkpoint_path: str) -> dict:
    """
    Loads env_config_used.json from the *trial dir* (same level as checkpoint_*).
    Falls back to {} if missing.
    """
    ckpt = Path(checkpoint_path).resolve()
    trial_dir = ckpt.parent  # .../PPO_.../ (checkpoint_* lives here)
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

    env = P2PEnergyEnv()

    if env_config["training_mode"]=='individual':
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


def deterministic_action(env, agent_id: str, policy_id: str, rl_module: RLModule, obs: np.ndarray):
    """
    Deterministic action from Gaussian: take mean, clip to action space, and ensure correct shape.

    IMPORTANT:
    - In env_v2, seller action shape is (num_buyers,)
    - buyer action shape is (1,)
    """
    # forward inference
    input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0).float()}
    with torch.no_grad():
        out = rl_module.forward_inference(input_dict)

    dist_cls = rl_module.get_inference_action_dist_cls()
    action_dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])

    # deterministic = mean
    mean = action_dist._dist.mean
    action_np = convert_to_numpy(mean)[0]

    # clip to env action space
    space = env.action_spaces[agent_id]
    action_np = np.clip(action_np, space.low, space.high)

    # Enforce correct dimensionality from the env action space (no hardcoding [:2])
    # This is the part you were missing; your old slicing was only valid for exactly 2 buyers.
    expected_dim = int(np.prod(space.shape)) if hasattr(space, "shape") else None
    if expected_dim is not None:
        action_np = np.asarray(action_np).reshape(-1)[:expected_dim]
        action_np = action_np.reshape(space.shape)

    return action_np


def evaluate_and_save_states(checkpoint_path: str, num_episodes: int, output_csv_name: str):
    print("\n" + "=" * 90)
    print(f"[eval] Using checkpoint: {checkpoint_path}")
    print("=" * 90 + "\n")

    # Load env_config used during training
    env_config = load_env_config_used(checkpoint_path)

    energy_policy_mapping_fn = policy_mode(env_config)

    # Create env (env_v2)
    env = P2PEnergyEnv(env_config)

    # Load RLModules
    rlmodules = load_rlmodules(checkpoint_path, env_config)

    # Output CSV next to checkpoint folder (trial dir)
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

            # Compute action per live agent
            for agent_id, agent_obs in obs.items():
                policy_id = energy_policy_mapping_fn(agent_id, ep, None)
                rl_module = rlmodules[policy_id]
                actions[agent_id] = deterministic_action(env, agent_id, policy_id, rl_module, agent_obs)

            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

            # -----------------------------
            # LOG env_v2 STATE CORRECTLY
            # -----------------------------
            # env.P: (num_sellers, num_buyers)
            # env.pi: (num_buyers,)
            # env.mu: scalar (or possibly float)
            # env.target: scalar
            # seller_ids: ["seller_0", ...]
            # buyer_ids: ["buyer_0", ...]
            # -----------------------------
            # LOG usando infos (robusto a cambios en el env)
            # -----------------------------

            row = {
                "episode": ep + 1,
                "step": step,
            }
            # 1) Snapshot consistente de infos
            info_any = {}
            if isinstance(infos, dict) and len(infos) > 0:
                # Todos traen lo mismo → toma el primero
                first_key = next(iter(infos))
                info_any = infos.get(first_key, {})

            # 2) Volcar dinámicamente lo que venga del entorno
            if isinstance(info_any, dict):
                for k, v in info_any.items():

                    # Si es lista (P_flat, pi, etc.) → serializar a JSON
                    if isinstance(v, (list, tuple)):
                        row[k] = json.dumps(v)

                    # Si es numpy array (por seguridad)
                    elif isinstance(v, np.ndarray):
                        row[k] = json.dumps(v.tolist())

                    # Escalares numéricos
                    elif isinstance(v, (int, float, np.number)):
                        row[k] = float(v)

                    # Cualquier otra cosa
                    else:
                        row[k] = str(v)

            # 3) Rewards
            row[f"mean_reward"] = np.mean(list(rewards.values()))

            possible_agents = env.possible_agents

            for aid in possible_agents:
                state_val = env.state[aid]

                # --- REWARD siempre escalar ---
                row[f"{aid}_reward"] = float(rewards[aid])

                # --- STATE ---
                # Caso 1: ya es escalar
                if np.isscalar(state_val):
                    row[f"{aid}_state"] = float(state_val)

                else:
                    # Convertir a numpy y aplanar
                    state_array = np.asarray(state_val, dtype=np.float32).flatten()

                    # Si por alguna razón terminó siendo de tamaño 1
                    if state_array.size == 1:
                        row[f"{aid}_state"] = float(state_array[0])
                    else:
                        for idx, value in enumerate(state_array):
                            row[f"{aid}_state_{idx}"] = float(value)

            rows.append(row)

            obs = next_obs
            step += 1

        print(f"[eval] Episode {ep+1}/{num_episodes} finished in {step} steps")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 90)
    print(f"[eval] Saved: {output_path}")
    print(f"[eval] Rows: {len(df)} | Cols: {len(df.columns)}")
    print("=" * 90 + "\n")
    return output_path


def main():
    # You said this is required (same as training)
    register_energy_env()

    # Find checkpoint
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


