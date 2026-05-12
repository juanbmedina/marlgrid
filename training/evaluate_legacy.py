"""
Provisional local evaluation script for LEGACY experiments.

Evaluates experiments produced by the OLD pipeline (no inline evaluation,
no preserved env_config_used.json). Reads ENV_CONFIG from the manually
saved train_ppo.py, and uses the manually saved energy24h_env.py if
present (falls back to envs.energy_env.P2PEnergyEnv otherwise).

Layout assumed:

  <experiment_dir>/
  ├── train_ppo.py                              ← saved, source of ENV_CONFIG
  ├── energy24h_env.py                          ← saved env (optional)
  ├── energy_market_training_seed42_run1/
  │   └── PPO_energy_market_run/
  │       ├── checkpoint_*/
  │       └── progress.csv
  ├── energy_market_training_seed43_run2/
  │   └── ...

Run from the project root (so `envs.community_welfare_rewards`,
`training.policy_mapping`, etc. resolve normally):

  python3 evaluate_legacy.py path/to/scenario1_ISGT
  python3 evaluate_legacy.py path/to/scenario1_ISGT --num-episodes 100
  python3 evaluate_legacy.py path/to/scenario1_ISGT --only-seed 42
  python3 evaluate_legacy.py path/to/scenario1_ISGT --skip-if-exists

For each seed, writes evaluation_agent_states.csv next to the checkpoint
(inside PPO_*/), so a follow-up rsync/zip captures it together with the
existing progress.csv.
"""

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy

from training.policy_mapping import policy_mode  # current project's mapping


# =====================================================================
# ENV_CONFIG extraction from saved train_ppo.py via AST
# =====================================================================

def extract_env_config(train_py_path: Path) -> dict:
    """Parse a saved train_ppo.py and return the value of ENV_CONFIG.

    Picks the LAST top-level assignment to ENV_CONFIG (so a commented-out
    earlier version is naturally ignored, and a live override later in the
    file wins). Evaluates only the dict(...) call in a sandboxed namespace
    that only knows the dict() builtin — no other code runs.
    """
    src = train_py_path.read_text()
    tree = ast.parse(src)

    last_value = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "ENV_CONFIG"):
            last_value = node.value

    if last_value is None:
        raise ValueError(f"No ENV_CONFIG assignment found in {train_py_path}")

    expr_src = ast.unparse(last_value)
    try:
        cfg = eval(expr_src, {"__builtins__": {}, "dict": dict})
    except Exception as e:
        raise ValueError(
            f"Could not evaluate ENV_CONFIG from {train_py_path}.\n"
            f"Source: {expr_src}\nError: {e!r}\n"
            f"(This script only supports literal ENV_CONFIGs — no variables.)"
        ) from e

    if not isinstance(cfg, dict):
        raise ValueError(f"ENV_CONFIG in {train_py_path} did not evaluate to a dict.")
    return cfg


# =====================================================================
# Saved env class loading (with fallback to current project's class)
# =====================================================================

def load_env_class(experiment_dir: Path):
    """Use <experiment_dir>/energy24h_env.py if it exists; otherwise fall
    back to envs.energy_env.P2PEnergyEnv.

    The saved file still relies on `envs.community_welfare_rewards` and
    `ray.rllib.env.multi_agent_env`, so this script must be run from the
    project root.
    """
    saved = experiment_dir / "energy24h_env.py"
    if saved.exists():
        spec = importlib.util.spec_from_file_location("legacy_energy_env", str(saved))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"[eval] Using SAVED env class from: {saved}")
        return module.P2PEnergyEnv

    from envs.energy_env import P2PEnergyEnv
    print(f"[eval] No saved energy24h_env.py at {experiment_dir}; "
          f"falling back to envs.energy_env.P2PEnergyEnv (current project).")
    return P2PEnergyEnv


# =====================================================================
# Discovery helpers
# =====================================================================

def find_seed_dirs(
    base_dir: Path,
    pattern: str,
    only_seed: Optional[str],
) -> List[Path]:
    candidates = sorted(base_dir.glob(pattern))
    candidates = [p for p in candidates if p.is_dir()]
    if only_seed:
        token = f"seed{only_seed}_"
        candidates = [p for p in candidates if token in p.name]
    return candidates


def find_latest_checkpoint_in_seed_dir(seed_dir: Path) -> Optional[Path]:
    trial_dirs = [p for p in seed_dir.iterdir()
                  if p.is_dir() and p.name.startswith("PPO_")]
    if not trial_dirs:
        return None
    trial_dir = max(trial_dirs, key=lambda p: p.stat().st_mtime)
    checkpoints = list(trial_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))


# =====================================================================
# RLModule loading + action selection (mirrors evaluate.py)
# =====================================================================

def load_rlmodules(checkpoint_path: Path, env_class, env_config: dict) -> dict:
    base = checkpoint_path / "learner_group" / "learner" / "rl_module"
    if not base.exists():
        raise FileNotFoundError(f"RLModule base path not found: {base}")

    env = env_class(env_config)
    training_mode = env_config.get("training_mode", "group")
    if training_mode == "individual":
        policy_ids = env.individual_policy_ids
    else:
        policy_ids = env.group_policy_ids

    rlmodules = {}
    for policy_id in policy_ids:
        policy_path = base / policy_id
        if not policy_path.exists():
            raise ValueError(f"RLModule path not found for {policy_id}: {policy_path}")
        print(f"[eval]   Loading {policy_id}")
        m = RLModule.from_checkpoint(str(policy_path))
        m.eval()
        rlmodules[policy_id] = m
    return rlmodules


def deterministic_action(env, agent_id, rl_module, obs):
    obs = np.asarray(obs, dtype=np.float32)
    input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}
    with torch.no_grad():
        out = rl_module.forward_inference(input_dict)
    if Columns.ACTIONS in out:
        action_np = convert_to_numpy(out[Columns.ACTIONS])[0]
    else:
        if Columns.ACTION_DIST_INPUTS not in out:
            raise KeyError(f"RLModule output keys: {list(out.keys())}")
        dist_cls = rl_module.get_inference_action_dist_cls()
        action_dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
        action_np = convert_to_numpy(action_dist.to_deterministic().sample())[0]

    space = env.action_spaces[agent_id]
    action_np = np.asarray(action_np, dtype=np.float32)
    action_np = np.clip(action_np, space.low, space.high)
    if hasattr(space, "shape") and space.shape is not None:
        expected_dim = int(np.prod(space.shape))
        action_np = action_np.reshape(-1)[:expected_dim]
        action_np = action_np.reshape(space.shape)
    return action_np


# =====================================================================
# Per-checkpoint evaluation (one seed)
# =====================================================================

def evaluate_one_checkpoint(
    checkpoint_path: Path,
    env_class,
    env_config: dict,
    num_episodes: int,
    output_csv_name: str,
) -> Path:
    print("\n" + "=" * 90)
    print(f"[eval] Checkpoint: {checkpoint_path}")
    print("=" * 90)

    energy_policy_mapping_fn = policy_mode(env_config)
    env = env_class(env_config)
    rlmodules = load_rlmodules(checkpoint_path, env_class, env_config)

    trial_dir = checkpoint_path.parent
    output_path = trial_dir / output_csv_name
    print(f"[eval] CSV -> {output_path}")

    rows = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        step = 0

        while not (terminateds.get("__all__", False)
                   or truncateds.get("__all__", False)):
            actions = {}
            for aid, agent_obs in obs.items():
                policy_id = energy_policy_mapping_fn(aid, ep, None)
                actions[aid] = deterministic_action(
                    env, aid, rlmodules[policy_id], agent_obs,
                )
            next_obs, _, terminateds, truncateds, infos = env.step(actions)

            row = {"episode": ep + 1, "step": step}

            info_any = {}
            if isinstance(infos, dict) and len(infos) > 0:
                first_key = next(iter(infos))
                info_any = infos.get(first_key, {})

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

            rewards = info_any.get("payoff", {}) if isinstance(info_any, dict) else {}
            reward_values = list(rewards.values()) if isinstance(rewards, dict) else []
            row["mean_reward"] = float(np.mean(reward_values)) if reward_values else 0.0

            for aid in env.possible_agents:
                row[f"{aid}_reward"] = (
                    float(rewards.get(aid, 0.0)) if isinstance(rewards, dict) else 0.0
                )
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

        print(f"[eval]   Ep {ep + 1}/{num_episodes} -> {step} steps")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[eval] Saved {len(df)} rows / {len(df.columns)} cols")
    return output_path


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate legacy experiments (no preserved env_config) "
                    "by extracting ENV_CONFIG from the saved train_ppo.py."
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to dir containing seed_dirs and train_ppo.py "
             "(e.g. exp_results_repro/exp_results_xxx/scenario1_ISGT).",
    )
    parser.add_argument(
        "--train-py", default=None,
        help="Path to saved train_ppo.py "
             "(default: <experiment_dir>/train_ppo.py).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Episodes per seed (default: 50).",
    )
    parser.add_argument(
        "--output-csv", default="evaluation_agent_states.csv",
        help="Output CSV filename, written next to the checkpoint.",
    )
    parser.add_argument(
        "--only-seed", default=None,
        help='Restrict to seeds whose dir name contains "seed<X>_" (e.g. 42).',
    )
    parser.add_argument(
        "--seed-glob", default="energy_market_training_seed*_run*",
        help="Glob pattern for seed directory names.",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip seeds whose output CSV already exists (resume mode).",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.is_dir():
        sys.exit(f"experiment_dir does not exist or is not a directory: "
                 f"{experiment_dir}")

    train_py = (Path(args.train_py).resolve()
                if args.train_py else experiment_dir / "train_ppo.py")
    if not train_py.is_file():
        sys.exit(f"train_ppo.py not found: {train_py}\n"
                 f"Pass --train-py if it lives somewhere else.")

    # ---- Recover ENV_CONFIG from saved train_ppo.py ----
    env_config = extract_env_config(train_py)
    print(f"\n[eval] Extracted ENV_CONFIG from: {train_py}")
    print(json.dumps(env_config, indent=2, sort_keys=True))

    # ---- Load env class (saved one if present, else current project's) ----
    env_class = load_env_class(experiment_dir)

    # ---- Discover seeds ----
    seed_dirs = find_seed_dirs(experiment_dir, args.seed_glob, args.only_seed)
    if not seed_dirs:
        sys.exit(f"No seed dirs matching '{args.seed_glob}' in {experiment_dir} "
                 f"(only_seed='{args.only_seed}').")

    print("\n" + "#" * 90)
    print(f"[eval] Discovered {len(seed_dirs)} seed dir(s):")
    for sd in seed_dirs:
        print(f"   - {sd.name}")
    print("#" * 90)

    successes, failures, skipped = [], [], []

    for seed_dir in seed_dirs:
        ckpt = find_latest_checkpoint_in_seed_dir(seed_dir)
        if ckpt is None:
            print(f"\n[eval] SKIP {seed_dir.name}: no PPO_*/checkpoint_* found")
            failures.append((seed_dir.name, "no checkpoint"))
            continue

        if args.skip_if_exists:
            existing = ckpt.parent / args.output_csv
            if existing.exists():
                print(f"\n[eval] SKIP {seed_dir.name}: {existing.name} already exists")
                skipped.append((seed_dir.name, str(existing)))
                continue

        try:
            out = evaluate_one_checkpoint(
                ckpt, env_class, env_config,
                args.num_episodes, args.output_csv,
            )
            successes.append((seed_dir.name, str(out)))
        except Exception as e:
            print(f"[eval] FAILED {seed_dir.name}: {e!r}")
            failures.append((seed_dir.name, str(e)))

    print("\n" + "#" * 90)
    print(f"[eval] DONE. {len(successes)} ok, "
          f"{len(failures)} failed, {len(skipped)} skipped.")
    for name, path in successes:
        print(f"  OK   {name} -> {path}")
    for name, path in skipped:
        print(f"  SKIP {name} -> {path}")
    for name, err in failures:
        print(f"  FAIL {name}: {err}")
    print("#" * 90)


if __name__ == "__main__":
    main()