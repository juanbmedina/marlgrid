"""
Evaluation script for ROBUSTNESS-UNDER-NOISE analysis.

Variant of evaluate_legacy.py that adds two capabilities:

  1. Override of `profile_noise_std` and `profile_noise_type` at
     evaluation time. The trained policies saw deterministic
     demand/generation profiles; this script can expose them to
     noisy profiles to measure how well they tolerate the gap
     between forecast and reality.

  2. Deterministic per-episode noise. Each episode is seeded with
     env.reset(seed=episode_seed_base + ep), so the SAME noise
     realisation is reused across runs (e.g. when comparing
     scenarios at the same noise level). This makes comparisons
     fair and reproducible.

Everything else is identical to evaluate_legacy.py:
  - reads ENV_CONFIG from the saved train_ppo.py
  - uses the saved energy24h_env.py if present, falls back to
    envs.energy_env.P2PEnergyEnv otherwise
  - writes the CSV next to the checkpoint inside PPO_*/

The output CSV name defaults to:
  eval_noise_states_<std>.csv     (e.g. eval_noise_states_0p10.csv)
This is intentionally DIFFERENT from the legacy file name
(evaluation_agent_states.csv), so a noisy evaluation never
overwrites a clean one. Multiple noise levels can also coexist
in the same trial folder.

Run from the project root:

  # Clean (no noise) -- same as legacy but with deterministic seeding:
  python3 evaluate_legacy_noise.py path/to/scenario1_ISGT

  # Gaussian noise, ~+/-10%:
  python3 evaluate_legacy_noise.py path/to/scenario1_ISGT \
      --eval-noise-std 0.10

  # Uniform noise, ~+/-20%, restricted to one seed:
  python3 evaluate_legacy_noise.py path/to/scenario1_ISGT \
      --eval-noise-std 0.20 --eval-noise-type uniform --only-seed 42

  # Sweep multiple noise levels (bash):
  for s in 0.0 0.05 0.10 0.15 0.20; do
      python3 evaluate_legacy_noise.py path/to/scenario1_ISGT \
          --eval-noise-std $s
  done
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

from training.policy_mapping import policy_mode


# =====================================================================
# ENV_CONFIG extraction from saved train_ppo.py via AST
# =====================================================================

def extract_env_config(train_py_path: Path) -> dict:
    """Parse a saved train_ppo.py and return the value of ENV_CONFIG.

    Picks the LAST top-level assignment to ENV_CONFIG (so a commented-out
    earlier version is naturally ignored, and a live override later in the
    file wins). Evaluates only the dict(...) call in a sandboxed namespace
    that only knows the dict() builtin -- no other code runs.
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
            f"(This script only supports literal ENV_CONFIGs -- no variables.)"
        ) from e

    if not isinstance(cfg, dict):
        raise ValueError(f"ENV_CONFIG in {train_py_path} did not evaluate to a dict.")
    return cfg


def apply_noise_override(env_config: dict,
                         noise_std: float,
                         noise_type: Optional[str]) -> dict:
    """Return a copy of env_config with noise params overridden.

    We work on a copy so that the original parsed config (used for
    traceability) is preserved. We also force `profile_noise_type`
    to a sane default when noise is enabled but training never set
    a type.
    """
    cfg = dict(env_config)
    cfg["profile_noise_std"] = float(noise_std)
    if noise_type is not None:
        cfg["profile_noise_type"] = str(noise_type)
    if noise_std > 0 and "profile_noise_type" not in cfg:
        cfg["profile_noise_type"] = "gaussian"
    return cfg


# =====================================================================
# Saved env class loading (with fallback to current project's class)
# =====================================================================

def load_env_class(experiment_dir: Path):
    """Use <experiment_dir>/energy24h_env.py if present, else fall back
    to envs.energy_env.P2PEnergyEnv."""
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
# RLModule loading + action selection (mirrors evaluate_legacy.py)
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
    episode_seed_base: int,
) -> Path:
    print("\n" + "=" * 90)
    print(f"[eval] Checkpoint: {checkpoint_path}")
    print(f"[eval] noise_std={env_config.get('profile_noise_std', 0.0)}  "
          f"noise_type={env_config.get('profile_noise_type', 'gaussian')}")
    print(f"[eval] episode_seed_base={episode_seed_base}")
    print("=" * 90)

    energy_policy_mapping_fn = policy_mode(env_config)
    env = env_class(env_config)
    rlmodules = load_rlmodules(checkpoint_path, env_class, env_config)

    trial_dir = checkpoint_path.parent
    output_path = trial_dir / output_csv_name
    print(f"[eval] CSV -> {output_path}")

    # Persist the exact env_config used to produce this CSV, with a
    # matching filename suffix, so a noisy run is identifiable later.
    cfg_path = trial_dir / output_csv_name.replace(".csv", "_env_config.json")
    with cfg_path.open("w") as f:
        json.dump(env_config, f, indent=2, sort_keys=True)
    print(f"[eval] CFG -> {cfg_path}")

    rows = []
    for ep in range(num_episodes):
        # Deterministic per-episode seed so noise realisation is
        # reproducible across runs AND across scenarios.
        ep_seed = episode_seed_base + ep
        obs, _ = env.reset(seed=ep_seed)
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

            row = {"episode": ep + 1, "step": step, "ep_seed": int(ep_seed)}

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

        print(f"[eval]   Ep {ep + 1}/{num_episodes} (seed={ep_seed}) -> {step} steps")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[eval] Saved {len(df)} rows / {len(df.columns)} cols")
    return output_path


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate legacy experiments with profile-noise injection "
                    "(robustness analysis). Same as evaluate_legacy.py but "
                    "overrides profile_noise_std/profile_noise_type and uses "
                    "deterministic per-episode seeding."
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to dir containing seed_dirs and train_ppo.py.",
    )
    parser.add_argument(
        "--train-py", default=None,
        help="Path to saved train_ppo.py (default: <experiment_dir>/train_ppo.py).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Episodes per seed (default: 50).",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Output CSV filename (per seed). Default: "
             "eval_noise_states_<std>.csv "
             "(e.g. eval_noise_states_0p10.csv). Intentionally distinct "
             "from the legacy 'evaluation_agent_states.csv' so the two "
             "never collide.",
    )
    parser.add_argument(
        "--only-seed", default=None,
        help='Restrict to seeds whose dir name contains "seed<X>_".',
    )
    parser.add_argument(
        "--seed-glob", default="energy_market_training_seed*_run*",
        help="Glob pattern for seed directory names.",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip seeds whose output CSV already exists (resume mode).",
    )

    # ---- noise-specific overrides ----
    parser.add_argument(
        "--eval-noise-std", type=float, default=0.0,
        help="Override profile_noise_std for evaluation. "
             "0.10 means roughly +/-10%% multiplicative noise on D/G "
             "(gaussian std, or uniform half-width).",
    )
    parser.add_argument(
        "--eval-noise-type", default=None, choices=[None, "gaussian", "uniform"],
        help="Override profile_noise_type. Defaults to gaussian when "
             "--eval-noise-std > 0 if the training config had none.",
    )
    parser.add_argument(
        "--episode-seed-base", type=int, default=0,
        help="Base seed for per-episode RNG (env.reset(seed=base+ep)). "
             "Use the SAME base across scenarios at the same noise level "
             "so their noise realisations are directly comparable. "
             "Default: 0.",
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

    # ---- Inject noise override ----
    env_config = apply_noise_override(
        env_config, args.eval_noise_std, args.eval_noise_type,
    )
    print("\n[eval] FINAL env_config (after noise override):")
    print(json.dumps(env_config, indent=2, sort_keys=True))

    # ---- Build default output CSV name from noise level ----
    # Prefix is intentionally NOT 'evaluation_agent_states_' so it
    # never collides visually or literally with the legacy CSV.
    if args.output_csv is None:
        noise_tag = f"{args.eval_noise_std:.2f}".replace(".", "p")
        output_csv = f"eval_noise_states_{noise_tag}.csv"
    else:
        output_csv = args.output_csv

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
    print(f"[eval] Output CSV (per seed): {output_csv}")
    print(f"[eval] episode_seed_base:     {args.episode_seed_base}")
    print("#" * 90)

    successes, failures, skipped = [], [], []

    for seed_dir in seed_dirs:
        ckpt = find_latest_checkpoint_in_seed_dir(seed_dir)
        if ckpt is None:
            print(f"\n[eval] SKIP {seed_dir.name}: no PPO_*/checkpoint_* found")
            failures.append((seed_dir.name, "no checkpoint"))
            continue

        if args.skip_if_exists:
            existing = ckpt.parent / output_csv
            if existing.exists():
                print(f"\n[eval] SKIP {seed_dir.name}: {existing.name} already exists")
                skipped.append((seed_dir.name, str(existing)))
                continue

        try:
            out = evaluate_one_checkpoint(
                ckpt, env_class, env_config,
                args.num_episodes, output_csv,
                args.episode_seed_base,
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