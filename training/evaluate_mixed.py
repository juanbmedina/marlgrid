"""
Mixed-policy evaluator: trained PPO agents play against heuristic agents.

This script extends evaluate_legacy.py with the ability to substitute one
or more agent slots with a fixed heuristic policy, while the remaining
slots use their trained PPO RLModule. The action space is identical for
all agents (Box(0,1,(2,))), so any slot can be served by either source.

Three heuristics are provided (same names as evaluate_heuristic.py):

  grid_only  : the agent never participates in P2P. q=0, so all generation
               is exported to the grid (sellers) or all demand is imported
               from the grid (buyers). Price field is unused.

  midpoint   : the agent offers full quantity at the midpoint of its valid
               price range. For sellers this is between min_cost and
               pi_max; for buyers between pi_min and pi_max.

  greedy     : the agent participates with full quantity and the most
               aggressive price for clearing. Sellers ask at min_cost
               (a_p=0), buyers bid at pi_max (a_p=1). This is the
               "always-trade-in-P2P" policy.

Usage (run from the project root, same as evaluate_legacy.py):

  # Replace agent_4 and agent_5 with the midpoint heuristic, keep
  # agent_0..agent_3 as the trained PPO policies, 50 episodes:
  python3 evaluate_mixed.py path/to/scenario1_ISGT \
      --heuristic midpoint --heuristic-agents 4,5 --num-episodes 50

  # Pure heuristic baseline (no trained policies loaded):
  python3 evaluate_mixed.py path/to/scenario1_ISGT \
      --heuristic greedy --heuristic-agents all

  # Single seed, resume mode:
  python3 evaluate_mixed.py path/to/scenario1_ISGT \
      --heuristic grid_only --heuristic-agents 5 \
      --only-seed 42 --skip-if-exists

Output naming: by default the CSV is

    eval_mixed_<heuristic>_h<idxs>.csv

written next to each seed's checkpoint, where <idxs> is the sorted list
of heuristic agent indices concatenated (e.g. h45 for agent_4 + agent_5,
h012345 for the all-heuristic case). Override with --output-csv if you
want something else.
"""

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.numpy import convert_to_numpy

from training.policy_mapping import policy_mode


# =====================================================================
# Heuristic policies
# =====================================================================

HEURISTIC_CHOICES = ("grid_only", "midpoint", "greedy")


def _heuristic_grid_only(env, agent_id: str) -> np.ndarray:
    """No P2P participation: q=0 -> all energy via grid slack settlement.

    The price component is irrelevant when q=0 (the agent does not appear
    in the clearing loop), but we set it to the midpoint just for a clean
    state log.
    """
    return np.array([0.0, 0.5], dtype=np.float32)


def _heuristic_midpoint(env, agent_id: str) -> np.ndarray:
    """Offer/bid full quantity at the midpoint of the valid price range.

    For sellers a_p=0.5 -> ask = (min_cost + pi_max)/2
    For buyers  a_p=0.5 -> bid = (pi_min   + pi_max)/2
    Neutral agents place no offer (q=0).
    """
    idx = env.agent_name_to_idx[agent_id]
    if env.role[idx] == 0:
        return np.array([0.0, 0.5], dtype=np.float32)
    return np.array([1.0, 0.5], dtype=np.float32)


def _heuristic_greedy(env, agent_id: str) -> np.ndarray:
    """Always try to clear in P2P with full quantity at the most aggressive
    feasible price.

    Sellers : a_q=1, a_p=0 -> ask = min_cost  (cheapest feasible)
    Buyers  : a_q=1, a_p=1 -> bid = pi_max    (highest feasible)
    Neutral : a_q=0
    """
    idx = env.agent_name_to_idx[agent_id]
    role = env.role[idx]
    if role == 1:    # seller
        return np.array([1.0, 0.0], dtype=np.float32)
    if role == -1:   # buyer
        return np.array([1.0, 1.0], dtype=np.float32)
    return np.array([0.0, 0.5], dtype=np.float32)


HEURISTIC_FNS: Dict[str, Callable[[object, str], np.ndarray]] = {
    "grid_only": _heuristic_grid_only,
    "midpoint":  _heuristic_midpoint,
    "greedy":    _heuristic_greedy,
}


def heuristic_action(env, agent_id: str, name: str) -> np.ndarray:
    """Compute an action in [0,1]^2 for `agent_id` under heuristic `name`.

    Clips to the env's action space, just like deterministic_action does
    for RLModule outputs, so the two paths produce comparable inputs to
    env.step().
    """
    if name not in HEURISTIC_FNS:
        raise ValueError(f"Unknown heuristic: {name!r}. "
                         f"Choices: {HEURISTIC_CHOICES}")
    a = HEURISTIC_FNS[name](env, agent_id)

    space = env.action_spaces[agent_id]
    a = np.asarray(a, dtype=np.float32)
    a = np.clip(a, space.low, space.high)
    if hasattr(space, "shape") and space.shape is not None:
        expected_dim = int(np.prod(space.shape))
        a = a.reshape(-1)[:expected_dim].reshape(space.shape)
    return a


# =====================================================================
# ENV_CONFIG extraction (verbatim from evaluate_legacy.py)
# =====================================================================

def extract_env_config(train_py_path: Path) -> dict:
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


def load_env_class(experiment_dir: Path):
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
# Discovery helpers (verbatim from evaluate_legacy.py)
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

def load_rlmodules_selective(
    checkpoint_path: Path,
    env_class,
    env_config: dict,
    trained_policy_ids: Set[str],
) -> dict:
    """Load only the RLModules needed by trained agents.

    This is the only departure from evaluate_legacy.py: in mixed mode
    some policies will not be used at all (their agents are heuristic),
    so we skip loading them. If trained_policy_ids is empty, returns {}.
    """
    if not trained_policy_ids:
        print("[eval] No trained policies needed (all agents are heuristic).")
        return {}

    base = checkpoint_path / "learner_group" / "learner" / "rl_module"
    if not base.exists():
        raise FileNotFoundError(f"RLModule base path not found: {base}")

    rlmodules = {}
    for policy_id in sorted(trained_policy_ids):
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
# Heuristic-agents argument parsing
# =====================================================================

def parse_heuristic_agents(arg: str, n_agents: int) -> Set[int]:
    """Parse --heuristic-agents into a set of agent indices.

    Accepted formats:
        "all"            -> {0, 1, ..., n_agents-1}
        "none" / ""      -> set()
        "4"              -> {4}
        "0,2,5"          -> {0, 2, 5}
        "0-3"            -> {0, 1, 2, 3}
        "0,3-5"          -> {0, 3, 4, 5}
    Indices must be in [0, n_agents).
    """
    if arg is None:
        return set()
    s = arg.strip().lower()
    if s in ("", "none"):
        return set()
    if s == "all":
        return set(range(n_agents))

    out: Set[int] = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if lo_i > hi_i:
                raise ValueError(f"Invalid range '{token}': lo > hi.")
            for i in range(lo_i, hi_i + 1):
                out.add(i)
        else:
            out.add(int(token))

    bad = [i for i in out if not (0 <= i < n_agents)]
    if bad:
        raise ValueError(
            f"Heuristic agent indices out of range {sorted(bad)} "
            f"(valid: 0..{n_agents - 1})."
        )
    return out


def default_output_csv(heuristic: str, h_indices: Set[int]) -> str:
    """Generate a deterministic CSV name from the experimental knobs.

    eval_mixed_<heuristic>_h<sorted-indices>.csv

    Examples:
        midpoint, {4,5}        -> eval_mixed_midpoint_h45.csv
        grid_only, {0,1,2,3,4,5} -> eval_mixed_grid_only_h012345.csv
        greedy, set()          -> eval_mixed_greedy_hNONE.csv
    """
    if not h_indices:
        suffix = "NONE"
    else:
        suffix = "".join(str(i) for i in sorted(h_indices))
    return f"eval_mixed_{heuristic}_h{suffix}.csv"


# =====================================================================
# Per-checkpoint evaluation
# =====================================================================

def evaluate_one_checkpoint_mixed(
    checkpoint_path: Path,
    env_class,
    env_config: dict,
    num_episodes: int,
    output_csv_name: str,
    heuristic: str,
    h_agent_indices: Set[int],
) -> Path:
    print("\n" + "=" * 90)
    print(f"[eval] Checkpoint: {checkpoint_path}")
    print(f"[eval] Heuristic : {heuristic}")
    print(f"[eval] H-agents  : "
          f"{sorted(h_agent_indices) if h_agent_indices else '(none)'}")
    print("=" * 90)

    energy_policy_mapping_fn = policy_mode(env_config)
    env = env_class(env_config)

    # Resolve which agents are heuristic vs trained, and which RLModules
    # we actually need to load.
    h_agent_ids = {env.possible_agents[i] for i in h_agent_indices}
    trained_agent_ids = [aid for aid in env.possible_agents
                         if aid not in h_agent_ids]
    trained_policy_ids = {
        energy_policy_mapping_fn(aid, 0, None) for aid in trained_agent_ids
    }

    print(f"[eval] Trained agents  : {trained_agent_ids}")
    print(f"[eval] Heuristic agents: {sorted(h_agent_ids)}")
    print(f"[eval] Loading {len(trained_policy_ids)} RLModule(s): "
          f"{sorted(trained_policy_ids) if trained_policy_ids else '(none)'}")

    rlmodules = load_rlmodules_selective(
        checkpoint_path, env_class, env_config, trained_policy_ids,
    )

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
                if aid in h_agent_ids:
                    actions[aid] = heuristic_action(env, aid, heuristic)
                else:
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

            # Annotate which agents were heuristic in this run (constant
            # across rows but cheap to include and useful when concatenating
            # multiple CSVs in analysis notebooks).
            row["heuristic_name"] = heuristic
            row["heuristic_agents"] = json.dumps(sorted(h_agent_indices))

            for aid in env.possible_agents:
                row[f"{aid}_reward"] = (
                    float(rewards.get(aid, 0.0)) if isinstance(rewards, dict) else 0.0
                )
                row[f"{aid}_is_heuristic"] = int(aid in h_agent_ids)

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
        description="Mixed-policy evaluator: trained PPO agents play "
                    "against heuristic agents in the same episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 4 trained + 2 heuristic (midpoint on agent_4, agent_5)\n"
            "  python3 evaluate_mixed.py path/to/scenario \\\n"
            "      --heuristic midpoint --heuristic-agents 4,5\n\n"
            "  # 3 trained + 3 heuristic (greedy on agent_3..agent_5)\n"
            "  python3 evaluate_mixed.py path/to/scenario \\\n"
            "      --heuristic greedy --heuristic-agents 3-5\n\n"
            "  # Pure heuristic baseline (no PPO loaded)\n"
            "  python3 evaluate_mixed.py path/to/scenario \\\n"
            "      --heuristic grid_only --heuristic-agents all\n"
        ),
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to dir containing seed_dirs and train_ppo.py.",
    )
    parser.add_argument(
        "--heuristic", required=True, choices=HEURISTIC_CHOICES,
        help="Heuristic policy applied to the agents listed in "
             "--heuristic-agents.",
    )
    parser.add_argument(
        "--heuristic-agents", required=True,
        help="Comma-separated agent indices that use the heuristic "
             "(rest use trained PPO). Examples: '4,5', '0-3', 'all', "
             "'none'. Indices are 0-based into env.possible_agents.",
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
        "--output-csv", default=None,
        help="Output CSV filename, written next to the checkpoint. "
             "Default: eval_mixed_<heuristic>_h<idxs>.csv",
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

    # ---- Resolve heuristic-agents using a probe env to know n_agents ----
    probe_env = env_class(env_config)
    n_agents = len(probe_env.possible_agents)
    try:
        h_indices = parse_heuristic_agents(args.heuristic_agents, n_agents)
    except ValueError as e:
        sys.exit(f"Bad --heuristic-agents: {e}")

    if not h_indices:
        print("[eval] WARNING: --heuristic-agents resolves to empty set. "
              "This is identical to evaluate_legacy.py — continuing anyway.")

    # ---- Resolve output CSV name ----
    out_csv_name = (args.output_csv
                    if args.output_csv
                    else default_output_csv(args.heuristic, h_indices))
    print(f"[eval] Output CSV name: {out_csv_name}")

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
            existing = ckpt.parent / out_csv_name
            if existing.exists():
                print(f"\n[eval] SKIP {seed_dir.name}: "
                      f"{existing.name} already exists")
                skipped.append((seed_dir.name, str(existing)))
                continue

        try:
            out = evaluate_one_checkpoint_mixed(
                ckpt, env_class, env_config,
                args.num_episodes, out_csv_name,
                args.heuristic, h_indices,
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