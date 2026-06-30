"""
Evaluation script for HEURISTIC BASELINES (no training required).

Runs three non-learning baseline policies in the same environment used by
the trained MARL policies. Produces evaluation_agent_states.csv files that
are drop-in compatible with the existing analysis pipeline
(plot_hourly_energy.py, plot_cumulative_rewards, etc.).

Baselines
---------
  grid_only   No P2P participation. Every agent offers q=0, so all
              surplus/deficit settles with the grid. Lower-bound benchmark
              -- shows what the community looks like with zero cooperation.

  midpoint    "Fair-split" cooperative heuristic. Every agent offers their
              full available capacity at the midpoint of (pi_gb, pi_gs).
              Sellers respect their unit-cost floor:
                  p_sell = max(unit_cost(q=cap), midpoint).
              All bids >= all asks by construction => all P2P clears.
              Upper-bound benchmark on cooperative behavior.

  greedy      Self-interested extreme. Sellers ask at pi_max, buyers bid
              at pi_min. P2P typically collapses to zero (bid < ask).
              Useful to show what "no concession" looks like.

ENV_CONFIG and env class are recovered from the saved scenario folder the
same way evaluate_legacy.py does, so the heuristic plays in the exact same
environment the trained policies do.

Layout assumed (identical to evaluate_legacy.py):

  <experiment_dir>/
  |- train_ppo.py            <- source of ENV_CONFIG (AST-parsed)
  |- energy24h_env.py        <- saved env (optional; falls back to current)
  |- energy_market_training_seed42_run1/   <- existing trained seeds
  |- ...

After running, you get:

  <experiment_dir>/
  |- heuristic_grid_only/evaluation_agent_states.csv
  |- heuristic_midpoint/evaluation_agent_states.csv
  |- heuristic_greedy/evaluation_agent_states.csv

CLI
---
  python3 evaluate_heuristic.py path/to/scenario_dir
      -> runs all three baselines

  python3 evaluate_heuristic.py path/to/scenario_dir --baseline midpoint
      -> runs only the cooperative midpoint baseline

  python3 evaluate_heuristic.py path/to/scenario_dir --num-episodes 100
  python3 evaluate_heuristic.py path/to/scenario_dir --skip-if-exists
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reuse extractors from evaluate_legacy.py. Both files are expected to live
# at the project root and to be invoked from there.
from training.evaluate_legacy import (
    extract_env_config,
    load_env_class,
)


# =====================================================================
# Heuristic policies
# =====================================================================

BASELINES = ["grid_only", "midpoint", "greedy"]


def heuristic_action(env, agent_id: str, baseline: str) -> np.ndarray:
    """Compute the env action [a_q, a_p] in [0,1]^2 for the given baseline.

    Action semantics (mirrors P2PEnergyEnv.step):
        seller:  q = a_q * cap
                 p = min_cost + a_p * (pi_gs - min_cost)
                 where min_cost = max(unit_cost(q), pi_gb)
        buyer:   q = a_q * cap
                 p = pi_gb + a_p * (pi_gs - pi_gb)
        neutral: action ignored
    """
    idx = env.agent_name_to_idx[agent_id]
    role = int(env.role[idx])
    pi_gb = float(env.pi_gb)
    pi_gs = float(env.pi_gs)
    eps = float(env.eps)

    # Neutral agents: action is ignored by the env, return zeros.
    if role == 0:
        return np.zeros(2, dtype=np.float32)

    # -------------------------------------------------
    # grid_only: q = 0 disables P2P participation
    # -------------------------------------------------
    if baseline == "grid_only":
        return np.array([0.0, 0.5], dtype=np.float32)

    # -------------------------------------------------
    # midpoint: full cap at (pi_gb + pi_gs) / 2
    # -------------------------------------------------
    if baseline == "midpoint":
        midpoint = 0.5 * (pi_gb + pi_gs)

        if role == 1:
            # SELLER: respect unit-cost floor at q = cap.
            cap = float(env.cap[idx])
            prof = env.agents_all[idx]
            if cap > eps:
                cost = prof.a * cap ** 2 + prof.b * cap + prof.c
                min_cost = max(cost / cap, pi_gb)
            else:
                min_cost = pi_gb
            target_price = max(min_cost, midpoint)

            # Inverse-map target price into a_p:
            #   p = min_cost + a_p * (pi_gs - min_cost)
            denom = pi_gs - min_cost
            a_p = (target_price - min_cost) / denom if denom > eps else 0.0
            a_p = float(np.clip(a_p, 0.0, 1.0))
            return np.array([1.0, a_p], dtype=np.float32)

        # BUYER:
        denom = pi_gs - pi_gb
        a_p = (midpoint - pi_gb) / denom if denom > eps else 0.5
        a_p = float(np.clip(a_p, 0.0, 1.0))
        return np.array([1.0, a_p], dtype=np.float32)

    # -------------------------------------------------
    # greedy: sellers ask pi_max, buyers bid pi_min
    # -------------------------------------------------
    if baseline == "greedy":
        if role == 1:
            # Seller asks at pi_gs -> a_p = 1.0
            return np.array([1.0, 1.0], dtype=np.float32)
        # Buyer bids at pi_gb -> a_p = 0.0
        return np.array([1.0, 0.0], dtype=np.float32)

    raise ValueError(f"Unknown baseline: {baseline!r}")


# =====================================================================
# Per-baseline rollout (CSV schema identical to evaluate_legacy.py)
# =====================================================================

def evaluate_one_baseline(
    env_class,
    env_config: dict,
    baseline: str,
    output_dir: Path,
    num_episodes: int,
    output_csv_name: str = "evaluation_agent_states.csv",
    seed: Optional[int] = 42,
) -> Path:
    """Roll out one baseline policy in the env, log states to CSV.

    Output schema matches evaluate_legacy.py exactly so existing plotting
    code (plot_hourly_energy.py, etc.) Just Works on these CSVs.
    """
    print("\n" + "=" * 90)
    print(f"[heuristic] baseline = {baseline}")
    print("=" * 90)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_csv_name
    print(f"[heuristic] CSV -> {output_path}")

    env = env_class(env_config)
    rows = []

    for ep in range(num_episodes):
        # Per-episode seed makes the noise (if any) reproducible across
        # baselines: episode N gets the same demand/generation realization
        # whether the policy is grid_only, midpoint, greedy, or trained.
        ep_seed = None if seed is None else int(seed) + ep
        obs, _ = env.reset(seed=ep_seed)

        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        step = 0

        while not (terminateds.get("__all__", False)
                   or truncateds.get("__all__", False)):
            actions = {
                aid: heuristic_action(env, aid, baseline)
                for aid in obs
            }
            next_obs, _, terminateds, truncateds, infos = env.step(actions)

            # ---- build row (identical to evaluate_legacy.py) ----
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
                        for idx_v, value in enumerate(state_array):
                            row[f"{aid}_state_{idx_v}"] = float(value)

            rows.append(row)
            obs = next_obs
            step += 1

        print(f"[heuristic]   Ep {ep + 1}/{num_episodes} -> {step} steps")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[heuristic] Saved {len(df)} rows / {len(df.columns)} cols")
    return output_path


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic baseline policies in the saved env.",
    )
    parser.add_argument(
        "experiment_dir",
        help="Scenario dir with train_ppo.py "
             "(same layout evaluate_legacy.py expects).",
    )
    parser.add_argument(
        "--train-py", default=None,
        help="Override train_ppo.py path (default: <experiment_dir>/train_ppo.py).",
    )
    parser.add_argument(
        "--baseline", default="all", choices=BASELINES + ["all"],
        help="Which heuristic to run (default: all).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Episodes per baseline (default: 50).",
    )
    parser.add_argument(
        "--out-subdir-template", default="heuristic_{baseline}",
        help="Output subdir at scenario root, formatted with {baseline}.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base episode seed; episode N gets seed=base+N. Set to -1 to disable.",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip baselines whose output CSV already exists.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.is_dir():
        sys.exit(f"experiment_dir does not exist: {experiment_dir}")

    train_py = (Path(args.train_py).resolve()
                if args.train_py else experiment_dir / "train_ppo.py")
    if not train_py.is_file():
        sys.exit(f"train_ppo.py not found: {train_py}")

    # ---- Recover ENV_CONFIG from saved train_ppo.py ----
    env_config = extract_env_config(train_py)
    print(f"\n[heuristic] Extracted ENV_CONFIG from: {train_py}")
    print(json.dumps(env_config, indent=2, sort_keys=True))

    # ---- Load env class (saved one if present, else current project's) ----
    env_class = load_env_class(experiment_dir)

    baselines_to_run = BASELINES if args.baseline == "all" else [args.baseline]
    seed = None if args.seed < 0 else args.seed

    results = []
    for b in baselines_to_run:
        out_dir = experiment_dir / args.out_subdir_template.format(baseline=b)
        out_csv = out_dir / "evaluation_agent_states.csv"

        if args.skip_if_exists and out_csv.exists():
            print(f"\n[heuristic] SKIP {b}: {out_csv} already exists")
            results.append((b, "skipped", str(out_csv)))
            continue

        try:
            path = evaluate_one_baseline(
                env_class=env_class,
                env_config=env_config,
                baseline=b,
                output_dir=out_dir,
                num_episodes=args.num_episodes,
                seed=seed,
            )
            results.append((b, "ok", str(path)))
        except Exception as e:
            print(f"[heuristic] FAILED {b}: {e!r}")
            results.append((b, "failed", str(e)))

    print("\n" + "#" * 90)
    print(f"[heuristic] DONE. summary:")
    for b, status, info in results:
        print(f"  {status.upper():7s}  {b:12s} -> {info}")
    print("#" * 90)


if __name__ == "__main__":
    main()