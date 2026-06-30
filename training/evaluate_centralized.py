"""
evaluate_centralized.py

Centralized Social Optimum baseline for the P2P energy market.

For each (episode, hour), this baseline solves the per-hour LP:

    maximize  Σ_{ij} x_{ij}
    subject to
        Σ_j x_{ij} ≤ cap_seller_i     ∀ seller i
        Σ_i x_{ij} ≤ cap_buyer_j      ∀ buyer  j
        x_{ij} ≥ 0

This gives the WELFARE-MAXIMIZING P2P allocation under the same
demand/generation realizations seen by the trained MARL policies.

Why "max Σ x_{ij}" is the right objective here
----------------------------------------------
Sellers always dispatch their full surplus (cap_i = G_i − D_i) and buyers
always cover their full deficit (cap_j = D_j − G_j) — what changes is
whether each kWh goes through P2P or through the grid slack node.
Summing payoffs across agents, the bilateral P2P prices M_{ij} CANCEL
(seller revenue = buyer payment), and the social welfare reduces to:

    W = (λ_buy − λ_sell) · Σ x_{ij}  +  constants

Since λ_buy > λ_sell, the social optimum is exactly the allocation that
saturates total P2P volume up to min(Σ cap_seller, Σ cap_buyer).
Production costs c(disp) = a·disp² + b·disp + c are sunk (disp = cap
always) and likewise drop out of the welfare-maximizing decision.

Prices M_{ij} only redistribute surplus between sellers and buyers; they
don't affect total welfare. We assign a uniform clearing price under one
of three rules:

    'midpoint': (π_min + π_max) / 2  -- default, fair split of surplus
    'pi_min'  : π_min                 -- seller-favorable
    'pi_max'  : π_max                 -- buyer-favorable

Output CSV schema matches evaluate_heuristic.py / evaluate_legacy.py, so
the existing plot_hourly_energy.py pipeline reads it without changes.

Usage
-----
    python3 -m training.evaluate_centralized path/to/scenario_dir
    python3 -m training.evaluate_centralized path/to/scenario_dir --pricing-rule midpoint
    python3 -m training.evaluate_centralized path/to/scenario_dir --num-episodes 100
    python3 -m training.evaluate_centralized path/to/scenario_dir --skip-if-exists
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from training.evaluate_legacy import extract_env_config, load_env_class


# =====================================================================
# Centralized clearing solver
# =====================================================================

def solve_centralized_clearing(cap_s: np.ndarray, cap_b: np.ndarray) -> np.ndarray:
    """Proportional welfare-optimal allocation: x_ij = k_i * k_j / max(sum_s, sum_b).
    Saturates V* = min(sum_s, sum_b); the long side is curtailed uniformly."""
    cap_s = np.asarray(cap_s, dtype=np.float64)
    cap_b = np.asarray(cap_b, dtype=np.float64)
    if len(cap_s) == 0 or len(cap_b) == 0:
        return np.zeros((len(cap_s), len(cap_b)), dtype=np.float64)
    denom = max(cap_s.sum(), cap_b.sum())
    if denom <= 0:
        return np.zeros((len(cap_s), len(cap_b)), dtype=np.float64)
    return np.outer(cap_s, cap_b) / denom


def _northwest_corner(cap_s: np.ndarray, cap_b: np.ndarray) -> np.ndarray:
    """Greedy max-flow fallback. Achieves the same total volume as the LP."""
    n_s = int(len(cap_s))
    n_b = int(len(cap_b))
    x = np.zeros((n_s, n_b), dtype=np.float64)
    rem_s = np.asarray(cap_s, dtype=np.float64).copy()
    rem_b = np.asarray(cap_b, dtype=np.float64).copy()
    i, j = 0, 0
    while i < n_s and j < n_b:
        flow = float(min(rem_s[i], rem_b[j]))
        x[i, j] = flow
        rem_s[i] -= flow
        rem_b[j] -= flow
        if rem_s[i] <= 1e-12:
            i += 1
        else:
            j += 1
    return x


# =====================================================================
# Dynamic subclass: replace bilateral auction with centralized LP
# =====================================================================

def build_centralized_env_class(BaseEnvClass, pricing_rule: str = "midpoint"):
    """Return a subclass of BaseEnvClass whose `_clear_market_full()` is the
    centralized LP allocation, leaving every other env method untouched
    (noise sampling, sub-step structure, payoff formulas, info dict, obs).
    """

    def _clear_market_full(self) -> None:
        # Reset market matrices (mirrors base implementation).
        self.P.fill(0.0)
        self.M.fill(0.0)
        self.grid_import.fill(0.0)
        self.grid_export.fill(0.0)

        s_idx = list(self.current_seller_idx)
        b_idx = list(self.current_buyer_idx)

        # No P2P if either side is empty -> everyone settles with grid.
        if not s_idx or not b_idx:
            for gi in s_idx:
                self.grid_export[gi] = float(self.cap[gi])
            for gj in b_idx:
                self.grid_import[gj] = float(self.cap[gj])
            return

        cap_s = np.array([self.cap[i] for i in s_idx], dtype=np.float64)
        cap_b = np.array([self.cap[j] for j in b_idx], dtype=np.float64)

        x = solve_centralized_clearing(cap_s, cap_b)

        # Uniform clearing price across all transacting pairs.
        if pricing_rule == "midpoint":
            price = 0.5 * (float(self.pi_gb) + float(self.pi_gs))
        elif pricing_rule == "pi_min":
            price = float(self.pi_gb)
        elif pricing_rule == "pi_max":
            price = float(self.pi_gs)
        else:
            raise ValueError(f"Unknown pricing_rule: {pricing_rule!r}")

        # Populate P (quantities) and M (per-pair prices) from the LP solution.
        for li, gi in enumerate(s_idx):
            for lj, gj in enumerate(b_idx):
                qty = float(x[li, lj])
                if qty > self.eps:
                    self.P[gi, gj] = qty
                    self.M[gi, gj] = price

        # Slack-node settlement (identical to base class).
        for gj in b_idx:
            bought_p2p = float(np.sum(self.P[:, gj]))
            self.grid_import[gj] = max(float(self.cap[gj]) - bought_p2p, 0.0)
        for gi in s_idx:
            sold_p2p = float(np.sum(self.P[gi, :]))
            leftover = max(float(self.cap[gi]) - sold_p2p, 0.0)
            if leftover > self.eps:
                self.grid_export[gi] = leftover

    CentralizedEnvClass = type(
        "CentralizedP2PEnv",
        (BaseEnvClass,),
        {"_clear_market_full": _clear_market_full},
    )
    return CentralizedEnvClass


# =====================================================================
# Rollout (CSV schema identical to evaluate_heuristic.py / evaluate_legacy.py)
# =====================================================================

def evaluate_centralized(
    env_class,
    env_config: dict,
    output_dir: Path,
    num_episodes: int,
    pricing_rule: str,
    output_csv_name: str = "evaluation_agent_states.csv",
    seed: Optional[int] = 42,
) -> Path:
    print("\n" + "=" * 90)
    print(f"[centralized] LP-based social-optimum clearing | pricing = {pricing_rule}")
    print(f"[centralized] scipy.optimize available: {_HAS_SCIPY}")
    print("=" * 90)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_csv_name
    print(f"[centralized] CSV -> {output_path}")

    env = env_class(env_config)
    rows = []

    # Dummy action: full cap (a_q=1), midpoint (a_p=0.5).
    # The centralized clearing IGNORES self.q / self.p, so the action value is
    # irrelevant for market outcomes; we just need something env.step() accepts.
    dummy_action = np.array([1.0, 0.5], dtype=np.float32)

    for ep in range(num_episodes):
        # Per-episode seed: episode N gets seed=base+N. This is the SAME
        # convention evaluate_heuristic.py uses, so noise realizations match
        # 1-to-1 across baselines under the noise sweep.
        ep_seed = None if seed is None else int(seed) + ep
        obs, _ = env.reset(seed=ep_seed)

        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        step = 0

        while not (terminateds.get("__all__", False)
                   or truncateds.get("__all__", False)):
            actions = {aid: dummy_action.copy() for aid in obs}
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
                        for idx_v, value in enumerate(state_array):
                            row[f"{aid}_state_{idx_v}"] = float(value)

            rows.append(row)
            obs = next_obs
            step += 1

        print(f"[centralized]   Ep {ep + 1}/{num_episodes} -> {step} steps")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[centralized] Saved {len(df)} rows / {len(df.columns)} cols")
    return output_path


# =====================================================================
# Main
# =====================================================================

PRICING_RULES = ["midpoint", "pi_min", "pi_max"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Centralized Social Optimum baseline."
    )
    parser.add_argument(
        "experiment_dir",
        help="Scenario dir with train_ppo.py "
             "(same layout evaluate_legacy.py expects).",
    )
    parser.add_argument(
        "--train-py", default=None,
        help="Override train_ppo.py path "
             "(default: <experiment_dir>/train_ppo.py).",
    )
    parser.add_argument(
        "--pricing-rule", default="midpoint", choices=PRICING_RULES,
        help="How to price P2P trades in the optimum allocation "
             "(default: midpoint). Prices only redistribute surplus; total "
             "welfare is identical across rules.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=50,
        help="Episodes to evaluate (default: 50).",
    )
    parser.add_argument(
        "--out-subdir-template", default="centralized_{pricing_rule}",
        help="Output subdir at scenario root, formatted with {pricing_rule}.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base episode seed; episode N gets seed=base+N. "
             "Set to -1 to disable.",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip if output CSV already exists.",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.is_dir():
        sys.exit(f"experiment_dir does not exist: {experiment_dir}")

    train_py = (Path(args.train_py).resolve()
                if args.train_py else experiment_dir / "train_ppo.py")
    if not train_py.is_file():
        sys.exit(f"train_ppo.py not found: {train_py}")

    env_config = extract_env_config(train_py)
    print(f"\n[centralized] Extracted ENV_CONFIG from: {train_py}")
    print(json.dumps(env_config, indent=2, sort_keys=True))

    BaseEnvClass = load_env_class(experiment_dir)
    CentralizedEnvClass = build_centralized_env_class(
        BaseEnvClass, pricing_rule=args.pricing_rule
    )

    out_dir = experiment_dir / args.out_subdir_template.format(
        pricing_rule=args.pricing_rule
    )
    out_csv = out_dir / "evaluation_agent_states.csv"

    if args.skip_if_exists and out_csv.exists():
        print(f"\n[centralized] SKIP: {out_csv} already exists")
        sys.exit(0)

    seed = None if args.seed < 0 else args.seed

    try:
        path = evaluate_centralized(
            env_class=CentralizedEnvClass,
            env_config=env_config,
            output_dir=out_dir,
            num_episodes=args.num_episodes,
            pricing_rule=args.pricing_rule,
            seed=seed,
        )
        print("\n" + "#" * 90)
        print(f"[centralized] DONE -> {path}")
        print("#" * 90)
    except Exception as e:
        print(f"[centralized] FAILED: {e!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()