"""
Hourly community-energy plotting -- both a CLI script and an importable
module for notebook use.

All five quantities (generation, demand, P2P, import, export) are derived
directly from the evaluation CSV, which can now come from four sources:

  1. evaluate_legacy.py       -> evaluation_agent_states.csv  (default)
  2. evaluate_legacy_noise.py -> eval_noise_states_<tag>.csv
                                 (selected via noise_std=<float>)
  3. evaluate_heuristic.py    -> heuristic_<name>/evaluation_agent_states.csv
                                 (selected via heuristic="grid_only"
                                  | "midpoint" | "greedy")
  4. evaluate_mixed.py        -> eval_mixed_<heuristic>_h<idxs>.csv
                                 (selected via mixed_tag="<heuristic>_h<idxs>",
                                  e.g. "midpoint_h45", "grid_only_h012345")

Computations are identical in all three cases:

    Generation per hour =  Sum cap[i]   over agents where role_vec[i] == +1
    Demand per hour     =  Sum cap[i]   over agents where role_vec[i] == -1
    P2P / Import / Export per hour = mean of total_p2p / total_import /
                                     total_export across the 4 sub-steps
                                     within that hour.

Why MEAN over sub-steps instead of SUM: the steps_per_hour=4 sub-steps are
there so the agents can stabilise their actions inside one market hour,
not to represent four separate hourly trades. The hourly traded energy is
the stabilised value, so we average.

Selecting noisy vs clean evaluations
------------------------------------
Pass `noise_std=<float>` to load the noisy-evaluation CSV produced by
evaluate_legacy_noise.py for that noise level. Default is `None`, which
keeps the legacy behaviour and reads `evaluation_agent_states.csv`.

Selecting a heuristic baseline
------------------------------
Pass `heuristic="grid_only" | "midpoint" | "greedy"` to load the
corresponding heuristic CSV produced by evaluate_heuristic.py at
<scenario_dir>/heuristic_<name>/evaluation_agent_states.csv. The per-seed
glob is bypassed entirely and `noise_std` is ignored (heuristic CSVs
always use the legacy filename). Mutually exclusive with seed/run.

Selecting a mixed-policy evaluation
-----------------------------------
Pass `mixed_tag="<heuristic>_h<idxs>"` to load CSVs produced by
evaluate_mixed.py, where some agent slots are filled by a heuristic and
the rest by trained PPO policies. The tag is exactly what appears in
the CSV filename between `eval_mixed_` and `.csv`. Examples:

    mixed_tag="midpoint_h45"     -> eval_mixed_midpoint_h45.csv
    mixed_tag="grid_only_h5"     -> eval_mixed_grid_only_h5.csv
    mixed_tag="greedy_h012345"   -> eval_mixed_greedy_h012345.csv

The per-seed glob is used (just like noise_std), so multiple seeds are
discovered and averaged. Mutually exclusive with `noise_std` and
`heuristic` (they pick different CSVs).

Notebook usage
--------------

    from plot_hourly_energy import plot_hourly_energy

    # Legacy (clean) evaluation -- averaged across seeds:
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp")

    # Noisy evaluation, +/-10% gaussian:
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp",
                            noise_std=0.10)

    # One particular seed:
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp",
                            seed=42, noise_std=0.10)

    # Heuristic baseline (no seed concept):
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp",
                            heuristic="midpoint")

CLI usage
---------

    # Legacy CSV (avg over seeds):
    python3 plot_hourly_energy.py exp_results_repro/.../scenario

    # Noisy CSV for noise_std=0.10:
    python3 plot_hourly_energy.py exp_results_repro/.../scenario \
        --noise-std 0.10

    # Heuristic baseline:
    python3 plot_hourly_energy.py exp_results_repro/.../scenario \
        --heuristic midpoint

    # Mixed-policy (4 trained + 2 heuristic, midpoint on agent_4,5):
    python3 plot_hourly_energy.py exp_results_repro/.../scenario \
        --mixed-tag midpoint_h45
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Colors borrowed from the ISGT paper figure 3 for visual consistency.
COLORS = {
    "generation":  "#2ca02c",  # green
    "demand":      "#d62728",  # red
    "p2p":         "#1f77b4",  # blue
    "grid_import": "#ff7f0e",  # orange
    "grid_export": "#9467bd",  # purple
}


# Valid heuristic names (must match the subdir convention used by
# evaluate_heuristic.py: heuristic_<name>/).
HEURISTIC_CHOICES = ["grid_only", "midpoint", "greedy"]


# =====================================================================
# CSV name resolution
# =====================================================================

def _eval_csv_name(
    noise_std: Optional[float],
    mixed_tag: Optional[str] = None,
) -> str:
    """Resolve the evaluation CSV filename to read.

    noise_std=None, mixed_tag=None  -> 'evaluation_agent_states.csv'  (legacy)
    noise_std=<float>               -> 'eval_noise_states_<tag>.csv'  where
                                       tag = f"{noise_std:.2f}".replace(".", "p")
                                       (mirror of evaluate_legacy_noise.py).
    mixed_tag=<str>                 -> 'eval_mixed_<mixed_tag>.csv'
                                       (mirror of evaluate_mixed.py).

    noise_std and mixed_tag are mutually exclusive (they pick different files).
    """
    if noise_std is not None and mixed_tag is not None:
        raise ValueError(
            f"noise_std={noise_std!r} and mixed_tag={mixed_tag!r} are "
            f"mutually exclusive (they map to different CSV filenames)."
        )
    if mixed_tag is not None:
        return f"eval_mixed_{mixed_tag}.csv"
    if noise_std is None:
        return "evaluation_agent_states.csv"
    tag = f"{float(noise_std):.2f}".replace(".", "p")
    return f"eval_noise_states_{tag}.csv"


def _noise_label(noise_std: Optional[float]) -> str:
    """Short human-readable noise tag for titles / filenames."""
    if noise_std is None:
        return ""
    return f"noise{float(noise_std):.2f}".replace(".", "p")


def _mixed_label(mixed_tag: Optional[str]) -> str:
    """Short human-readable mixed-policy tag for titles / filenames."""
    if mixed_tag is None:
        return ""
    return f"mixed_{mixed_tag}"


# =====================================================================
# Discovery
# =====================================================================

def find_runs(
    scenario_dir: Union[str, Path],
    seed_glob: str = "energy_market_training_*seed*_run*",
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
) -> List[Tuple[str, Path]]:
    """Return list of (run_label, csv_path) for each evaluation CSV found.

    Pass `noise_std=<float>` to discover the noisy-evaluation CSV
    produced by evaluate_legacy_noise.py for that noise level.

    Pass `heuristic=<name>` (one of HEURISTIC_CHOICES) to discover the
    heuristic baseline CSV produced by evaluate_heuristic.py at
        <scenario_dir>/heuristic_<name>/evaluation_agent_states.csv
    In that case the per-seed glob is bypassed entirely and noise_std
    is ignored (the heuristic CSV always uses the legacy filename).
    seed/run are also ignored.

    Pass `mixed_tag=<str>` (e.g. "midpoint_h45") to discover the
    mixed-policy CSV produced by evaluate_mixed.py for that combination.
    The per-seed glob IS used (each seed has its own CSV). Mutually
    exclusive with `heuristic` and `noise_std`.
    """
    scenario_dir = Path(scenario_dir)

    # ---- Mutual-exclusion checks ----
    if heuristic is not None and mixed_tag is not None:
        raise ValueError(
            f"heuristic={heuristic!r} and mixed_tag={mixed_tag!r} are "
            f"mutually exclusive (different CSV layouts)."
        )

    # ---- Heuristic short-circuit ----
    if heuristic is not None:
        if heuristic not in HEURISTIC_CHOICES:
            raise ValueError(
                f"heuristic={heuristic!r} not in {HEURISTIC_CHOICES}"
            )
        if noise_std is not None:
            print(f"  WARN: noise_std={noise_std} ignored in heuristic mode "
                  f"(heuristic CSVs always use 'evaluation_agent_states.csv').")
        if seed is not None or run is not None:
            print(f"  WARN: seed/run ignored in heuristic mode.")

        csv = scenario_dir / f"heuristic_{heuristic}" / "evaluation_agent_states.csv"
        if csv.exists():
            return [(f"heuristic_{heuristic}", csv)]
        print(f"  WARN: missing {csv}")
        return []

    # ---- Original seed-glob behaviour (handles legacy / noise / mixed) ----
    csv_name = _eval_csv_name(noise_std, mixed_tag=mixed_tag)

    out = []
    for seed_dir in sorted(scenario_dir.glob(seed_glob)):
        if not seed_dir.is_dir():
            continue
        if seed is not None and f"seed{seed}_" not in seed_dir.name:
            continue
        if run is not None and not seed_dir.name.endswith(f"_run{run}"):
            continue

        csv = seed_dir / "PPO_energy_market_run" / csv_name
        if csv.exists():
            out.append((seed_dir.name, csv))
        else:
            print(f"  WARN: missing {csv}")
    return out

# =====================================================================
# Hourly clearing price (M matrix) & Generation Costs -- mean + variability band
#                                  + per-hour seller/buyer counts
# =====================================================================

def _parse_matrix_column(series: pd.Series) -> np.ndarray:
    """Parse a Series of JSON 2-D-array strings (e.g. '[[0,1],[2,3]]')
    into a 3-D numpy array of shape (n_rows, n_agents, n_agents)."""
    return np.array([json.loads(s) for s in series], dtype=np.float64)

# NOTA: Se asume que _parse_array_column ya existe en tu código original.
# def _parse_array_column(series: pd.Series) -> np.ndarray: ...

def _collect_trade_prices_and_roles(
    csv_path: Union[str, Path],
    cost_params: dict,
    eps: float = 1e-8,
) -> Tuple[dict, dict, dict, dict]:
    """Read one evaluation CSV, return four dicts:
        prices_by_hour  : {hour_index -> [trade prices...]}
        roles_by_hour   : {hour_index -> [(n_sellers, n_buyers), ...]}
        costs_by_hour   : {hour_index -> [cost_per_kw...]}
        volumes_by_hour : {hour_index -> [total_p2p_volume_per_row...]}
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in ("hour_index", "P", "M", "role_vec")
               if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    P_arr = _parse_matrix_column(df["P"])
    M_arr = _parse_matrix_column(df["M"])
    role_arr = _parse_array_column(df["role_vec"]).astype(np.int8)
    hours = df["hour_index"].to_numpy()

    # Extraer los estados (potencia generada) de cada agente
    n_agents = len(cost_params)
    states_0 = np.zeros((len(df), n_agents))
    for i in range(n_agents):
        col_name = f"agent_{i}_state_0"
        if col_name in df.columns:
            states_0[:, i] = df[col_name].to_numpy()

    prices_by_hour: dict = {}
    roles_by_hour: dict = {}
    costs_by_hour: dict = {}
    volumes_by_hour: dict = {}

    for i in range(len(df)):
        h = float(hours[i])
        
        # Prices: only cells that actually traded.
        mask = P_arr[i] > eps
        if mask.any():
            prices_by_hour.setdefault(h, []).extend(M_arr[i][mask].tolist())
            
        # Roles:
        n_s = int((role_arr[i] == 1).sum())
        n_b = int((role_arr[i] == -1).sum())
        roles_by_hour.setdefault(h, []).append((n_s, n_b))

        # Total P2P traded volume this row (sum of all matched quantities).
        # We always append, including rows with zero trade -- the 0 is a
        # meaningful value when computing the per-hour mean.
        total_p2p_vol = float(P_arr[i].sum())
        volumes_by_hour.setdefault(h, []).append(total_p2p_vol)

        # Generation Costs: only for sellers who actually CLOSED a P2P trade
        # in this row. A seller `a_idx` traded P2P iff some quantity was sold
        # to at least one buyer, i.e. P[a_idx, :].sum() > eps. Sellers whose
        # ask was rejected (no match -> grid export) do not contribute to the
        # market clearing price and are excluded from the mean.
        row_roles = role_arr[i]
        row_states = states_0[i]
        row_P = P_arr[i]   # (n_agents, n_agents) matrix for this row
        seller_costs = []
        for a_idx in range(n_agents):
            if row_roles[a_idx] != 1:
                continue   # not a seller
            if row_P[a_idx, :].sum() <= eps:
                continue   # seller did not close any P2P trade this row
            x = row_states[a_idx]
            if x <= eps:
                continue   # avoid div-by-zero on degenerate dispatch
            a, b, c = cost_params[a_idx]
            total_cost = a * (x ** 2) + b * x + c
            cost_per_kw = total_cost / x
            seller_costs.append(cost_per_kw)
        
        if seller_costs:
            costs_by_hour.setdefault(h, []).extend(seller_costs)

    return prices_by_hour, roles_by_hour, costs_by_hour, volumes_by_hour


def plot_hourly_clearing_price(
    scenario_dir: Union[str, Path],
    profiles_json_path: Union[str, Path],  # <--- NUEVO PARÁMETRO REQUERIDO
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    figsize: Tuple[float, float] = (12, 7),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    band: Optional[str] = "std",
    band_alpha: float = 0.20,
    color: str = "#1f77b4",
    show_summary: bool = False,
    show_roles: bool = True,
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    eps: float = 1e-8,
    # ---- Constrained Nash bargaining prediction overlay ----
    lambda_sell: float = 50.0,
    lambda_buy: float = 110.0,
    show_prediction: bool = True,
    show_volume_prediction: bool = True,
    show_residuals: bool = False,
    # ---- P2P traded volume on a secondary right-side axis ----
    show_p2p_volume: bool = True,
) -> pd.DataFrame:
    """
    Hourly mean clearing price and generation cost of the P2P market.

    Two theoretical predictions can be overlaid:

      show_prediction (green, cost-floor only):
          p_pred(h) = max( mean_cost(h),  midpoint )
          where midpoint = (lambda_sell + lambda_buy) / 2.

      show_volume_prediction (brown, cost-floor + volumetric α-Nash):
          p_pred_vol(h) = max( mean_cost(h),
                               (1 - α(h)) * lambda_sell + α(h) * lambda_buy )
          with α(h) = Q_B(h) / (Q_S(h) + Q_B(h)) computed from the JSON
          (Q_S = total seller surplus, Q_B = total buyer deficit per hour).

    The volume model captures hours where aggregate demand exceeds
    aggregate supply (sellers extracting more surplus). If the observed
    price tracks p_pred_vol better than p_pred in those hours, the policy
    is responding to volumetric demand pressure on top of the cost floor.

    If `show_residuals=True`, an inset box reports RMSE / MAE / bias for
    both predictions across active hours.
    """
    scenario_dir = Path(scenario_dir).resolve()
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"scenario_dir not found: {scenario_dir}")

    # ---- Cargar perfiles y parámetros de costo JSON ----
    profiles_json_path = Path(profiles_json_path).resolve()
    if not profiles_json_path.is_file():
        raise FileNotFoundError(f"profiles JSON not found: {profiles_json_path}")
        
    with open(profiles_json_path, 'r') as f:
        profiles_data = json.load(f)

    # Construir diccionario de costos (por defecto [0, 0, 0] si no existe)
    # Ajustado para 6 agentes (0 al 5)
    cost_params = {}
    for i in range(6): 
        agent_key = f"agent_{i}"
        if agent_key in profiles_data:
            cost_params[i] = profiles_data[agent_key].get("cost_params", [0.0, 0.0, 0.0])
        else:
            cost_params[i] = [0.0, 0.0, 0.0]

    # ---- Per-hour aggregate volumes for the α-Nash prediction ----
    # Q_S(h) = total seller surplus, Q_B(h) = total buyer deficit at hour h.
    # Computed across ALL agents in the JSON (not only the 6 indexed above),
    # so the proxy is faithful to whatever community size the JSON encodes.
    first_profile = next(iter(profiles_data.values()))
    num_profile_hours = len(first_profile["consumer_profile"])

    Q_S_per_hour = np.zeros(num_profile_hours, dtype=np.float64)
    Q_B_per_hour = np.zeros(num_profile_hours, dtype=np.float64)
    for _, p in profiles_data.items():
        G = np.asarray(p["generator_profile"], dtype=np.float64)
        D = np.asarray(p["consumer_profile"], dtype=np.float64)
        net = G - D
        Q_S_per_hour += np.maximum(net, 0.0)
        Q_B_per_hour += np.maximum(-net, 0.0)

    total_vol = Q_S_per_hour + Q_B_per_hour
    alpha_per_hour = np.where(
        total_vol > eps,
        Q_B_per_hour / np.maximum(total_vol, eps),
        0.5,   # default to midpoint behaviour when there's no activity
    )
    p_alpha_nash_per_hour = (
        (1.0 - alpha_per_hour) * lambda_sell + alpha_per_hour * lambda_buy
    )

    # Asume que la funcion find_runs ya existe en tu entorno
    runs = find_runs(
        scenario_dir, seed_glob=seed_glob,
        seed=seed, run=run, noise_std=noise_std,
        heuristic=heuristic, mixed_tag=mixed_tag,
    )
    if not runs:
        raise FileNotFoundError(f"No runs in {scenario_dir} ...")

    # ---- Pool prices, roles, costs, and volumes per hour ----
    prices_pool: dict = {}
    roles_pool: dict = {}
    costs_pool: dict = {}
    volumes_pool: dict = {}
    
    for label, csv in runs:
        try:
            by_hour_p, by_hour_r, by_hour_c, by_hour_v = _collect_trade_prices_and_roles(
                csv, cost_params=cost_params, eps=eps)
        except Exception as e:
            print(f"  WARN: skipping {label}: {e!r}")
            continue
            
        for h, prices in by_hour_p.items():
            prices_pool.setdefault(h, []).extend(prices)
        for h, role_counts in by_hour_r.items():
            roles_pool.setdefault(h, []).extend(role_counts)
        for h, costs in by_hour_c.items():
            costs_pool.setdefault(h, []).extend(costs)
        for h, volumes in by_hour_v.items():
            volumes_pool.setdefault(h, []).extend(volumes)

    if not roles_pool:
        raise RuntimeError("No rows found in any evaluation CSV.")

    all_hours = sorted(roles_pool.keys())
    h_min, h_max = int(min(all_hours)), int(max(all_hours))
    hour_grid = list(range(h_min, h_max + 1))

    records = []
    for h in hour_grid:
        prices = np.asarray(prices_pool.get(float(h), []), dtype=np.float64)
        costs = np.asarray(costs_pool.get(float(h), []), dtype=np.float64)
        volumes = np.asarray(volumes_pool.get(float(h), []), dtype=np.float64)
        rcs = roles_pool.get(float(h), [])
        
        n_sellers = float(np.mean([r[0] for r in rcs])) if rcs else 0.0
        n_buyers = float(np.mean([r[1] for r in rcs])) if rcs else 0.0

        # Pull volumetric stats for this hour from the precomputed arrays.
        if 0 <= h < num_profile_hours:
            Q_S_h = float(Q_S_per_hour[h])
            Q_B_h = float(Q_B_per_hour[h])
            alpha_h = float(alpha_per_hour[h])
            p_alpha_h = float(p_alpha_nash_per_hour[h])
        else:
            Q_S_h = Q_B_h = alpha_h = p_alpha_h = np.nan

        rec = {
            "hour_index": h,
            "mean_price": float(np.mean(prices)) if prices.size > 0 else np.nan,
            "std_price":  float(np.std(prices, ddof=0)) if prices.size > 0 else np.nan,
            "min_price":  float(np.min(prices)) if prices.size > 0 else np.nan,
            "max_price":  float(np.max(prices)) if prices.size > 0 else np.nan,
            "q25_price":  float(np.quantile(prices, 0.25)) if prices.size > 0 else np.nan,
            "q75_price":  float(np.quantile(prices, 0.75)) if prices.size > 0 else np.nan,
            "n_trades":   int(prices.size),
            "mean_cost_per_kw": float(np.mean(costs)) if costs.size > 0 else np.nan,
            "mean_p2p_volume":  float(np.mean(volumes)) if volumes.size > 0 else 0.0,
            "n_sellers": n_sellers,
            "n_buyers": n_buyers,
            "Q_S": Q_S_h,
            "Q_B": Q_B_h,
            "alpha": alpha_h,
            "p_alpha_nash": p_alpha_h,
        }
        records.append(rec)
        
    per_hour = pd.DataFrame(records).set_index("hour_index")

    # ---- Plot ----
    own_fig = ax is None
    use_role_subplot = own_fig and show_roles

    if use_role_subplot:
        fig, (ax, ax_roles) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    elif own_fig:
        fig, ax = plt.subplots(figsize=figsize)
        ax_roles = None
    else:
        fig = ax.figure
        ax_roles = None

    x = per_hour.index.to_numpy()
    mean_price = per_hour["mean_price"].to_numpy()
    mean_cost = per_hour["mean_cost_per_kw"].to_numpy()

    # Graficar Clearing Price
    ax.plot(
        x, mean_price,
        marker="o", color=color, linewidth=2.0,
        label="Mean clearing price", zorder=3,
    )

    # Graficar Costo Promedio de Generación por kW
    ax.plot(
        x, mean_cost,
        marker="s", color="#ff7f0e", linewidth=2.0, linestyle="--",
        label="Mean Gen Cost per kW (Sellers)", zorder=4,
    )

    # ---- Constrained Nash bargaining prediction overlay ----
    midpoint = 0.5 * (lambda_sell + lambda_buy)
    # Cost-floor only: predicted = max(cost, midpoint)
    # NaN-safe: where mean_cost is NaN (no active sellers), prediction = NaN.
    with np.errstate(invalid="ignore"):
        predicted_price = np.where(
            np.isnan(mean_cost),
            np.nan,
            np.maximum(mean_cost, midpoint),
        )
    per_hour["predicted_price"] = predicted_price

    # Cost-floor + volumetric α-Nash: predicted_vol = max(cost, α-Nash price)
    p_alpha_nash = per_hour["p_alpha_nash"].to_numpy()
    with np.errstate(invalid="ignore"):
        predicted_price_vol = np.where(
            np.isnan(mean_cost) | np.isnan(p_alpha_nash),
            np.nan,
            np.maximum(mean_cost, p_alpha_nash),
        )
    per_hour["predicted_price_vol"] = predicted_price_vol

    if show_prediction:
        ax.plot(
            x, predicted_price,
            marker="^", color="#2ca02c", linewidth=2.0,
            label=f"Predicted = max(cost, midpoint={midpoint:.0f})",
            zorder=5,
        )
        ax.axhline(
            midpoint, color="grey", linestyle=":", linewidth=1.2,
            alpha=0.7, label=f"Midpoint = {midpoint:.0f}", zorder=1,
        )
    if show_volume_prediction:
        ax.plot(
            x, predicted_price_vol,
            marker="D", color="#8c564b", linewidth=2.0, linestyle="-",
            label=r"Predicted = max(cost, $\alpha$-Nash by volume)",
            zorder=6,
        )

    if band == "std":
        std = per_hour["std_price"].to_numpy()
        lower, upper = mean_price - std, mean_price + std
        band_label = "Price +/- 1 std"
    elif band == "minmax":
        lower = per_hour["min_price"].to_numpy()
        upper = per_hour["max_price"].to_numpy()
        band_label = "Price [min, max]"
    elif band == "iqr":
        lower = per_hour["q25_price"].to_numpy()
        upper = per_hour["q75_price"].to_numpy()
        band_label = "Price IQR [Q25, Q75]"
    else:
        lower = upper = band_label = None

    if lower is not None:
        ax.fill_between(
            x, lower, upper,
            color=color, alpha=band_alpha, linewidth=0,
            label=band_label, zorder=2,
        )

    if title is None:
        title = "Mean clearing price & Generation cost over 24 h"
    if heuristic is not None:
        title = f"{title} (heuristic={heuristic})"
    elif noise_std is not None:
        title = f"{title}  ·  noise sigma={float(noise_std):.2f}"

    ax.set_title(title, pad=12)
    ax.set_ylabel("Price / Cost (per kW)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # ---- Secondary right-side axis: P2P traded volume ----
    if show_p2p_volume:
        ax_vol = ax.twinx()
        p2p_vol = per_hour["mean_p2p_volume"].to_numpy()
        vol_color = "#9467bd"   # purple, distinct from existing palette
        ax_vol.plot(
            x, p2p_vol,
            marker="P", color=vol_color, linewidth=1.8,
            alpha=0.85, label="P2P traded volume",
            zorder=2,
        )
        ax_vol.set_ylabel("P2P traded volume (kWh)", color=vol_color)
        ax_vol.tick_params(axis="y", labelcolor=vol_color)
        # Always start volume axis at 0 so the relative size is honest.
        vol_max = float(np.nanmax(p2p_vol)) if p2p_vol.size else 0.0
        ax_vol.set_ylim(0, max(vol_max * 1.10, 1e-6))

        # Combined legend: gather handles from both axes onto the main one.
        h_main, l_main = ax.get_legend_handles_labels()
        h_vol, l_vol = ax_vol.get_legend_handles_labels()
        ax.legend(h_main + h_vol, l_main + l_vol,
                  loc="best", framealpha=0.95)
    else:
        ax.legend(loc="best", framealpha=0.95)

    # ---- Bottom subplot: seller / buyer counts ----
    if ax_roles is not None:
        ns = per_hour["n_sellers"].to_numpy()
        nb = per_hour["n_buyers"].to_numpy()
        bar_w = 0.38
        
        # Asume que COLORS está definido globalmente en tu script
        ax_roles.bar(
            x - bar_w / 2, ns, bar_w,
            color="orange", label="Sellers", zorder=2, # Cambia "orange" por COLORS["generation"]
        )
        ax_roles.bar(
            x + bar_w / 2, nb, bar_w,
            color="blue", label="Buyers", zorder=2, # Cambia "blue" por COLORS["demand"]
        )
        ax_roles.set_ylabel("# agents")
        ax_roles.set_xlabel("Hour of day")
        ax_roles.grid(True, axis="y", alpha=0.3)
        ax_roles.set_axisbelow(True)
        ax_roles.set_xticks(x)
        ax_roles.set_xticklabels([f"{int(h)}" for h in x])
        ax_roles.legend(loc="upper right", ncol=2,
                        framealpha=0.95)
        y_max = int(np.ceil(max(ns.max(), nb.max())))
        ax_roles.set_yticks(range(0, y_max + 1))
        ax_roles.set_ylim(0, y_max * 1.30 + 0.3)
    else:
        ax.set_xlabel("Hour of day")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(h)}" for h in x])

    # ---- Residuals box: observed vs predicted (both models) ----
    if show_residuals and (show_prediction or show_volume_prediction):
        obs = mean_price

        def _stats(pred):
            valid = ~(np.isnan(obs) | np.isnan(pred))
            if valid.sum() == 0:
                return None
            r = obs[valid] - pred[valid]
            return (
                float(np.sqrt(np.mean(r ** 2))),  # RMSE
                float(np.mean(np.abs(r))),        # MAE
                float(np.mean(r)),                # bias
            )

        lines = ["observed - predicted"]
        if show_prediction:
            s = _stats(predicted_price)
            if s is not None:
                lines.append(f"  cost-floor:")
                lines.append(f"    RMSE = {s[0]:.2f}")
                lines.append(f"    MAE  = {s[1]:.2f}")
                lines.append(f"    bias = {s[2]:+.2f}")
        if show_volume_prediction:
            s = _stats(predicted_price_vol)
            if s is not None:
                lines.append(f"  vol α-Nash:")
                lines.append(f"    RMSE = {s[0]:.2f}")
                lines.append(f"    MAE  = {s[1]:.2f}")
                lines.append(f"    bias = {s[2]:+.2f}")
        if len(lines) > 1:
            ax.text(
                0.012, 0.97,
                "\n".join(lines),
                transform=ax.transAxes, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor="lightgrey", alpha=0.92),
                zorder=10,
            )

    if show_summary:
        n_total = int(per_hour["n_trades"].sum())
        n_active = int((per_hour["n_trades"] > 0).sum())
        global_mean = float(np.nanmean(mean_price))
        global_std = float(np.nanstd(mean_price, ddof=0))
        summary = (
            f"Active hours = {n_active}/{len(hour_grid)}   "
            f"Total trades = {n_total}   "
            f"Mean of hourly means = {global_mean:.1f}   "
            f"Std of hourly means = {global_std:.1f}"
        )
        if own_fig:
            fig.text(0.5, 0.005, summary, ha="center",
                    color="#333")
        else:
            ax.text(0.5, -0.18, summary, transform=ax.transAxes,
                    ha="center", va="top", color="#333")

    if own_fig:
        plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {save_path}")

    return per_hour


# =====================================================================
# Per-agent cost per kW over 24h
# =====================================================================

def plot_per_agent_cost_per_kw(
    profiles_json_path: Union[str, Path],
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    show_legend: bool = True,
    mode: str = "surplus",
    eps: float = 1e-8,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Plot per-agent generation cost per kW across the 24-hour horizon.

    Always creates its own standalone figure (no axis injection).

    For each agent uses the quadratic cost C(q) = a·q² + b·q + c to compute

        cost_per_kw(h) = C(q) / q = a·q + b + c/q

    where the quantity q depends on `mode`:

      - "surplus" (default): q = max(G(h) - D(h), 0).
        Defined only on seller-hours, when the agent has tradable surplus.
        Matches the cost floor used in the env's seller pricing rule.

      - "generation": q = G(h).
        Defined whenever G(h) > 0. A continuous view of production cost
        regardless of role.

    Hours with q = 0 are left as NaN (gaps in the line). Agents whose
    curve is entirely NaN are skipped in the plot but remain in the
    returned DataFrame.

    Returns DataFrame indexed by hour_index with one column per agent.
    """
    if mode not in ("surplus", "generation"):
        raise ValueError(f"mode must be 'surplus' or 'generation', got {mode!r}")

    with open(Path(profiles_json_path), "r") as f:
        profiles_data = json.load(f)
    if not profiles_data:
        raise ValueError(f"Empty profiles file: {profiles_json_path}")

    num_hours = len(next(iter(profiles_data.values()))["consumer_profile"])
    hours = np.arange(num_hours)

    per_agent: dict = {}
    for name in sorted(profiles_data.keys()):
        p = profiles_data[name]
        G = np.asarray(p["generator_profile"], dtype=np.float64)
        D = np.asarray(p["consumer_profile"], dtype=np.float64)
        a, b, c = (float(x) for x in p.get("cost_params", [0.0, 0.0, 0.0]))

        q = np.maximum(G - D, 0.0) if mode == "surplus" else G
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_cost = np.where(
                q > eps,
                a * q + b + c / np.maximum(q, eps),
                np.nan,
            )
        per_agent[name] = unit_cost

    df = pd.DataFrame(per_agent, index=pd.Index(hours, name="hour_index"))

    # ---- Plot (always a fresh standalone figure) ----
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(df.columns):
        values = df[name].values
        if np.all(np.isnan(values)):
            continue   # never a seller / never generates -> skip in plot
        ax.plot(
            df.index, values,
            marker="o", markersize=4, linewidth=1.5,
            color=cmap(i % 10),
            label=name,
        )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Cost per kW")
    if title is None:
        mode_label = "surplus dispatch" if mode == "surplus" else "total generation"
        title = f"Per-agent cost per kW over 24 h ({mode_label})"
    ax.set_title(title)
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)

    if show_legend:
        n_visible = sum(not np.all(np.isnan(df[c].values)) for c in df.columns)
        ncol = 2 if n_visible > 4 else 1
        ax.legend(loc="best", framealpha=0.95, ncol=ncol)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {save_path}")

    return df


# =====================================================================
# CSV aggregation
# =====================================================================

def _parse_array_column(series: pd.Series) -> np.ndarray:
    """Parse a Series of JSON-array strings (e.g. '[1.5, 2.0, 0.0]') into a
    2-D numpy array of shape (n_rows, n_agents)."""
    return np.array([json.loads(s) for s in series], dtype=np.float64)


def aggregate_csv_per_seed(
    csv_path: Union[str, Path],
    seed_label: str,
) -> pd.DataFrame:
    """Reduce one evaluation CSV to per-hour means.

    Steps:
      1. Decode `cap` and `role_vec` (JSON arrays) into per-row totals:
           generation = Sum cap[i] for sellers (role_vec[i] == +1)
           demand     = Sum cap[i] for buyers  (role_vec[i] == -1)
         These are constant within an hour (set in _set_hour, not modified
         by sub-steps).
      2. MEAN across the 4 sub-steps within each (episode, hour_index).
         For generation/demand this is a no-op (constant); for
         total_p2p / total_import / total_export it gives the stabilised
         hourly value.
      3. MEAN across episodes for this seed.

    Returns DataFrame with columns: seed, hour_index,
                                    generation, demand, p2p, imp, exp.
    """
    df = pd.read_csv(csv_path)

    required = ["episode", "hour_index",
                "total_p2p", "total_import", "total_export",
                "cap", "role_vec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    cap_arr = _parse_array_column(df["cap"])
    role_arr = _parse_array_column(df["role_vec"]).astype(np.int8)

    df = df.copy()
    df["generation"] = (cap_arr * (role_arr == 1)).sum(axis=1)
    df["demand"] = (cap_arr * (role_arr == -1)).sum(axis=1)

    # Mean across sub-steps within (episode, hour_index).
    per_ep_hour = df.groupby(["episode", "hour_index"], as_index=False).agg(
        generation=("generation", "mean"),
        demand=("demand", "mean"),
        p2p=("total_p2p", "mean"),
        imp=("total_import", "mean"),
        exp=("total_export", "mean"),
    )
    # Mean across episodes for this seed.
    per_hour = per_ep_hour.groupby("hour_index", as_index=False).agg(
        generation=("generation", "mean"),
        demand=("demand", "mean"),
        p2p=("p2p", "mean"),
        imp=("imp", "mean"),
        exp=("exp", "mean"),
    )
    per_hour["seed"] = seed_label
    return per_hour


def gini_coefficient(values: np.ndarray) -> float:
    """
    Standard Gini coefficient. Negative values are handled by shifting the
    distribution so its minimum is zero (preserves relative spread).
    Returns 0 for a perfectly equal allocation, approaching 1 for maximal
    inequality.
    """
    x = np.asarray(values, dtype=np.float64).flatten()
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        x = x - x.min()
    s = x.sum()
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x_sorted) / (n * s)) - (n + 1.0) / n)


def jain_index(values: np.ndarray, shift_negatives: bool = True) -> float:
    """
    Jain's fairness index: 1 = perfectly equal, 1/N = one agent takes all.

    With negatives in the input, sum(x)^2 in the numerator can collapse
    misleadingly. By default we shift the distribution to be non-negative
    first (same convention used elsewhere here for Gini).
    """
    x = np.asarray(values, dtype=np.float64).flatten()
    n = x.size
    if n == 0:
        return 0.0
    if shift_negatives and x.min() < 0:
        x = x - x.min()
    denom = n * np.sum(x ** 2)
    if denom <= 0:
        return 1.0
    return float((x.sum() ** 2) / denom)


def loss_share(values: np.ndarray) -> dict:
    """
    Two complementary loss diagnostics on a per-agent payoff vector.

        loss_count:  fraction of agents with strictly negative payoff in [0, 1].
        loss_volume: total losses / total gains.
                     0.0 when no agent loses; -> 1.0 when losses match gains;
                     > 1.0 if losses exceed gains; inf if losses but zero gains.
    """
    x = np.asarray(values, dtype=np.float64).flatten()
    if x.size == 0:
        return {"loss_count": 0.0, "loss_volume": 0.0}

    losses = np.maximum(0.0, -x).sum()
    gains  = np.maximum(0.0, x).sum()

    if losses == 0.0:
        loss_volume = 0.0
    elif gains == 0.0:
        loss_volume = float("inf")
    else:
        loss_volume = float(losses / gains)

    return {
        "loss_count": float(np.mean(x < 0.0)),
        "loss_volume": loss_volume,
    }


def plot_cumulative_rewards(
    scenario_dir: Union[str, Path],
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    figsize: Tuple[float, float] = (10, 6),
    ylim: Tuple[float, float] = (0, 800),
    title: str = "Cumulative reward per agent",
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
):
    """
    Daily cumulative reward per agent, averaged across episodes and seeds.

    Pipeline:
      1. Average the 4 sub-steps inside each hour_index.
      2. Sum hourly averages over the 24 hours -> daily total per episode.
      3. Average daily totals across episodes (per seed), then across seeds.

    Equity diagnostics shown:
      Total  - sum of the per-agent rewards plotted (community daily payoff).
      Jain   - fairness index in [1/N, 1]; 1 = perfectly equal split.
      LossN  - fraction of agents ending the day with negative payoff.
      LossV  - total losses / total gains across agents.
      Gini   - inequality index (with shift to handle negatives).

    Pass `noise_std=<float>` to read the noisy-evaluation CSV for that
    noise level instead of the legacy clean one.

    Pass `heuristic=<name>` to read the heuristic baseline CSV from
    <scenario_dir>/heuristic_<name>/ instead of the trained seed runs.
    seed/run/noise_std are ignored in that case.

    Pass `mixed_tag=<str>` (e.g. "midpoint_h45") to read the mixed-policy
    CSVs produced by evaluate_mixed.py. Per-seed aggregation still
    applies. Mutually exclusive with `heuristic` and `noise_std`.

    Per-seed values are printed to stdout when seed=None and there is
    more than one run (i.e. not in heuristic mode).
    """
    runs = find_runs(
        scenario_dir, seed_glob=seed_glob,
        seed=seed, run=run, noise_std=noise_std,
        heuristic=heuristic, mixed_tag=mixed_tag,
    )
    if not runs:
        raise FileNotFoundError(
            f"No se encontraron ejecuciones en {scenario_dir} "
            f"(noise_std={noise_std!r}, heuristic={heuristic!r}, "
            f"mixed_tag={mixed_tag!r})."
        )

    all_runs_rewards = []
    per_seed_metrics: List[Tuple[str, dict]] = []

    for label, csv_path in runs:
        df = pd.read_csv(csv_path)

        reward_cols = [c for c in df.columns
                       if c.startswith("agent_") and c.endswith("_reward")]
        reward_cols = sorted(reward_cols, key=lambda x: int(x.split('_')[1]))

        # 1 & 2: hourly mean over sub-steps, then sum over the 24 hours.
        per_hour_mean = df.groupby(["episode", "hour_index"])[reward_cols].mean()
        daily_cumulative = per_hour_mean.groupby("episode").sum()

        # 3a: mean across episodes of this seed.
        mean_rewards_this_seed = daily_cumulative.mean()
        all_runs_rewards.append(mean_rewards_this_seed)

        # All metrics on this seed's per-agent vector.
        v = mean_rewards_this_seed.values
        ls = loss_share(v)
        per_seed_metrics.append((label, {
            "total": float(np.sum(v)),
            "gini":  gini_coefficient(v),
            "jain":  jain_index(v),
            "lossN": ls["loss_count"],
            "lossV": ls["loss_volume"],
        }))

    # 3b: mean across seeds (no-op in heuristic mode, single run).
    final_rewards = pd.concat(all_runs_rewards, axis=1).mean(axis=1)
    agent_labels = [c.replace("_reward", "").replace("_", " ")
                    for c in final_rewards.index]

    # Aggregate metrics on the cross-seed mean vector.
    v_agg = final_rewards.values
    agg_gini  = gini_coefficient(v_agg)
    agg_jain  = jain_index(v_agg)
    agg_loss  = loss_share(v_agg)
    agg_lossN = agg_loss["loss_count"]
    agg_lossV = agg_loss["loss_volume"]
    agg_total = float(np.sum(v_agg))   # sum of the plotted bars

    # ---------- Print per-seed metrics when aggregating ----------
    # Skip in heuristic mode (only one "run", nothing to compare).
    if heuristic is None and seed is None and len(per_seed_metrics) > 1:
        noise_suffix = (f"  (noise_std={noise_std})"
                        if noise_std is not None else "")
        print(f"\n[plot_cumulative_rewards] {Path(scenario_dir).name}"
              f"{noise_suffix}")
        print(f"  Aggregate (cross-seed mean vector):")
        print(f"    Total = {agg_total:.2f}    Jain  = {agg_jain:.4f}    "
              f"LossN = {agg_lossN:.2%}    LossV = {agg_lossV:.4f}    "
              f"Gini = {agg_gini:.4f}")
        print(f"  Per-seed:")
        header = (f"    {'seed/run':<35s}  {'Total':>8s}  {'Jain':>7s}  "
                  f"{'LossN':>7s}  {'LossV':>7s}  {'Gini':>7s}")
        print(header)
        print(f"    {'-'*35}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for lab, m in sorted(per_seed_metrics, key=lambda t: t[0]):
            lv_str = "inf" if np.isinf(m["lossV"]) else f"{m['lossV']:.4f}"
            print(f"    {lab:<35s}  {m['total']:>8.2f}  {m['jain']:>7.4f}  "
                  f"{m['lossN']:>6.2%}  {lv_str:>7s}  {m['gini']:>7.4f}")
        # mean / std across seeds for each metric
        for key, fmt in [("total", "{:.2f}"), ("jain", "{:.4f}"),
                         ("lossN", "{:.2%}"), ("lossV", "{:.4f}"),
                         ("gini", "{:.4f}")]:
            arr = np.array([m[key] for _, m in per_seed_metrics
                            if not np.isinf(m[key])], dtype=float)
            if arr.size:
                print(f"    {key:>5s}:  mu = {fmt.format(arr.mean())}   "
                      f"sigma = {fmt.format(arr.std(ddof=0))}")

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=figsize)

    ref_colors = ["#0077b6", "#ff8c32", "#00a844", "#e3242b", "#9b6ab3", "#8c564b"]
    colors = [ref_colors[i % len(ref_colors)] for i in range(len(final_rewards))]

    bars = ax.bar(agent_labels, final_rewards.values,
                  color=colors, edgecolor='lightgrey', zorder=3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    # Annotate title with mode info.
    if heuristic is not None:
        title = f"{title} (heuristic={heuristic})"
    elif mixed_tag is not None:
        title = f"{title}  ·  mixed={mixed_tag}"
    elif noise_std is not None:
        title = f"{title}  ·  noise sigma={float(noise_std):.2f}"

    ax.set_title(title, pad=15)
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("Agent")

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(ylim)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # Equity diagnostics inset (top-right). Total is the sum of the
    # plotted bars; Jain and LossN as primary; LossV and Gini as context.
    lossV_str = "inf" if np.isinf(agg_lossV) else f"{agg_lossV:.3f}"
    inset_text = (
        f"Total  = {agg_total:.1f}\n"
        f"Jain   = {agg_jain:.3f}\n"
        f"LossN  = {agg_lossN:.1%}\n"
        f"LossV  = {lossV_str}\n"
        f"Gini   = {agg_gini:.3f}"
    )
    ax.text(
        0.97, 0.95, inset_text,
        transform=ax.transAxes,
        ha="right", va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="lightgrey", alpha=0.9),
    )

    plt.tight_layout()
    return fig, ax, agg_jain

def _rank_seeds(
    per_seed_metrics: List[Tuple[str, dict]],
    weights: Optional[dict] = None,
    loss_threshold: float = 0.0,
) -> Tuple[Optional[str], pd.DataFrame, bool]:
    """
    Rankea semillas: filtro duro (lossN, lossV == 0) + score ponderado.
    Retorna (mejor_label, ranking_df, all_failed).
    """
    if weights is None:
        weights = {"total": 0.30, "jain": 0.50, "gini": 0.20}

    rows = []
    for label, m in per_seed_metrics:
        rows.append({
            "seed":   label,
            "total":  m["total"],
            "jain":   m["jain"],
            "gini":   m["gini"],
            "lossN":  m["lossN"],
            "lossV":  m["lossV"] if not np.isinf(m["lossV"]) else np.nan,
            "passed": m["lossN"] <= loss_threshold and m["lossV"] <= loss_threshold,
        })
    df = pd.DataFrame(rows)

    all_failed = not df["passed"].any()
    pool = df if all_failed else df[df["passed"]].copy()
    pool_idx = pool.index

    def _norm(col, invert=False):
        lo, hi = pool[col].min(), pool[col].max()
        if hi == lo:
            return pd.Series(1.0, index=pool_idx)
        n = (pool[col] - lo) / (hi - lo)
        return 1.0 - n if invert else n

    score = (
        weights["total"] * _norm("total") +
        weights["jain"]  * _norm("jain") +
        weights["gini"]  * _norm("gini", invert=True)
    )

    df["score"] = np.nan
    df.loc[pool_idx, "score"] = score.values
    df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)

    best = df.loc[df["score"].idxmax(), "seed"] if not df["score"].isna().all() else None
    return best, df, all_failed


def plot_cumulative_rewards_multi(
    scenario_dirs: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    seeds: Optional[List[Optional[Union[str, int]]]] = None,
    run: Optional[Union[str, int]] = None,
    figsize: Tuple[float, float] = (12, 6),
    ylim: Optional[Tuple[float, float]] = None,
    title: str = "Cumulative reward per agent",
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
    bar_width: float = 0.25,
    show_inset: bool = True,
    weights: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, List[dict]]:
    """
    Daily cumulative reward per agent comparando múltiples experimentos.
    Una barra por experimento por agente. Imprime ranking de semillas por
    experimento y retorna (fig, ax, all_metrics).
    """
    exp_colors = ["#0077b6", "#e3242b", "#00a844", "#ff8c32", "#9b6ab3", "#8c564b"]

    all_final_rewards: List[pd.Series] = []
    all_metrics: List[dict] = []
    exp_labels: List[str] = []

    for i, scenario_dir in enumerate(scenario_dirs):
        lbl = (labels[i] if labels is not None and i < len(labels)
               else Path(scenario_dir).name)
        exp_labels.append(lbl)

        seed_i = (seeds[i] if seeds is not None and i < len(seeds) else None)

        runs = find_runs(
            scenario_dir, seed_glob=seed_glob,
            seed=seed_i, run=run, noise_std=noise_std,
            heuristic=heuristic, mixed_tag=mixed_tag,
        )
        if not runs:
            raise FileNotFoundError(
                f"No se encontraron ejecuciones en {scenario_dir} "
                f"(noise_std={noise_std!r}, heuristic={heuristic!r}, "
                f"mixed_tag={mixed_tag!r})."
            )

        per_seed_rewards: List[pd.Series] = []
        per_seed_metrics: List[Tuple[str, dict]] = []

        for run_label, csv_path in runs:
            df = pd.read_csv(csv_path)
            reward_cols = sorted(
                [c for c in df.columns
                 if c.startswith("agent_") and c.endswith("_reward")],
                key=lambda x: int(x.split('_')[1])
            )
            per_hour_mean = df.groupby(["episode", "hour_index"])[reward_cols].mean()
            daily_cumulative = per_hour_mean.groupby("episode").sum()
            mean_this_seed = daily_cumulative.mean()
            per_seed_rewards.append(mean_this_seed)

            v = mean_this_seed.values
            ls = loss_share(v)
            per_seed_metrics.append((run_label, {
                "total": float(np.sum(v)),
                "gini":  gini_coefficient(v),
                "jain":  jain_index(v),
                "lossN": ls["loss_count"],
                "lossV": ls["loss_volume"],
            }))

        final_rewards = pd.concat(per_seed_rewards, axis=1).mean(axis=1)
        all_final_rewards.append(final_rewards)

        # Métricas agregadas cross-seed
        v_agg = final_rewards.values
        ls_agg = loss_share(v_agg)
        lossV_val = ls_agg["loss_volume"]
        all_metrics.append({
            "label": lbl,
            "total": float(np.sum(v_agg)),
            "gini":  gini_coefficient(v_agg),
            "jain":  jain_index(v_agg),
            "lossN": ls_agg["loss_count"],
            "lossV": lossV_val,
        })

        # ── Ranking de semillas ───────────────────────────────────────────
        best_seed, rank_df, all_failed = _rank_seeds(per_seed_metrics, weights=weights)

        # Tabla: acortar label a seed{N}_run{N}
        import re as _re
        def _short(s):
            m = _re.search(r"(seed\d+_run\d+)", s)
            return m.group(1) if m else s

        rank_display = rank_df[["seed", "total", "jain", "gini", "lossN", "score", "passed"]].copy()
        rank_display["seed"]   = rank_display["seed"].map(_short)
        rank_display["total"]  = rank_display["total"].map("{:>8.1f}".format)
        rank_display["jain"]   = rank_display["jain"].map("{:.4f}".format)
        rank_display["gini"]   = rank_display["gini"].map("{:.4f}".format)
        rank_display["lossN"]  = rank_display["lossN"].map("{:.0%}".format)
        rank_display["score"]  = rank_display["score"].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else "—"
        )
        rank_display["passed"] = rank_display["passed"].map(lambda x: "✓" if x else "✗")

        print(f"\n[{lbl}]")
        print(rank_display.to_string(index=False))
        if all_failed:
            print("  ⚠  Ninguna semilla pasó el filtro — ranking sobre pool completo")
        if best_seed is not None:
            best_score = rank_df.loc[rank_df["seed"] == best_seed, "score"].values[0]
            print(f"  → mejor: {_short(best_seed)}  (score={best_score:.4f})")
        else:
            print("  → mejor: N/A")

    # ── Posiciones de barras ──────────────────────────────────────────────
    agent_labels = [
        c.replace("_reward", "").replace("_", " ")
        for c in all_final_rewards[0].index
    ]
    n_agents = len(agent_labels)
    n_exps   = len(scenario_dirs)

    x = np.arange(n_agents, dtype=float)
    total_group_width = bar_width * n_exps
    offsets = np.linspace(
        -total_group_width / 2 + bar_width / 2,
         total_group_width / 2 - bar_width / 2,
        n_exps,
    )

    # ── Figura ────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, (final_rewards, color, lbl) in enumerate(
        zip(all_final_rewards, exp_colors, exp_labels)
    ):
        bars = ax.bar(
            x + offsets[i], final_rewards.values,
            width=bar_width,
            color=color,
            edgecolor="lightgrey",
            label=lbl,
            zorder=3,
        )
        for bar in bars:
            height = bar.get_height()
            va    = "bottom" if height >= 0 else "top"
            ytext = 3        if height >= 0 else -3
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, ytext),
                textcoords="offset points",
                ha="center", va=va,
                color=color,
            )

    suffix = ""
    if heuristic is not None:
        suffix = f" (heuristic={heuristic})"
    elif mixed_tag is not None:
        suffix = f"  ·  mixed={mixed_tag}"
    elif noise_std is not None:
        suffix = f"  ·  noise σ={float(noise_std):.2f}"

    ax.set_title(f"{title}{suffix}", pad=15)
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_labels)
    ax.yaxis.grid(True, linestyle="-", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(loc="upper left", framealpha=0.9)

    # ── Inset métricas agregadas (sin lossN) ──────────────────────────────
    if show_inset:
        lines = []
        for m in all_metrics:
            lv = "inf" if np.isinf(m["lossV"]) else f"{m['lossV']:.3f}"
            lines.append(
                f"{m['label'][:8]:<8s}  "
                f"Pf.={m['total']:>7.1f}  "
                f"Jain={m['jain']:.3f}  "
                f"Gini={m['gini']:.3f}"
            )
        ax.text(
            0.99, 0.97, "\n".join(lines),
            transform=ax.transAxes,
            ha="right", va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", edgecolor="lightgrey", alpha=0.9),
        )

    if ax is None:
        plt.tight_layout()
    return fig, ax, all_metrics

def plot_scenario_metrics_box(
    scenario_dirs: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    seeds: Optional[List[Optional[Union[str, int]]]] = None,
    run: Optional[Union[str, int]] = None,
    figsize: Tuple[float, float] = (12, 4),
    title: Optional[str] = None,
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
    ylim_total: Optional[Tuple[float, float]] = None,
    ylim_jain:  Optional[Tuple[float, float]] = None,
    ylim_gini:  Optional[Tuple[float, float]] = None,
    box_color: str = "#0077b6",
    show_points: bool = True,
    ax: Optional[np.ndarray] = None,
    show_xlabel: bool = True,
    n_figures: Optional[float] = 3
) -> Tuple[plt.Figure, np.ndarray, List[dict]]:
    """
    Figura comparativa de escenarios con 3 subplots: Total | Jain | Gini.
    Cada caja representa la distribución entre semillas de un escenario.
    Opcionalmente superpone los puntos individuales por semilla.

    Retorna (fig, axes[3], all_metrics) donde all_metrics contiene los
    vectores por semilla además de los agregados.
    """
    # ── 1. Calcular métricas por semilla para cada escenario ──────────────
    all_metrics: List[dict] = []

    for i, scenario_dir in enumerate(scenario_dirs):
        lbl = (labels[i] if labels is not None and i < len(labels)
               else Path(scenario_dir).name)
        seed_i = (seeds[i] if seeds is not None and i < len(seeds) else None)

        runs = find_runs(
            scenario_dir, seed_glob=seed_glob,
            seed=seed_i, run=run, noise_std=noise_std,
            heuristic=heuristic, mixed_tag=mixed_tag,
        )
        if not runs:
            raise FileNotFoundError(
                f"No se encontraron ejecuciones en {scenario_dir} "
                f"(noise_std={noise_std!r}, heuristic={heuristic!r}, "
                f"mixed_tag={mixed_tag!r})."
            )

        # Una entrada por semilla
        seed_totals, seed_jains, seed_ginis = [], [], []

        for _, csv_path in runs:
            df = pd.read_csv(csv_path)
            reward_cols = sorted(
                [c for c in df.columns
                 if c.startswith("agent_") and c.endswith("_reward")],
                key=lambda x: int(x.split('_')[1])
            )
            per_hour_mean = df.groupby(["episode", "hour_index"])[reward_cols].mean()
            daily_cumulative = per_hour_mean.groupby("episode").sum()
            v = daily_cumulative.mean().values  # vector por agente, esta semilla

            seed_totals.append(float(np.sum(v)))
            seed_jains.append(jain_index(v))
            seed_ginis.append(gini_coefficient(v))

        all_metrics.append({
            "label":  lbl,
            "totals": seed_totals,   # lista de N_seeds valores
            # "jains":  seed_jains,
            "ginis":  seed_ginis,
        })

    # ── 2. Figura ─────────────────────────────────────────────────────────
    exp_labels = [m["label"] for m in all_metrics]
    x = np.arange(len(exp_labels))

    if ax is None:
        fig, ax = plt.subplots(1, n_figures, figsize=figsize)
    else:
        fig = ax[0].figure

    specs = [
        (ax[0], "totals", "Payoff", ylim_total),
        (ax[1], "ginis",  "Gini Coeff.",  ylim_gini),
        # (ax[2], "jains",  "Jain Index",   ylim_jain),
        
    ]

    # Desempacar color base en RGB para calcular versión más clara del fill
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(box_color)
    face_color  = (r, g, b, 0.25)   # relleno translúcido
    median_color = box_color

    for ax, key, ylabel, ylim in specs:
        data = [m[key] for m in all_metrics]   # lista de listas

        bp = ax.boxplot(
            data,
            positions=x,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color=median_color, linewidth=2),
            boxprops=dict(facecolor=face_color, edgecolor=box_color, linewidth=1.2),
            whiskerprops=dict(color=box_color, linewidth=1.2, linestyle="-"),
            capprops=dict(color=box_color, linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor="white",
                            markeredgecolor=box_color, markersize=5,
                            linewidth=0.8),
            zorder=3,
        )

        if show_points:
            rng = np.random.default_rng(0)
            for xi, vals in zip(x, data):
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    xi + jitter, vals,
                    color=box_color, alpha=0.7, s=18, zorder=4,
                )

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        if show_xlabel:
            ax.set_xticklabels(exp_labels, rotation=0)
        else:
            ax.set_xticklabels([])
        ax.yaxis.grid(True, linestyle="-", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        if ylim is not None:
            ax.set_ylim(ylim)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, ax, all_metrics

def aggregate_runs(
    runs: List[Tuple[str, Path]],
) -> Tuple[pd.DataFrame, int, str]:
    """Aggregate one or many runs into a per-hour DataFrame.

    Returns (per_hour_df_indexed_by_hour, n_runs, mode_label).
    """
    per_seed_dfs = []
    for label, csv in runs:
        try:
            per_seed_dfs.append(aggregate_csv_per_seed(csv, label))
        except Exception as e:
            print(f"  WARN: skipping {label}: {e!r}")

    if not per_seed_dfs:
        raise RuntimeError("No usable runs after aggregation.")

    if len(per_seed_dfs) == 1:
        run_label = per_seed_dfs[0]["seed"].iloc[0]
        per_hour = (per_seed_dfs[0]
                    .drop(columns=["seed"])
                    .set_index("hour_index"))
        return per_hour, 1, run_label

    all_df = pd.concat(per_seed_dfs, ignore_index=True)
    per_hour = all_df.groupby("hour_index", as_index=True).agg(
        generation=("generation", "mean"),
        demand=("demand", "mean"),
        p2p=("p2p", "mean"),
        imp=("imp", "mean"),
        exp=("exp", "mean"),
    )
    return per_hour, len(per_seed_dfs), f"avg over {len(per_seed_dfs)} seeds"


# =====================================================================
# Plotting
# =====================================================================

def _draw_bars(
    per_hour_df: pd.DataFrame,
    title: str,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (15, 6),
    show_summary: bool = True,
):
    """Grouped bar chart, 5 series per hour. Returns (fig, ax)."""
    hours = per_hour_df.index.values.astype(int)
    G_per_h = per_hour_df["generation"].values
    D_per_h = per_hour_df["demand"].values
    p2p = per_hour_df["p2p"].values
    imp = per_hour_df["imp"].values
    exp = per_hour_df["exp"].values

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    width = 0.16
    x = np.arange(len(hours), dtype=float)

    ax.bar(x - 2 * width, G_per_h, width,
           label="Generation", color=COLORS["generation"])
    ax.bar(x - 1 * width, D_per_h, width,
           label="Demand", color=COLORS["demand"])
    ax.bar(x + 0 * width, p2p, width,
           label="P2P trade", color=COLORS["p2p"])
    ax.bar(x + 1 * width, imp, width,
           label="Grid import", color=COLORS["grid_import"])
    ax.bar(x + 2 * width, exp, width,
           label="Grid export", color=COLORS["grid_export"])

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}" for h in hours])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=5, framealpha=0.95)

    if show_summary:
        total_p2p_day = float(p2p.sum())
        total_imp_day = float(imp.sum())
        total_exp_day = float(exp.sum())
        total_D_day = float(D_per_h.sum())
        total_G_day = float(G_per_h.sum())

        p2p_share = (100.0 * total_p2p_day / total_D_day) if total_D_day > 1e-9 else 0.0
        imp_share = (100.0 * total_imp_day / total_D_day) if total_D_day > 1e-9 else 0.0
        exp_share = (100.0 * total_exp_day / total_G_day) if total_G_day > 1e-9 else 0.0

        summary = (
            f"Daily totals (kWh):  G={total_G_day:.1f}   D={total_D_day:.1f}   "
            f"P2P={total_p2p_day:.1f} ({p2p_share:.1f}% of D)   "
            f"Imp={total_imp_day:.1f} ({imp_share:.1f}% of D)   "
            f"Exp={total_exp_day:.1f} ({exp_share:.1f}% of G)"
        )
        if own_fig:
            fig.text(0.5, 0.005, summary, ha="center", color="#333")
        else:
            ax.text(0.5, -0.18, summary, transform=ax.transAxes,
                    ha="center", va="top", color="#333")

    if own_fig:
        plt.tight_layout(rect=[0, 0.03, 1, 1])

    return fig, ax


# =====================================================================
# Top-level convenience function (notebook-friendly)
# =====================================================================

def plot_hourly_energy(
    scenario_dir: Union[str, Path],
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    figsize: Tuple[float, float] = (15, 6),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_summary: bool = True,
    seed_glob: str = "energy_market_training_seed*_run*",
    verbose: bool = True,
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Discover -> aggregate -> plot, in one call.

    Parameters
    ----------
    noise_std : float, optional
        If set, reads the noisy-evaluation CSV produced by
        evaluate_legacy_noise.py for that noise level
        (`eval_noise_states_<tag>.csv`) instead of the clean
        legacy CSV (`evaluation_agent_states.csv`).
    heuristic : {"grid_only", "midpoint", "greedy"}, optional
        If set, reads the heuristic baseline CSV produced by
        evaluate_heuristic.py at
        `<scenario_dir>/heuristic_<name>/evaluation_agent_states.csv`
        instead of trained seed runs. Mutually exclusive with seed/run;
        also ignores `noise_std`.
    mixed_tag : str, optional
        If set, reads the mixed-policy CSV produced by evaluate_mixed.py
        for that tag (e.g. "midpoint_h45" reads
        `eval_mixed_midpoint_h45.csv`). Per-seed aggregation still
        applies. Mutually exclusive with `heuristic` and `noise_std`.

    Returns
    -------
    DataFrame with columns
        ['generation', 'demand', 'p2p_trade', 'grid_import', 'grid_export']
    indexed by hour_index.
    """
    scenario_dir = Path(scenario_dir).resolve()
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"scenario_dir not found: {scenario_dir}")

    runs = find_runs(
        scenario_dir, seed_glob=seed_glob,
        seed=seed, run=run, noise_std=noise_std,
        heuristic=heuristic, mixed_tag=mixed_tag,
    )
    if not runs:
        if heuristic is not None:
            expected_loc = f"heuristic_{heuristic}/evaluation_agent_states.csv"
        else:
            expected_loc = _eval_csv_name(noise_std, mixed_tag=mixed_tag)
        raise FileNotFoundError(
            f"No runs in {scenario_dir} "
            f"(seed={seed!r}, run={run!r}, noise_std={noise_std!r}, "
            f"heuristic={heuristic!r}, mixed_tag={mixed_tag!r}). "
            f"Expected file: {expected_loc}"
        )
    if verbose:
        if heuristic is not None:
            print(f"Heuristic mode (baseline={heuristic}) "
                  f"in {scenario_dir.name}:")
        else:
            csv_name = _eval_csv_name(noise_std, mixed_tag=mixed_tag)
            print(f"Found {len(runs)} run(s) in {scenario_dir.name} "
                  f"(CSV={csv_name}):")
        for label, csv in runs:
            print(f"  - {label}")

    per_hour, n_runs, mode_label = aggregate_runs(runs)

    if title is None:
        if heuristic is not None:
            title = (f"Hourly community energy · {scenario_dir.name} · "
                     f"heuristic={heuristic}")
        else:
            title = f"Hourly community energy · {scenario_dir.name} · {mode_label}"
            if mixed_tag is not None:
                title = f"{title} · mixed={mixed_tag}"
            elif noise_std is not None:
                title = f"{title} · noise σ={float(noise_std):.2f}"

    fig, _ = _draw_bars(per_hour, title,
                       ax=ax, figsize=figsize, show_summary=show_summary)

    if save_path and ax is None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"\nSaved figure to: {save_path}")

    out_df = per_hour.rename(columns={
        "p2p": "p2p_trade",
        "imp": "grid_import",
        "exp": "grid_export",
    })
    return out_df[["generation", "demand", "p2p_trade", "grid_import", "grid_export"]]


def plot_hourly_energy_overlay(
    scenario_dir,
    seed=None,
    run=None,
    title="Community energy metrics by hour",
    figsize=(11, 6),
    ax=None,
    alpha=0.7,
    bar_widths=(0.72, 0.56, 0.42, 0.28, 0.15),
    edge_lw=1.0,
    seed_glob="energy_market_training_*seed*_run*",
    verbose=False,
    noise_std=None,
    heuristic=None,
    mixed_tag=None,
):
    runs = find_runs(
        scenario_dir, seed_glob=seed_glob,
        seed=seed, run=run, noise_std=noise_std,
        heuristic=heuristic, mixed_tag=mixed_tag,
    )
    if not runs:
        if heuristic is not None:
            expected_loc = f"heuristic_{heuristic}/evaluation_agent_states.csv"
        else:
            expected_loc = _eval_csv_name(noise_std, mixed_tag=mixed_tag)
        raise FileNotFoundError(
            f"No runs in {scenario_dir} "
            f"(seed={seed!r}, run={run!r}, noise_std={noise_std!r}, "
            f"heuristic={heuristic!r}, mixed_tag={mixed_tag!r}). "
            f"Expected file: {expected_loc}"
        )
    if verbose:
        if heuristic is not None:
            print(f"Heuristic mode (baseline={heuristic}) "
                  f"in {Path(scenario_dir).name}")
        else:
            print(f"Found {len(runs)} run(s). "
                  f"CSV={_eval_csv_name(noise_std, mixed_tag=mixed_tag)}")

    per_hour, _, _ = aggregate_runs(runs)
    hours = per_hour.index.values.astype(int)

    # ── Métricas de resumen ──────────────────────────────────────────────
    total_demand  = per_hour["demand"].sum()
    total_p2p     = per_hour["p2p"].sum()
    total_gen     = per_hour["generation"].sum()
    total_imp     = per_hour["imp"].sum()
    total_exp     = per_hour["exp"].sum()

    p2p_demand_pct = 100.0 * total_p2p / np.minimum(per_hour["demand"], per_hour["generation"]).sum()

    series = [
        ("Generation",  per_hour["generation"].values, "#2ca02c"),
        ("Demand",      per_hour["demand"].values,     "#d62728"),
        ("P2P trade",   per_hour["p2p"].values,        "#1f77b4"),
        ("Grid import", per_hour["imp"].values,        "#ff7f0e"),
        ("Grid export", per_hour["exp"].values,        "#9467bd"),
    ]

    if len(bar_widths) != len(series):
        raise ValueError(f"bar_widths must have {len(series)} entries.")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(hours))
    for (label, vals, color), w in zip(series, bar_widths):
        ax.bar(
            x, vals, width=w, label=label,
            color=color, alpha=alpha,
            edgecolor=color, linewidth=edge_lw,
        )

    # Append mode tag to the title.
    if heuristic is not None:
        title = f"{title} · heuristic={heuristic}"
    elif mixed_tag is not None:
        title = f"{title} · mixed={mixed_tag}"
    elif noise_std is not None:
        title = f"{title} · noise σ={float(noise_std):.2f}"

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hours])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.margins(x=0.01)

    # ── Cuadro de resumen (esquina superior izquierda) ───────────────────
    stats_line = (
        f"P2P traded ({p2p_demand_pct:.1f}%): {total_p2p:.1f} kWh   ·   "
        f"G: {total_gen:.1f} kWh\n"
        f"D: {total_demand:.1f} kWh   ·   "
        f"I: {total_imp:.1f} kWh   ·   "
        f"E: {total_exp:.1f} kWh"
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy (kWh)")
    
    # Ponemos el título principal y le damos más margen superior (pad=25)
    ax.set_title(title, pad=55)
    
    # Colocamos el subtítulo justo debajo del título principal
    ax.text(
        0.5, 1.02,  # Y=1.02 está justo encima del borde superior de la gráfica
        stats_line,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#333333",  # Un gris oscuro para que sea más sutil
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hours])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.margins(x=0.01)

    if own_fig:
        fig.tight_layout()
    else:
        fig = ax.figure

    return per_hour

def plot_daily_energy_summary(
    scenario_dirs: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    seeds: Optional[List[Optional[Union[str, int]]]] = None,
    run: Optional[Union[str, int]] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Daily community energy summary",
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    heuristic: Optional[str] = None,
    mixed_tag: Optional[str] = None,
    alpha: float = 0.65,
    bar_widths: Tuple[float, ...] = (0.55, 0.44, 0.33, 0.22, 0.11),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, List[dict]]:
    """
    Una barra overlay por escenario con los totales diarios agregados.
    Replica el estilo de plot_hourly_energy_overlay pero colapsando las
    24 horas en un único valor por escenario.

    Retorna (fig, ax, all_stats).
    """
    series_cfg = [
        ("Generation",  "generation", "#2ca02c"),
        ("Demand",      "demand",     "#d62728"),
        ("P2P trade",   "p2p",        "#1f77b4"),
        ("Grid import", "imp",        "#ff7f0e"),
        ("Grid export", "exp",        "#9467bd"),
    ]

    if len(bar_widths) != len(series_cfg):
        raise ValueError(f"bar_widths debe tener {len(series_cfg)} entradas.")

    # ── 1. Cargar y agregar totales por escenario ─────────────────────────
    all_stats: List[dict] = []

    for i, scenario_dir in enumerate(scenario_dirs):
        lbl = (labels[i] if labels is not None and i < len(labels)
               else Path(scenario_dir).name)
        seed_i = (seeds[i] if seeds is not None and i < len(seeds) else None)

        runs = find_runs(
            scenario_dir, seed_glob=seed_glob,
            seed=seed_i, run=run, noise_std=noise_std,
            heuristic=heuristic, mixed_tag=mixed_tag,
        )
        if not runs:
            raise FileNotFoundError(
                f"No se encontraron ejecuciones en {scenario_dir} "
                f"(noise_std={noise_std!r}, heuristic={heuristic!r})."
            )

        per_hour, _, _ = aggregate_runs(runs)

        total_gen    = per_hour["generation"].sum()
        total_demand = per_hour["demand"].sum()
        total_p2p    = per_hour["p2p"].sum()
        total_imp    = per_hour["imp"].sum()
        total_exp    = per_hour["exp"].sum()

        # p2p_pct = 100.0 * total_p2p / np.minimum(
        #     per_hour["demand"], per_hour["generation"]
        # ).sum()


        p2p_pct = 100.0 * total_p2p / total_demand
        imp_pct = 100.0 * total_imp / total_demand

        all_stats.append({
            "label":      lbl,
            "generation": total_gen,
            "demand":     total_demand,
            "p2p":        total_p2p,
            "imp":        total_imp,
            "exp":        total_exp,
            "p2p_pct":    p2p_pct,
            "imp_pct":    imp_pct,
        })

    # ── 2. Figura ─────────────────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(all_stats))

    for (label, key, color), w in zip(series_cfg, bar_widths):
        vals = [s[key] for s in all_stats]
        ax.bar(
            x, vals, width=w,
            label=label, color=color,
            alpha=alpha, edgecolor=color,
            zorder=3,
        )

    # ── Anotaciones debajo de cada barra ──────────────────────────────────
    y_min = ax.get_ylim()[0]
    for xi, s in enumerate(all_stats):
        ax.text(
            xi, -0.08,
            f"P2P: {s['p2p_pct']:.1f}%\nImp: {s['imp_pct']:.1f}%",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            color="#333333",
        )

    # ── Decoración ────────────────────────────────────────────────────────
    suffix = ""
    if heuristic is not None:
        suffix = f" · heuristic={heuristic}"
    elif mixed_tag is not None:
        suffix = f" · mixed={mixed_tag}"
    elif noise_std is not None:
        suffix = f" · noise σ={float(noise_std):.2f}"

    ax.set_title(f"{title}{suffix}")
    ax.set_ylabel("Energy (kWh)")
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in all_stats], rotation=0)
    ax.yaxis.grid(True, linestyle="-", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.1)
    ax.legend(loc="upper right", framealpha=0.95)

    if own_fig:
        fig.tight_layout()

    return fig, ax, all_stats

# =====================================================================
# Training curve discovery
# =====================================================================

def find_training_csvs(
    scenario_dir: Union[str, Path],
    seed_glob: str = "energy_market_training_*seed*_run*",  # mismo default que find_runs
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    progress_filename: str = "progress.csv",
) -> List[Tuple[str, Path]]:
    """Return [(run_label, progress_csv_path)] para cada seed encontrada.

    Espeja find_runs() pero apunta al progress.csv de entrenamiento en
    lugar del CSV de evaluación.

    Estructura esperada:
        <scenario_dir>/
          energy_market_training_*seed{S}_run{R}/
            PPO_energy_market_run/
              progress.csv
    """
    scenario_dir = Path(scenario_dir)
    out = []
    for seed_dir in sorted(scenario_dir.glob(seed_glob)):
        if not seed_dir.is_dir():
            continue
        if seed is not None and f"seed{seed}_" not in seed_dir.name:
            continue
        if run is not None and not seed_dir.name.endswith(f"_run{run}"):
            continue
        csv = seed_dir / "PPO_energy_market_run" / progress_filename
        if csv.exists():
            out.append((seed_dir.name, csv))
        else:
            print(f"  WARN: missing {csv}")
    return out


# =====================================================================
# Training reward comparison
# =====================================================================

def plot_rewards_comparison(
    # --- Fuente de datos ---
    scenario_dir: Union[str, Path, List[Union[str, Path]], None] = None,
    seed_glob: str = "energy_market_training_*seed*_run*",
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    # --- O bien, datos pre-cargados (dict label -> [paths]) ---
    data: Optional[Union[List[str], dict]] = None,
    # --- Columnas del CSV ---
    iter_col: str = "training_iteration",
    mean_col: str = "env_runners/episode_return_mean",
    min_col: str = "env_runners/episode_return_min",
    max_col: str = "env_runners/episode_return_max",
    # --- Apariencia ---
    title: str = "Training episode return",
    xlabel: str = "Training iteration",
    ylabel: str = "Episode return",
    figsize: Tuple[float, float] = (8, 4.5),
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    linewidth: float = 2.0,
    # --- Estilo de incertidumbre ---
    band_style: str = "fill",          # "fill" (banda) | "errorbar" (barras)
    error_metric: str = "seed_std",    # "seed_std" (entre semillas) | "minmax"
    errorevery: int = 5,               # espaciado de las barras (modo errorbar)
    capsize: float = 3.0,
    linestyle: str = "-",
    band_alpha: float = 0.20,
    seed_aggregation: str = "mean",    # "mean" | "envelope"
    show_individual_traces: bool = False,
    individual_alpha: float = 0.15,
    # --- Modo individual (múltiples experimentos etiquetados) ---
    labels: Optional[List[str]] = None,
    highlight_seeds: Optional[List[int]] = None,
    highlight_colors: Optional[List[str]] = None,
    muted_color: str = "lightgray",
) -> Tuple[plt.Figure, plt.Axes]:
    """Curvas de entrenamiento con incertidumbre entre semillas.

    `band_style` controla cómo se muestra la dispersión:
      * "fill"     -> banda sombreada (mean + min/max), comportamiento previo.
      * "errorbar" -> barras de error periódicas sobre la curva media, al
                      estilo de las figuras con whiskers. La fuente del error
                      la fija `error_metric`:
                        - "seed_std": desviación estándar ENTRE semillas del
                          episode_return_mean en cada iteración (recomendado
                          para el análisis multi-seed);
                        - "minmax": semi-rango medio min/max de cada corrida.

    Tres formas de uso (igual que antes):

    1. Un solo escenario:
       >>> fig, ax = plot_rewards_comparison("exp_results/.../scenario")
    2. Lista de escenarios:
       >>> fig, ax = plot_rewards_comparison([NORM_LOCAL, NORM_PARTIAL, NORM_TOTAL])
    3. Dict con etiquetas custom:
       >>> fig, ax = plot_rewards_comparison(data={"Base": [...], "Shuffle": [...]})
    """
    # ── 1. Resolver fuente de datos ──────────────────────────────────────
    if scenario_dir is not None and data is not None:
        raise ValueError("Usa scenario_dir O data, no ambos.")

    if scenario_dir is not None:
        if isinstance(scenario_dir, list):
            dirs = scenario_dir
        else:
            dirs = [scenario_dir]

        experiments = {}
        for d in dirs:
            runs = find_training_csvs(d, seed_glob=seed_glob, seed=seed, run=run)
            if not runs:
                raise FileNotFoundError(
                    f"No se encontró progress.csv en {d} "
                    f"(seed={seed!r}, run={run!r})."
                )
            experiments[Path(d).name] = [str(p) for _, p in runs]

    else:
        if data is None:
            raise ValueError("Debes pasar scenario_dir o data.")
        experiments = {"Experiment": data} if isinstance(data, list) else data

    # Aplicar etiquetas custom si se pasaron
    if labels is not None:
        keys = list(experiments.keys())
        experiments = {
            labels[i] if i < len(labels) else keys[i]: experiments[keys[i]]
            for i in range(len(keys))
        }

    # ── 2. Configurar figura ─────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if colors is None:
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(experiments))]

    # ── 3. Un experimento = una curva con incertidumbre ──────────────────
    for ci, ((label, paths), color) in enumerate(zip(experiments.items(), colors)):
        series_mean, series_min, series_max = {}, {}, {}

        for k, p in enumerate(paths):
            df = pd.read_csv(p)
            idx = (df[iter_col].astype(int).values
                   if iter_col in df.columns else np.arange(1, len(df) + 1))
            key = f"s{k}"
            series_mean[key] = pd.Series(df[mean_col].values, index=idx)
            if min_col in df.columns and max_col in df.columns:
                series_min[key] = pd.Series(df[min_col].values, index=idx)
                series_max[key] = pd.Series(df[max_col].values, index=idx)

        means_wide = pd.DataFrame(series_mean).sort_index()
        mid = means_wide.mean(axis=1, skipna=True)
        x = mid.index.values

        # Resaltado de seeds específicas (igual que antes, tiene prioridad)
        if highlight_seeds is not None:
            import re as _re
            hl_set = set(highlight_seeds)
            hc_map = dict(zip(highlight_seeds, highlight_colors)) if highlight_colors else {}
            muted_done = False
            for k, p in enumerate(paths):
                m = _re.search(r"seed(\d+)", str(p))
                s_id = int(m.group(1)) if m else k
                if s_id not in hl_set:
                    ax.plot(means_wide.index, means_wide[f"s{k}"],
                            color=muted_color, linewidth=2, alpha=1.0, zorder=1,
                            label=None if muted_done else "other seeds")
                    muted_done = True
                else:
                    c = hc_map.get(s_id, color_cycle[len(hc_map) % len(color_cycle)])
                    lbl = f"Seed {s_id}"
                    ax.plot(means_wide.index, means_wide[f"s{k}"],
                            color=c, linewidth=3.5, alpha=1.0, zorder=10, label=lbl)
            continue

        # Trazos individuales tenues (opcional)
        if show_individual_traces:
            for col in means_wide.columns:
                ax.plot(means_wide.index, means_wide[col],
                        color=color, alpha=individual_alpha, linewidth=1, zorder=1)

        if band_style == "errorbar":
            # Fuente del error: dispersión entre semillas (default) o min/max.
            if error_metric == "seed_std":
                err = means_wide.std(axis=1, ddof=1).fillna(0.0).values
            else:  # "minmax" -> semi-rango medio como error simétrico
                if series_min and series_max:
                    lo = pd.DataFrame(series_min).sort_index().mean(axis=1)
                    hi = pd.DataFrame(series_max).sort_index().mean(axis=1)
                    err = ((hi - lo) / 2.0).values
                else:
                    err = means_wide.std(axis=1, ddof=1).fillna(0.0).values
            # Offset por curva para que las barras no se solapen entre escenarios.
            offset = (ci * max(1, errorevery // max(1, len(experiments)))) % errorevery
            ax.errorbar(
                x, mid.values, yerr=err,
                errorevery=(offset, errorevery),
                color=color, linewidth=linewidth, linestyle=linestyle,
                capsize=capsize, capthick=1.0, elinewidth=1.0,
                label=label, zorder=3,
            )
        else:
            # Banda sombreada (comportamiento previo).
            if series_min and series_max:
                mins_w = pd.DataFrame(series_min).sort_index()
                maxs_w = pd.DataFrame(series_max).sort_index()
                if seed_aggregation == "envelope":
                    lower, upper = mins_w.min(axis=1), maxs_w.max(axis=1)
                else:  # "mean"
                    lower, upper = mins_w.mean(axis=1), maxs_w.mean(axis=1)
                ax.fill_between(x, lower.values, upper.values,
                                color=color, alpha=band_alpha, linewidth=0, zorder=2)
            ax.plot(x, mid.values, color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label, zorder=3)

    # ── 4. Formato final ─────────────────────────────────────────────────
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.margins(x=0)
    ax.legend(loc="best", fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
# =====================================================================
# CLI entry
# =====================================================================

def _cli():
    parser = argparse.ArgumentParser(
        description="Hourly community-energy bar chart from evaluation CSVs."
    )
    parser.add_argument("scenario_dir")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--run", default=None)
    parser.add_argument("--seed-glob", default="energy_market_training_seed*_run*")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--noise-std", type=float, default=None,
        help="If set, read the noisy-evaluation CSV "
             "(eval_noise_states_<tag>.csv) for that noise level instead "
             "of the legacy evaluation_agent_states.csv.",
    )
    parser.add_argument(
        "--heuristic", default=None, choices=HEURISTIC_CHOICES,
        help="If set, plot the heuristic baseline CSV from "
             "<scenario_dir>/heuristic_<name>/evaluation_agent_states.csv "
             "produced by evaluate_heuristic.py. "
             "Ignores --noise-std/--seed/--run.",
    )
    parser.add_argument(
        "--mixed-tag", default=None,
        help="If set, plot the mixed-policy CSVs produced by "
             "evaluate_mixed.py for that tag (e.g. 'midpoint_h45' reads "
             "eval_mixed_midpoint_h45.csv next to each seed checkpoint). "
             "Per-seed aggregation still applies. "
             "Mutually exclusive with --heuristic and --noise-std.",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    noise_tag = _noise_label(args.noise_std)  # '' or 'noise0p10'
    mixed_tag_lbl = _mixed_label(args.mixed_tag)  # '' or 'mixed_midpoint_h45'

    if args.output:
        save_path = Path(args.output).resolve()
    else:
        if args.heuristic:
            base = f"hourly_energy_heuristic_{args.heuristic}"
        elif args.mixed_tag:
            base = f"hourly_energy_mixed_{args.mixed_tag}"
        elif args.seed is not None:
            run_label = f"seed{args.seed}"
            if args.run is not None:
                run_label += f"_run{args.run}"
            base = f"hourly_energy_{run_label}"
        else:
            base = "hourly_energy_avg"
        # noise tag only meaningful outside heuristic/mixed mode
        if noise_tag and not args.heuristic and not args.mixed_tag:
            base = f"{base}_{noise_tag}"
        save_path = scenario_dir / f"{base}.png"

    df = plot_hourly_energy(
        scenario_dir,
        seed=args.seed,
        run=args.run,
        seed_glob=args.seed_glob,
        save_path=save_path,
        noise_std=args.noise_std,
        heuristic=args.heuristic,
        mixed_tag=args.mixed_tag,
    )

    if args.save_csv:
        csv_out = save_path.with_suffix(".csv")
        df.to_csv(csv_out)
        print(f"Saved aggregated CSV to: {csv_out}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    _cli()