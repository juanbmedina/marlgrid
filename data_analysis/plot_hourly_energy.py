"""
Hourly community-energy plotting — both a CLI script and an importable
module for notebook use.

All five quantities (generation, demand, P2P, import, export) are derived
directly from the evaluation CSV (either `evaluation_agent_states.csv`
produced by evaluate_legacy.py, or `eval_noise_states_<tag>.csv`
produced by evaluate_legacy_noise.py):

    Generation per hour =  Σ cap[i]   over agents where role_vec[i] == +1
    Demand per hour     =  Σ cap[i]   over agents where role_vec[i] == -1
    P2P / Import / Export per hour = mean of total_p2p / total_import /
                                     total_export across the 4 sub-steps
                                     within that hour.

Why MEAN over sub-steps instead of SUM: the steps_per_hour=4 sub-steps are
there so the agents can stabilise their actions inside one market hour,
not to represent four separate hourly trades. The hourly traded energy is
the stabilised value, so we average.

Why no agents_profiles_24h.json: cap (and role_vec) are emitted in info
by the env at every step, so generation and demand are reconstructable
without going back to the source profiles.

Selecting noisy vs clean evaluations
------------------------------------
Pass `noise_std=<float>` to load the noisy-evaluation CSV produced by
evaluate_legacy_noise.py for that noise level. Default is `None`, which
keeps the legacy behaviour and reads `evaluation_agent_states.csv`.

Notebook usage:

    from plot_hourly_energy import plot_hourly_energy

    # Legacy (clean) evaluation:
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp")

    # Noisy evaluation, +/-10% gaussian:
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp",
                            noise_std=0.10)

    # One particular seed:
    df = plot_hourly_energy("exp_results_repro/.../descentralized_exp",
                            seed=42, noise_std=0.10)

CLI usage:

    # Legacy CSV:
    python3 plot_hourly_energy.py exp_results_repro/.../descentralized_exp

    # Noisy CSV for noise_std=0.10:
    python3 plot_hourly_energy.py exp_results_repro/.../descentralized_exp \
        --noise-std 0.10
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


# =====================================================================
# CSV name resolution
# =====================================================================

def _eval_csv_name(noise_std: Optional[float]) -> str:
    """Resolve the evaluation CSV filename to read.

    noise_std=None        -> 'evaluation_agent_states.csv'  (legacy)
    noise_std=<float>     -> 'eval_noise_states_<tag>.csv'  where
                             tag = f"{noise_std:.2f}".replace(".", "p")
                             (mirror of evaluate_legacy_noise.py).
    """
    if noise_std is None:
        return "evaluation_agent_states.csv"
    tag = f"{float(noise_std):.2f}".replace(".", "p")
    return f"eval_noise_states_{tag}.csv"


def _noise_label(noise_std: Optional[float]) -> str:
    """Short human-readable noise tag for titles / filenames."""
    if noise_std is None:
        return ""
    return f"noise{float(noise_std):.2f}".replace(".", "p")


# =====================================================================
# Discovery
# =====================================================================

def find_runs(
    scenario_dir: Union[str, Path],
    seed_glob: str = "energy_market_training_seed*_run*",
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    noise_std: Optional[float] = None,
) -> List[Tuple[str, Path]]:
    """Return list of (run_label, csv_path) for each evaluation CSV found.

    Pass `noise_std=<float>` to discover the noisy-evaluation CSV
    produced by evaluate_legacy_noise.py for that noise level.
    """
    scenario_dir = Path(scenario_dir)
    csv_name = _eval_csv_name(noise_std)

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
           generation = Σ cap[i] for sellers (role_vec[i] == +1)
           demand     = Σ cap[i] for buyers  (role_vec[i] == -1)
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
    seed_glob: str = "energy_market_training_seed*_run*",
    noise_std: Optional[float] = None,
):
    """
    Daily cumulative reward per agent, averaged across episodes and seeds.

    Pipeline:
      1. Average the 4 sub-steps inside each hour_index.
      2. Sum hourly averages over the 24 hours -> daily total per episode.
      3. Average daily totals across episodes (per seed), then across seeds.

    Equity diagnostics shown:
      Jain   - fairness index in [1/N, 1]; 1 = perfectly equal split.
      LossN  - fraction of agents ending the day with negative payoff.
      LossV  - total losses / total gains across agents.
      Gini   - inequality index (with shift to handle negatives).

    Pass `noise_std=<float>` to read the noisy-evaluation CSV for that
    noise level instead of the legacy clean one.

    Per-seed values are printed to stdout when seed=None.
    """
    runs = find_runs(
        scenario_dir, seed_glob=seed_glob,
        seed=seed, run=run, noise_std=noise_std,
    )
    if not runs:
        raise FileNotFoundError(
            f"No se encontraron ejecuciones en {scenario_dir} "
            f"(noise_std={noise_std!r})."
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
            "gini":  gini_coefficient(v),
            "jain":  jain_index(v),
            "lossN": ls["loss_count"],
            "lossV": ls["loss_volume"],
        }))

    # 3b: mean across seeds.
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

    # ---------- Print per-seed metrics when aggregating ----------
    if seed is None and len(per_seed_metrics) > 1:
        noise_suffix = (f"  (noise_std={noise_std})"
                        if noise_std is not None else "")
        print(f"\n[plot_cumulative_rewards] {Path(scenario_dir).name}"
              f"{noise_suffix}")
        print(f"  Aggregate (cross-seed mean vector):")
        print(f"    Jain  = {agg_jain:.4f}    LossN = {agg_lossN:.2%}    "
              f"LossV = {agg_lossV:.4f}    Gini = {agg_gini:.4f}")
        print(f"  Per-seed:")
        header = f"    {'seed/run':<35s}  {'Jain':>7s}  {'LossN':>7s}  {'LossV':>7s}  {'Gini':>7s}"
        print(header)
        print(f"    {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
        for lab, m in sorted(per_seed_metrics, key=lambda t: t[0]):
            lv_str = "inf" if np.isinf(m["lossV"]) else f"{m['lossV']:.4f}"
            print(f"    {lab:<35s}  {m['jain']:>7.4f}  {m['lossN']:>6.2%}  "
                  f"{lv_str:>7s}  {m['gini']:>7.4f}")
        # mean / std across seeds for each metric
        for key, fmt in [("jain", "{:.4f}"), ("lossN", "{:.2%}"),
                         ("lossV", "{:.4f}"), ("gini", "{:.4f}")]:
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
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Append noise tag to the title if noise is active.
    if noise_std is not None:
        title = f"{title} · noise σ={float(noise_std):.2f}"

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("Agent")

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(ylim)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # Equity diagnostics inset (top-right). Jain and LossN as primary;
    # LossV and Gini as context.
    lossV_str = "inf" if np.isinf(agg_lossV) else f"{agg_lossV:.3f}"
    inset_text = (
        f"Jain   = {agg_jain:.3f}\n"
        f"LossN  = {agg_lossN:.1%}\n"
        f"LossV  = {lossV_str}\n"
        f"Gini   = {agg_gini:.3f}"
    )
    ax.text(
        0.97, 0.95, inset_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="lightgrey", alpha=0.9),
    )

    plt.tight_layout()
    return fig, ax, agg_jain


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
            fig.text(0.5, 0.005, summary, ha="center", fontsize=9, color="#333")
        else:
            ax.text(0.5, -0.18, summary, transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, color="#333")

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
) -> pd.DataFrame:
    """Discover → aggregate → plot, in one call.

    Parameters
    ----------
    noise_std : float, optional
        If set, reads the noisy-evaluation CSV produced by
        evaluate_legacy_noise.py for that noise level
        (`eval_noise_states_<tag>.csv`) instead of the clean
        legacy CSV (`evaluation_agent_states.csv`).

    Returns
    -------
    DataFrame with columns
        ['generation', 'demand', 'p2p_trade', 'grid_import', 'grid_export']
    indexed by hour_index.
    """
    scenario_dir = Path(scenario_dir).resolve()
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"scenario_dir not found: {scenario_dir}")

    runs = find_runs(scenario_dir, seed_glob=seed_glob,
                     seed=seed, run=run, noise_std=noise_std)
    if not runs:
        raise FileNotFoundError(
            f"No runs in {scenario_dir} "
            f"(seed={seed!r}, run={run!r}, noise_std={noise_std!r}). "
            f"Expected file: {_eval_csv_name(noise_std)}"
        )
    if verbose:
        csv_name = _eval_csv_name(noise_std)
        print(f"Found {len(runs)} run(s) in {scenario_dir.name} "
              f"(CSV={csv_name}):")
        for label, csv in runs:
            print(f"  - {label}")

    per_hour, n_runs, mode_label = aggregate_runs(runs)

    if title is None:
        title = f"Hourly community energy · {scenario_dir.name} · {mode_label}"
        if noise_std is not None:
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
    seed_glob="energy_market_training_seed*_run*",
    verbose=False,
    noise_std=None,
):
    runs = find_runs(scenario_dir, seed_glob=seed_glob,
                     seed=seed, run=run, noise_std=noise_std)
    if not runs:
        raise FileNotFoundError(
            f"No runs in {scenario_dir} "
            f"(seed={seed!r}, run={run!r}, noise_std={noise_std!r}). "
            f"Expected file: {_eval_csv_name(noise_std)}"
        )
    if verbose:
        print(f"Found {len(runs)} run(s). CSV={_eval_csv_name(noise_std)}")

    per_hour, _, _ = aggregate_runs(runs)
    hours = per_hour.index.values.astype(int)

    # ── Métricas de resumen ──────────────────────────────────────────────
    total_demand  = per_hour["demand"].sum()
    total_p2p     = per_hour["p2p"].sum()
    total_gen     = per_hour["generation"].sum()
    total_imp     = per_hour["imp"].sum()
    total_exp     = per_hour["exp"].sum()

    p2p_demand_pct = 100.0 * total_p2p / min(total_demand, total_gen)

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

    # Append noise tag to the title if noise is active.
    if noise_std is not None:
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
        f"G: {total_gen:.1f} kWh   ·   "
        f"D: {total_demand:.1f} kWh   ·   "
        f"I: {total_imp:.1f} kWh   ·   "
        f"E: {total_exp:.1f} kWh"
    )
    ax.text(
        0.5, -0.15,
        stats_line,
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        va="top",
        color="black",
        clip_on=False,
    )

    if own_fig:
        fig.subplots_adjust(bottom=0.18)   # hace espacio para la línea extra
    return per_hour


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
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    noise_tag = _noise_label(args.noise_std)  # '' or 'noise0p10'

    if args.output:
        save_path = Path(args.output).resolve()
    else:
        if args.seed is not None:
            run_label = f"seed{args.seed}"
            if args.run is not None:
                run_label += f"_run{args.run}"
            base = f"hourly_energy_{run_label}"
        else:
            base = "hourly_energy_avg"
        if noise_tag:
            base = f"{base}_{noise_tag}"
        save_path = scenario_dir / f"{base}.png"

    df = plot_hourly_energy(
        scenario_dir,
        seed=args.seed,
        run=args.run,
        seed_glob=args.seed_glob,
        save_path=save_path,
        noise_std=args.noise_std,
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