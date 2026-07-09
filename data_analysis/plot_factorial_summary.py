"""
Factorial 3x3 summary table and efficiency-equity heatmap for the
observation x reward experiment grid.

Companion to plot_hourly_energy.py. Reuses the same conventions:
  * functions accept an optional external `ax`/`axes` and create their own
    figure only when none is passed;
  * styling (fonts, weights) is left to the global matplotlib rcParams,
    never hardcoded inside the body;
  * per-seed metrics are computed exactly as in plot_scenario_metrics_box,
    so the table and the heatmap report mean +/- std over the seeds, which
    is the multi-seed protocol that differentiates this work.

Two entry points per artifact:
  * `collect_factorial_metrics(...)` -> reads the exp_results_repro/ tree
                          (exact numbers, mean +/- std over the 10 seeds);
  * `cell_stats_from_values(...)`-> takes a dict of pre-computed numbers,
                          useful to reproduce the figures from values
                          already at hand.

Metric naming note
------------------
The monetary metric is reported as "payoff" (community economic payoff),
deliberately distinct from the welfare signals (Gini, Jain) used INSIDE the
reward. Jain and Gini here are PERFORMANCE measures on the realised payoff
distribution, not the reward terms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plot_hourly_energy import (
    find_runs,
    gini_coefficient,
    jain_index,
)

# =====================================================================
# Fixed factorial ordering (keeps every artifact consistent with the
# experimental-setup tables: S1->S2->S3, R1->R2->R3).
# =====================================================================

SCENARIOS = ["S1", "S2", "S3"]            # decreasing observability ->
SCENARIO_LABELS = {
    "S1": "S1",
    "S2": "S2",
    "S3": "S3",
}

REWARDS = ["R1", "R2", "R3"]
REWARD_LABELS = {
    "R1": "R1",
    "R2": "R2",
    "R3": "R3",
}

# Heatmap colormaps oriented so that "greener/darker = better".
CMAP_PAYOFF = "YlGn"      # higher payoff = better
CMAP_JAIN   = "YlGn"      # higher Jain   = better (1 = equal)
CMAP_GINI   = "YlGn_r"    # lower Gini    = better (0 = equal) -> reversed


# =====================================================================
# Per-configuration metric computation (mean +/- std over seeds)
# =====================================================================

def _seed_metrics_for_dir(
    scenario_dir: Union[str, Path],
    seed_glob: str = "energy_market_training_*seed*_run*",
    seed: Optional[Union[str, int]] = None,
    run: Optional[Union[str, int]] = None,
    noise_std: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Return per-seed vectors of payoff, Jain and Gini for one configuration.

    Mirrors the per-seed reduction used in plot_scenario_metrics_box:
    average the per-hour reward over sub-steps, sum over the day to get a
    per-agent daily payoff vector, then collapse that vector into the three
    scalar metrics. One scalar per seed.
    """
    runs = find_runs(
        scenario_dir, seed_glob=seed_glob,
        seed=seed, run=run, noise_std=noise_std,
    )
    if not runs:
        raise FileNotFoundError(f"No runs found in {scenario_dir}.")

    payoffs, jains, ginis = [], [], []
    for _, csv_path in runs:
        df = pd.read_csv(csv_path)
        reward_cols = sorted(
            [c for c in df.columns
             if c.startswith("agent_") and c.endswith("_reward")],
            key=lambda x: int(x.split("_")[1]),
        )
        per_hour_mean = df.groupby(["episode", "hour_index"])[reward_cols].mean()
        daily_cumulative = per_hour_mean.groupby("episode").sum()
        v = daily_cumulative.mean().values   # per-agent payoff, this seed

        payoffs.append(float(np.sum(v)))
        jains.append(jain_index(v))
        ginis.append(gini_coefficient(v))

    return {
        "payoff": np.asarray(payoffs, dtype=float),
        "jain":   np.asarray(jains,   dtype=float),
        "gini":   np.asarray(ginis,   dtype=float),
    }


def collect_factorial_metrics(
    dir_grid: Dict[Tuple[str, str], Union[str, Path]],
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    """Compute mean and std of each metric for every (reward, scenario) cell.

    Parameters
    ----------
    dir_grid : dict keyed by (reward, scenario) -> scenario_dir
        e.g. {("R3", "S3"): "../exp_results_repro/final_tesis_jain_local", ...}

    Returns
    -------
    dict keyed by (reward, scenario) -> {metric: (mean, std)}.
    """
    out: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for (rwd, scn), d in dir_grid.items():
        m = _seed_metrics_for_dir(d, seed_glob=seed_glob, noise_std=noise_std)
        out[(rwd, scn)] = {
            k: (float(np.mean(v)), float(np.std(v, ddof=1) if v.size > 1 else 0.0))
            for k, v in m.items()
        }
    return out


# =====================================================================
# 1. Summary table (LaTeX, IEEEtran-friendly)
# =====================================================================

def factorial_table_latex(
    cell_stats: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    p2p_share: Optional[Dict[Tuple[str, str], float]] = None,
    payoff_decimals: int = 0,
    index_decimals: int = 3,
    label: str = "tab:factorial_results",
    caption: str = ("Performance of the nine experimental configurations, "
                    "reported as mean $\\pm$ standard deviation over 10 seeds. "
                    "Payoff is the community economic payoff; Jain and Gini are "
                    "performance measures on the realised payoff distribution, "
                    "distinct from the welfare signals used inside the reward. "
                    "Best value of each metric in bold."),
) -> str:
    """Render the 3x3 factorial results as an IEEEtran LaTeX table string.

    Rows are grouped by reward (R1, R2, R3); columns are the three metrics
    (and optionally the P2P share) for each observation scenario. Best value
    per metric (across all nine cells) is bolded: max for payoff and Jain,
    min for Gini.
    """
    def _best(metric: str, maximize: bool) -> Tuple[str, str]:
        items = [(k, v[metric][0]) for k, v in cell_stats.items()]
        return (max if maximize else min)(items, key=lambda kv: kv[1])[0]

    best_payoff = _best("payoff", maximize=True)
    best_jain   = _best("jain",   maximize=True)
    best_gini   = _best("gini",   maximize=False)

    def _fmt(mean: float, std: float, dec: int, bold: bool) -> str:
        body = f"{mean:.{dec}f} $\\pm$ {std:.{dec}f}"
        return f"\\textbf{{{body}}}" if bold else body

    has_p2p = p2p_share is not None

    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    metric_block = "ccc" + ("c" if has_p2p else "")
    lines.append("\\begin{tabular}{l" + metric_block * 3 + "}")
    lines.append("\\toprule")

    span = 4 if has_p2p else 3
    header_top = ["\\textbf{Reward}"]
    for scn in SCENARIOS:
        header_top.append(
            f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{SCENARIO_LABELS[scn]}}}}}"
        )
    lines.append(" & ".join(header_top) + " \\\\")

    start = 2
    rules = []
    for _ in SCENARIOS:
        rules.append(f"\\cmidrule(lr){{{start}-{start + span - 1}}}")
        start += span
    lines.append("".join(rules))

    sub = [""]
    metric_names = ["Payoff", "Jain", "Gini"] + (["P2P\\%"] if has_p2p else [])
    for _ in SCENARIOS:
        sub.extend(metric_names)
    lines.append(" & ".join(sub) + " \\\\")
    lines.append("\\midrule")

    for rwd in REWARDS:
        row = [f"\\textbf{{{REWARD_LABELS[rwd]}}}"]
        for scn in SCENARIOS:
            st = cell_stats[(rwd, scn)]
            row.append(_fmt(*st["payoff"], payoff_decimals, (rwd, scn) == best_payoff))
            # row.append(_fmt(*st["jain"],  index_decimals,  (rwd, scn) == best_jain))
            row.append(_fmt(*st["gini"],  index_decimals,  (rwd, scn) == best_gini))
            if has_p2p:
                row.append(f"{p2p_share[(rwd, scn)]:.1f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# =====================================================================
# 2. Efficiency-equity heatmaps (one panel per metric)
# =====================================================================

def _grid_array(cell_stats, metric: str) -> np.ndarray:
    """Return a (3 rewards) x (3 scenarios) array of means for one metric."""
    arr = np.full((len(REWARDS), len(SCENARIOS)), np.nan)
    for i, rwd in enumerate(REWARDS):
        for j, scn in enumerate(SCENARIOS):
            arr[i, j] = cell_stats[(rwd, scn)][metric][0]
    return arr


def _annot_array(cell_stats, metric: str, decimals: int) -> np.ndarray:
    """Return the matching grid of 'mean\\nstd' annotation strings."""
    arr = np.empty((len(REWARDS), len(SCENARIOS)), dtype=object)
    for i, rwd in enumerate(REWARDS):
        for j, scn in enumerate(SCENARIOS):
            mean, std = cell_stats[(rwd, scn)][metric]
            arr[i, j] = f"{mean:.{decimals}f}\n$\\pm${std:.{decimals}f}"
    return arr


def _draw_metric_heatmap(ax, grid, annot, cmap: str, title: str) -> None:
    """Draw a single metric heatmap on `ax` with per-cell annotations.

    Text color flips to white on dark cells for readability. Font size and
    weight are left to rcParams; only the per-cell color is set here.
    """
    norm = mcolors.Normalize(vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(SCENARIOS)
    ax.set_yticks(np.arange(len(REWARDS)))
    ax.set_yticklabels(REWARDS)
    ax.set_title(title)

    cmap_obj = plt.get_cmap(cmap)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            rgba = cmap_obj(norm(grid[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, annot[i, j], ha="center", va="center", color=txt_color)

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(-0.5, len(SCENARIOS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(REWARDS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

def plot_factorial_heatmap(
    cell_stats,
    metrics: Tuple[str, ...] = ("payoff", "gini"),
    figsize: Tuple[float, float] = (4, 12),  # 1. Invertido para que sea más ancho que alto
    title: Optional[str] = "Efficiency-equity grid (mean over seeds)",
    payoff_decimals: int = 0,
    index_decimals: int = 3,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Render the factorial grid as one heatmap per metric.

    Rows are rewards (R1->R3), columns are observation scenarios (S1->S3),
    so the horizontal axis is decreasing observability. Colormaps are
    oriented so darker always means better (high payoff, low Gini).

    Returns (fig, axes). Pass an array of axes to compose into a larger
    figure; otherwise a new 1 x len(metrics) figure is created.
    """
    cmap_for = {"payoff": CMAP_PAYOFF, "gini": CMAP_GINI}
    title_for = {"payoff": "Community Payoff", "gini": "Gini coeff."}
    dec_for = {"payoff": payoff_decimals, "gini": index_decimals}

    if ax is None:
        # 2. Cambiado a (1, len(metrics)) para tener 1 fila y N columnas horizontales
        fig, ax = plt.subplots(len(metrics), figsize=figsize)
    else:
        fig = ax[0].figure
    ax = np.atleast_1d(ax)

    for axis, metric in zip(ax, metrics):
        grid = _grid_array(cell_stats, metric)
        annot = _annot_array(cell_stats, metric, dec_for[metric])
        _draw_metric_heatmap(axis, grid, annot, cmap_for[metric], title_for[metric])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax


# =====================================================================
# Convenience: build cell_stats straight from pre-computed values
# =====================================================================

def cell_stats_from_values(values):
    """Pass-through validator for hand-provided (mean, std) values.

    `values` must be keyed by (reward, scenario) with each metric mapped to
    a (mean, std) tuple. Std may be 0.0 if only means are available.
    """
    missing = {(r, s) for r in REWARDS for s in SCENARIOS} - set(values.keys())
    if missing:
        raise ValueError(f"Missing cells: {sorted(missing)}")
    return values


# =====================================================================
# 3. Efficiency-equity trade-off scatter (Pareto view)
# =====================================================================

# Marker per scenario, color per reward: both levers readable at once.
SCENARIO_MARKERS = {"S1": "o", "S2": "s", "S3": "^"}
REWARD_COLORS = {"R1": "#1f77b4", "R2": "#ff7f0e", "R3": "#2ca02c"}


def _pareto_front(
    points: List[Tuple[str, float, float]],
    higher_x_better: bool = True,
    higher_y_better: bool = True,
) -> List[Tuple[str, float, float]]:
    """Return the non-dominated subset of (label, x, y), sorted by x.

    A point dominates another when it is at least as good on both axes and
    strictly better on one. Orientation per axis is configurable so the same
    routine serves Jain (higher better) and Gini (lower better).
    """
    def _dominates(a, b) -> bool:
        ax_, ay_ = a[1], a[2]
        bx_, by_ = b[1], b[2]
        x_ge = ax_ >= bx_ if higher_x_better else ax_ <= bx_
        y_ge = ay_ >= by_ if higher_y_better else ay_ <= by_
        x_gt = ax_ > bx_ if higher_x_better else ax_ < bx_
        y_gt = ay_ > by_ if higher_y_better else ay_ < by_
        return x_ge and y_ge and (x_gt or y_gt)

    front = [p for p in points if not any(_dominates(q, p) for q in points if q is not p)]
    front.sort(key=lambda p: p[1], reverse=not higher_x_better)
    return front


# Style for each baseline marker on the trade-off plane.
BASELINE_STYLES = {
    "Centralized optimum": ("*", "#d62728", 320),
    "Midpoint heuristic":  ("D", "#9467bd", 110),
}


def plot_tradeoff_scatter(
    cell_stats: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
    equity: str = "jain",
    baselines: Optional[Dict[str, Dict[str, float]]] = None,
    figsize: Tuple[float, float] = (7, 6),
    title: Optional[str] = None,
    annotate: bool = True,
    show_errorbars: bool = True,
    show_pareto: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter the nine configurations as payoff (x) vs equity (y).

    `equity` is 'jain' (higher = fairer) or 'gini' (lower = fairer); the
    Pareto frontier orientation adapts accordingly. Color encodes the reward
    and marker shape encodes the observation scenario, so the efficiency-
    equity tension and both design levers are visible at once. Error bars use
    the per-seed standard deviation.

    `annotate` labels only the points on the Pareto frontier, which keeps the
    dominated cluster uncluttered.

    `baselines` overlays non-MARL reference points as distinct markers, keyed
    by label with a dict {'payoff': ..., 'jain': ..., 'gini': ...} as produced
    by `baseline_point`. The centralized optimum and the midpoint heuristic
    frame the MARL cloud as a theoretical ceiling and a non-learned reference.
    Returns (fig, ax).
    """
    if equity not in ("jain", "gini"):
        raise ValueError("equity must be 'jain' or 'gini'.")
    higher_y_better = (equity == "jain")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    points: List[Tuple[str, float, float]] = []
    for rwd in REWARDS:
        for scn in SCENARIOS:
            st = cell_stats[(rwd, scn)]
            px, pxs = st["payoff"]
            py, pys = st[equity]
            label = f"{rwd}-{scn}"
            points.append((label, px, py))

            if show_errorbars:
                ax.errorbar(px, py, xerr=pxs, yerr=pys, fmt="none",
                            ecolor=REWARD_COLORS[rwd], elinewidth=1,
                            capsize=3, alpha=0.5, zorder=2)
            ax.scatter(px, py, marker=SCENARIO_MARKERS[scn],
                       color=REWARD_COLORS[rwd], s=90, zorder=3,
                       edgecolors="white", linewidths=0.8)

    # Compute the Pareto frontier once. We reuse it for the dashed line and to
    # decide which points carry a label.
    front: List[Tuple[str, float, float]] = []
    if show_pareto or annotate:
        front = _pareto_front(points, higher_x_better=True,
                              higher_y_better=higher_y_better)

    if show_pareto and front:
        fx = [p[1] for p in front]
        fy = [p[2] for p in front]
        ax.plot(fx, fy, linestyle="--", color="0.4", linewidth=1.5,
                zorder=1, label="Pareto frontier")

    # Label only the frontier points to avoid overlap in the dominated cluster.
    if annotate:
        for label, px, py in front:
            ax.annotate(label, (px, py), textcoords="offset points",
                        xytext=(6, 6), zorder=4)

    # --- overlay non-MARL baselines as distinct markers ---
    if baselines:
        for label, pt in baselines.items():
            bx = pt["payoff"]
            by = pt[equity]

            mk, col, sz = BASELINE_STYLES.get(label, ("X", "black", 140))
            ax.scatter(bx, by, marker=mk, color=col, s=sz, zorder=6,
                       edgecolors="black", linewidths=1.0)

    ax.set_xlabel("Community payoff")
    ax.set_ylabel("Jain index" if equity == "jain" else "Gini coeff.")
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Two compact legends: reward color (left) and scenario marker + baselines (right).
    from matplotlib.lines import Line2D
    reward_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=REWARD_COLORS[r],
               label=REWARD_LABELS[r], markersize=9)
        for r in REWARDS
    ]
    scenario_handles = [
        Line2D([0], [0], marker=SCENARIO_MARKERS[s], linestyle="", color="0.3",
               label=SCENARIO_LABELS[s], markersize=9)
        for s in SCENARIOS
    ]
    baseline_handles = [
        Line2D([0], [0], marker=BASELINE_STYLES.get(k, ("X", "black", 140))[0],
               linestyle="", color=BASELINE_STYLES.get(k, ("X", "black", 140))[1],
               markeredgecolor="black", label=k, markersize=12)
        for k in (baselines or {})
    ]
    leg1 = ax.legend(handles=reward_handles, loc="lower left",
                     frameon=False, title="Reward")
    ax.add_artist(leg1)
    ax.legend(handles=scenario_handles + baseline_handles, loc="lower right",
              frameon=False)

    fig.tight_layout()
    return fig, ax

# =====================================================================
# 4. Hourly P2P clearing price across scenarios (one panel per reward)
# =====================================================================

from plot_hourly_energy import _collect_trade_prices_and_roles

# Scenario colors reused from the trade-off scatter for consistency.
SCEN_PRICE_COLORS = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c"}


def _hourly_price_stats(
    runs: List[Tuple[str, str]],
    cost_params: dict,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool settlement prices across seeds; return per-hour mean and std.

    `runs` is a list of (label, csv_path) as returned by find_runs. Prices are
    collected with the same routine used elsewhere, so only cells that
    actually traded contribute to the per-hour pool.
    """
    pool: dict = {}
    for _, csv in runs:
        by_hour_p, _, _, _ = _collect_trade_prices_and_roles(
            csv, cost_params=cost_params, eps=eps)
        for h, prices in by_hour_p.items():
            pool.setdefault(h, []).extend(prices)

    hours = np.array(sorted(pool.keys()), dtype=float)
    mean = np.array([np.mean(pool[h]) if pool[h] else np.nan for h in hours])
    std = np.array([np.std(pool[h], ddof=0) if pool[h] else np.nan for h in hours])
    return hours, mean, std


def plot_clearing_price_grid(
    dir_grid: Dict[Tuple[str, str], Union[str, Path]],
    profiles_json_path: Union[str, Path],
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    lambda_sell: float = 50.0,
    lambda_buy: float = 100.0,
    band: Optional[str] = "std",
    band_alpha: float = 0.18,
    figsize: Tuple[float, float] = (15, 4.5),
    title: Optional[str] = "Hourly P2P clearing price",
    ylabel: str = "Clearing price",
    xlabel: str = "Hour",
    show_grid_bounds: bool = True,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """One panel per reward; three scenario curves (S1/S2/S3) per panel."""
    from plot_hourly_energy import find_runs
    import json

    with open(Path(profiles_json_path).resolve(), "r") as f:
        profiles_data = json.load(f)
    cost_params = {
        i: profiles_data.get(f"agent_{i}", {}).get("cost_params", [0.0, 0.0, 0.0])
        for i in range(6)
    }

    if ax is None:
        fig, ax = plt.subplots(1, len(REWARDS), figsize=figsize, sharey=True)
    else:
        fig = ax[0].figure
    ax = np.atleast_1d(ax)

    for axis, rwd in zip(ax, REWARDS):
        if show_grid_bounds:
            axis.axhline(lambda_buy, linestyle=":", color="0.5", linewidth=1, zorder=1)
            axis.axhline(lambda_sell, linestyle=":", color="0.5", linewidth=1, zorder=1)
            axis.axhline(0.5 * (lambda_sell + lambda_buy),
                         linestyle="--", color="0.7", linewidth=1, zorder=1)

        for scn in SCENARIOS:
            runs = find_runs(dir_grid[(rwd, scn)], seed_glob=seed_glob,
                             noise_std=noise_std)
            if not runs:
                raise FileNotFoundError(f"No runs for {(rwd, scn)} "
                                        f"in {dir_grid[(rwd, scn)]}.")
            hours, mean, std = _hourly_price_stats(runs, cost_params)
            color = SCEN_PRICE_COLORS[scn]
            if band == "std":
                axis.fill_between(hours, mean - std, mean + std,
                                  color=color, alpha=band_alpha, linewidth=0, zorder=2)
            axis.plot(hours, mean, color=color, linewidth=2,
                      label=SCENARIO_LABELS[scn], zorder=3)

        axis.set_title(REWARD_LABELS[rwd])
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.3)
        
        # -------------------------------------------------------------
        # CAMBIOS AQUÍ: Forzar límites de 0 a 24 y definir marcas fijas
        # -------------------------------------------------------------
        axis.set_xlim(0, 24)
        # Opción A (marcas cada 4 horas): [0, 4, 8, 12, 16, 20, 24]
        axis.set_xticks([0, 4, 8, 12, 16, 20, 24]) 
        
        # Opción B (marcas cada 6 horas por si el gráfico queda muy saturado):
        # axis.set_xticks([0, 6, 12, 18, 24])
        # -------------------------------------------------------------

        axis.margins(x=0)

    ax[0].set_ylabel(ylabel)
    ax[0].legend(loc="best", frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax

# =====================================================================
# 5. P2P trade-flow network (daily aggregate, one panel per scenario)
# =====================================================================

import json as _json
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.cm import ScalarMappable


def _parse_matrix_column(series) -> np.ndarray:
    """Parse a Series of JSON 2-D-array strings into a 3-D numpy array."""
    return np.array([_json.loads(s) for s in series], dtype=np.float64)


def _aggregate_flow(
    runs: List[Tuple[str, str]],
    n_agents: int = 6,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate the trade matrices over the day and pool over seeds.

    Returns the total traded power per seller->buyer pair and the
    power-weighted mean settlement price of each pair.
    """
    P_tot = np.zeros((n_agents, n_agents))
    MP_tot = np.zeros((n_agents, n_agents))   # sum of price*power, for weighting
    for _, csv in runs:
        df = pd.read_csv(csv)
        P_arr = _parse_matrix_column(df["P"])
        M_arr = _parse_matrix_column(df["M"])
        for i in range(len(df)):
            P_tot += P_arr[i]
            MP_tot += M_arr[i] * P_arr[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        price_w = np.where(P_tot > eps, MP_tot / P_tot, np.nan)
    return P_tot, price_w


def _node_positions(n: int) -> np.ndarray:
    """Place n nodes evenly on a circle, starting at the top."""
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _concentration(P_tot: np.ndarray, eps: float = 1e-8) -> Tuple[float, int]:
    """Return the throughput share of the busiest agent and the number of
    active seller->buyer pairs. A higher share means a more concentrated
    market, which is the structural counterpart of an unequal payoff."""
    total = P_tot.sum()
    if total <= eps:
        return float("nan"), 0
    throughput = P_tot.sum(axis=1) + P_tot.sum(axis=0)
    top_share = throughput.max() / (2.0 * total)   # /2: each trade counted twice
    n_active_pairs = int((P_tot > eps).sum())
    return top_share, n_active_pairs


def plot_flow_graph(
    flow_grid: Dict[str, Union[str, Path]],
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    n_agents: int = 6,
    price_range: Tuple[float, float] = (50.0, 100.0),
    figsize: Tuple[float, float] = (15, 5.5),
    title: Optional[str] = "P2P trade network (daily aggregate)",
    cmap: str = "viridis",
    max_lw: float = 9.0,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Directed trade-flow network, one panel per observation scenario.

    Nodes are agents; an arrow goes from seller to buyer. Arrow width encodes
    the total power traded over the day, and arrow color encodes the
    power-weighted mean settlement price. Each panel is annotated with the
    number of active pairs and the throughput share of the busiest agent, so
    the concentration of the market is directly comparable across scenarios.

    `flow_grid` maps a scenario id ("S1"/"S2"/"S3") to its results directory,
    for a single fixed reward.
    """
    from plot_hourly_energy import find_runs

    if ax is None:
        fig, ax = plt.subplots(1, len(SCENARIOS), figsize=figsize)
    else:
        fig = ax[0].figure
    ax = np.atleast_1d(ax)

    pos = _node_positions(n_agents)
    norm = mcolors.Normalize(vmin=price_range[0], vmax=price_range[1])
    cmap_obj = plt.get_cmap(cmap)

    # First pass: aggregate every panel and find the global max power, so the
    # line-width scale is shared and panels are visually comparable.
    flows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    gmax = 0.0
    for scn in SCENARIOS:
        runs = find_runs(flow_grid[scn], seed_glob=seed_glob, noise_std=noise_std)
        if not runs:
            raise FileNotFoundError(f"No runs for {scn} in {flow_grid[scn]}.")
        P_tot, price_w = _aggregate_flow(runs, n_agents=n_agents)
        flows[scn] = (P_tot, price_w)
        gmax = max(gmax, P_tot.max())

    for axis, scn in zip(ax, SCENARIOS):
        P_tot, price_w = flows[scn]
        for s in range(n_agents):
            for b in range(n_agents):
                p = P_tot[s, b]
                if p <= 1e-8 or s == b:
                    continue
                lw = max_lw * (p / gmax)
                arrow = FancyArrowPatch(
                    pos[s], pos[b],
                    connectionstyle="arc3,rad=0.15",
                    arrowstyle="-|>", mutation_scale=12,
                    lw=lw, color=cmap_obj(norm(price_w[s, b])),
                    alpha=0.85, zorder=2, shrinkA=14, shrinkB=14,
                )
                axis.add_patch(arrow)
        for i in range(n_agents):
            axis.add_patch(Circle(pos[i], 0.13, facecolor="white",
                                  edgecolor="0.3", lw=1.5, zorder=3))
            axis.text(pos[i, 0], pos[i, 1], str(i), ha="center", va="center",
                      zorder=4)
        top_share, n_pairs = _concentration(P_tot)
        axis.set_title(f"{SCENARIO_LABELS[scn]}\n"
                       f"active pairs: {n_pairs}  |  top agent: {top_share*100:.0f}%")
        axis.set_xlim(-1.4, 1.4)
        axis.set_ylim(-1.4, 1.4)
        axis.set_aspect("equal")
        axis.axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean settlement price")
    if title:
        fig.suptitle(title)
    return fig, ax

# =====================================================================
# 6. P2P trade matrix heatmap (seller x buyer, daily aggregate)
# =====================================================================

def plot_flow_heatmap(
    flow_grid: Dict[str, Union[str, Path]],
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    n_agents: int = 6,
    quantity: str = "power",                  # "power" | "price"
    price_range: Tuple[float, float] = (50.0, 100.0),
    figsize: Tuple[float, float] = (15, 4.8),
    title: Optional[str] = None,
    annotate: bool = True,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Seller x buyer trade matrix, one heatmap per observation scenario.

    `quantity="power"` colors each cell by the total power traded between the
    seller (row) and the buyer (column) over the day, on a shared scale across
    panels. `quantity="price"` colors by the power-weighted mean settlement
    price of the pair, on the fixed [FiT, ToU] scale. Empty cells (no trade)
    are left blank so they do not bias the price scale. A concentrated market
    lights up few cells; a distributed one spreads the intensity, which is the
    structural reading of equity.

    `flow_grid` maps a scenario id ("S1"/"S2"/"S3") to its results directory,
    for a single fixed reward.
    """
    from plot_hourly_energy import find_runs

    if quantity not in ("power", "price"):
        raise ValueError("quantity must be 'power' or 'price'.")

    if ax is None:
        fig, ax = plt.subplots(1, len(SCENARIOS), figsize=figsize)
    else:
        fig = ax[0].figure
    ax = np.atleast_1d(ax)

    # Aggregate every panel; find global max power for a shared power scale.
    flows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    gmax = 0.0
    for scn in SCENARIOS:
        runs = find_runs(flow_grid[scn], seed_glob=seed_glob, noise_std=noise_std)
        if not runs:
            raise FileNotFoundError(f"No runs for {scn} in {flow_grid[scn]}.")
        P_tot, price_w = _aggregate_flow(runs, n_agents=n_agents)
        flows[scn] = (P_tot, price_w)
        gmax = max(gmax, P_tot.max())

    if quantity == "power":
        cmap = "magma"
        norm = mcolors.Normalize(0.0, gmax)
        clabel = "Traded power"
    else:
        cmap = "viridis"
        norm = mcolors.Normalize(*price_range)
        clabel = "Mean settlement price"
    cmap_obj = plt.get_cmap(cmap)

    for axis, scn in zip(ax, SCENARIOS):
        P_tot, price_w = flows[scn]
        if quantity == "power":
            grid = np.where(P_tot > 1e-8, P_tot, np.nan)
        else:
            grid = price_w
        # Mask empty cells so they render blank rather than as the scale min.
        masked = np.ma.masked_invalid(grid)
        cmap_obj.set_bad(color="white")
        axis.imshow(masked, cmap=cmap_obj, norm=norm, aspect="equal")

        axis.set_title(SCENARIO_LABELS[scn])
        axis.set_xticks(range(n_agents))
        axis.set_yticks(range(n_agents))
        axis.set_xlabel("Buyer")
        if scn == SCENARIOS[0]:
            axis.set_ylabel("Seller")

        if annotate:
            for s in range(n_agents):
                for b in range(n_agents):
                    v = grid[s, b]
                    if np.isnan(v):
                        continue
                    rgba = cmap_obj(norm(v))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    axis.text(b, s, f"{v:.0f}", ha="center", va="center",
                              color="white" if lum < 0.5 else "black")

        axis.set_xticks(np.arange(-0.5, n_agents, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, n_agents, 1), minor=True)
        axis.grid(which="minor", color="0.6", linewidth=1.0)
        axis.tick_params(which="minor", length=0)

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(clabel)
    if title:
        fig.suptitle(title)
    return fig, ax

# =====================================================================
# 7. P2P surplus-split matrix (color = split, opacity = power)
# =====================================================================

def plot_split_matrix(
    flow_grid: Dict[str, Union[str, Path]],
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    n_agents: int = 6,
    lambda_sell: float = 50.0,
    lambda_buy: float = 100.0,
    figsize: Tuple[float, float] = (15, 4.8),
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    alpha_floor: float = 0.15,
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Seller x buyer surplus-split matrix, one panel per scenario.

    A single matrix encodes both quantities of each pair: the color is the
    signed surplus-split index s = (M - midpoint) / half_range in [-1, 1],
    where midpoint = (lambda_sell + lambda_buy)/2 and half_range is half the
    tariff gap. With RdBu_r, blue means the trade favored the buyer, red the
    seller, and white an equal 50/50 split. The opacity encodes the traded
    power normalized to the global maximum, so large trades appear solid and
    small ones faint; cells with no trade stay blank.

    Each panel is annotated with the power-weighted mean split s-bar, a single
    scalar that summarizes whether the market favored sellers (positive) or
    buyers (negative) on average. `flow_grid` maps a scenario id to its
    results directory for a single fixed reward.
    """
    from plot_hourly_energy import find_runs

    midpoint = 0.5 * (lambda_sell + lambda_buy)
    half_range = 0.5 * (lambda_buy - lambda_sell)

    if ax is None:
        fig, ax = plt.subplots(1, len(SCENARIOS), figsize=figsize)
    else:
        fig = ax[0].figure
    ax = np.atleast_1d(ax)

    flows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    gmax = 0.0
    for scn in SCENARIOS:
        runs = find_runs(flow_grid[scn], seed_glob=seed_glob, noise_std=noise_std)
        if not runs:
            raise FileNotFoundError(f"No runs for {scn} in {flow_grid[scn]}.")
        P_tot, price_w = _aggregate_flow(runs, n_agents=n_agents)
        flows[scn] = (P_tot, price_w)
        gmax = max(gmax, P_tot.max())

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap_obj = plt.get_cmap(cmap)

    for axis, scn in zip(ax, SCENARIOS):
        P_tot, price_w = flows[scn]
        s = (price_w - midpoint) / half_range

        rgba = np.ones((n_agents, n_agents, 4))   # white background
        pw_num, pw_den = 0.0, 0.0
        for i in range(n_agents):
            for j in range(n_agents):
                p = P_tot[i, j]
                if p <= 1e-8 or np.isnan(s[i, j]):
                    continue
                base = np.array(cmap_obj(norm(np.clip(s[i, j], -1.0, 1.0))))
                base[3] = alpha_floor + (1.0 - alpha_floor) * (p / gmax)
                rgba[i, j] = base
                pw_num += p * s[i, j]
                pw_den += p
        axis.imshow(rgba, aspect="equal")

        s_bar = pw_num / pw_den if pw_den > 0 else float("nan")
        axis.set_title(f"{SCENARIO_LABELS[scn]}\n" fr"$\bar{{s}}$ = {s_bar:+.2f}")
        axis.set_xticks(range(n_agents))
        axis.set_yticks(range(n_agents))
        axis.set_xlabel("Buyer")
        if scn == SCENARIOS[0]:
            axis.set_ylabel("Seller")

        if annotate:
            for i in range(n_agents):
                for j in range(n_agents):
                    if P_tot[i, j] > 1e-8 and not np.isnan(s[i, j]):
                        axis.text(j, i, f"{s[i, j]:+.2f}", ha="center", va="center",
                                  color="0.15")

        axis.set_xticks(np.arange(-0.5, n_agents, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, n_agents, 1), minor=True)
        axis.grid(which="minor", color="0.6", linewidth=1.0)
        axis.tick_params(which="minor", length=0)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02,
                        ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Surplus split  (buyer $-$ seller)")
    if title:
        fig.suptitle(title)
    return fig, ax

# =====================================================================
# 9. Per-agent P2P surplus, decomposed by counterpart
# =====================================================================

def _agent_surplus(
    runs: List[Tuple[str, str]],
    n_agents: int = 6,
    lambda_sell: float = 50.0,
    lambda_buy: float = 100.0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Total P2P surplus captured by each agent over the day, pooled over seeds
    and decomposed by counterpart.

    For each pairwise trade seller i -> buyer j at price M and quantity P, the
    seller captures (M - lambda_sell)*P over exporting and the buyer captures
    (lambda_buy - M)*P over importing. Returns the per-agent surplus vector and
    the [agent, counterpart] matrix whose row-sum is that vector.
    """
    by_pair = np.zeros((n_agents, n_agents))
    for _, csv in runs:
        df = pd.read_csv(csv)
        P_arr = _parse_matrix_column(df["P"])
        M_arr = _parse_matrix_column(df["M"])
        for t in range(len(df)):
            P, M = P_arr[t], M_arr[t]
            for i in range(n_agents):       # seller
                for j in range(n_agents):   # buyer
                    p = P[i, j]
                    if p <= eps:
                        continue
                    by_pair[i, j] += (M[i, j] - lambda_sell) * p   # seller gain
                    by_pair[j, i] += (lambda_buy - M[i, j]) * p     # buyer gain
    by_pair /= max(1, len(runs))
    return by_pair.sum(axis=1), by_pair


def plot_agent_surplus(
    flow_grid: Dict[str, Union[str, Path]],
    seed_glob: str = "energy_market_training_*seed*_run*",
    noise_std: Optional[float] = None,
    n_agents: int = 6,
    lambda_sell: float = 50.0,
    lambda_buy: float = 100.0,
    eval_days: Optional[float] = 50.0,
    figsize: Tuple[float, float] = (15, 4.8),
    title: Optional[str] = None,
    cmap: str = "tab10",
    ax: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Per-agent mean daily P2P surplus, stacked by counterpart, one panel per scenario.

    Each bar is the mean P2P surplus an agent captures over a single day. The
    stacked segments show which counterparts that surplus came from. The
    height shows how evenly value is distributed across agents. The segments
    show how concentrated each agent's trading relationships are. Each panel is
    annotated with the Gini coefficient of the surplus vector, which measures
    the inequality of the captured value across agents.

    `_agent_surplus` accumulates surplus over every evaluation episode. We
    divide by `eval_days` to recover a per-day mean. Set it to the number of
    days summed into each bar (the evaluation episode count, times the number of
    seeds if seeds are pooled). Pass None to plot the raw accumulated value.

    `flow_grid` maps a scenario id to its results directory for a fixed reward.
    """
    from plot_hourly_energy import find_runs

    if ax is None:
        fig, ax = plt.subplots(1, len(SCENARIOS), figsize=figsize, sharey=True)
    else:
        fig = ax[0].figure
    ax = np.atleast_1d(ax)
    cmap_obj = plt.get_cmap(cmap)

    flows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scn in SCENARIOS:
        runs = find_runs(flow_grid[scn], seed_glob=seed_glob, noise_std=noise_std)
        if not runs:
            raise FileNotFoundError(f"No runs for {scn} in {flow_grid[scn]}.")
        surplus, by_pair = _agent_surplus(runs, n_agents, lambda_sell, lambda_buy)
        if eval_days:
            surplus = surplus / eval_days
            by_pair = by_pair / eval_days
        flows[scn] = (surplus, by_pair)

    for axis, scn in zip(ax, SCENARIOS):
        surplus, by_pair = flows[scn]
        bottom = np.zeros(n_agents)
        for c in range(n_agents):
            axis.bar(range(n_agents), by_pair[:, c], bottom=bottom,
                     color=cmap_obj(c % 10), edgecolor="white", linewidth=0.5,
                     label=f"Agent {c}" if scn == SCENARIOS[-1] else None)
            bottom += by_pair[:, c]

        # Vectorized Gini coefficient of the surplus vector.
        surplus_sum = surplus.sum()
        if surplus_sum > 0:
            abs_diffs = np.abs(surplus[:, None] - surplus[None, :])
            gini = abs_diffs.sum() / (2 * n_agents * surplus_sum)
        else:
            gini = float("nan")

        axis.set_title(f"{SCENARIO_LABELS[scn]}\nGini = {gini:.3f}")
        axis.set_xlabel("Agent")
        axis.set_xticks(range(n_agents))
        axis.grid(True, axis="y", alpha=0.3)

    ax[0].set_ylabel("Mean daily P2P economic surplus")
    ax[-1].legend(title="Counterpart", loc="upper right", frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax
# =====================================================================
# 10. Non-MARL baselines on the efficiency-equity plane
# =====================================================================

def baseline_point(
    csv_path: Union[str, Path],
    n_agents: int = 6,
    steps_per_hour: int = 4,
) -> Dict[str, float]:
    """Reduce a baseline evaluation CSV (heuristic or centralized) to the
    payoff, Jain and Gini of its per-agent daily payoff vector.

    The reduction matches the canonical pipeline: the reward is averaged over
    the sub-steps within each hour and then summed over the hours, since the
    steps_per_hour sub-steps represent the same hour and must not be added.
    The per-agent daily payoff is then averaged over episodes. Returns a dict
    with keys 'payoff', 'jain', 'gini'.
    """
    df = pd.read_csv(csv_path)
    rcols = sorted(
        c for c in df.columns
        if c.endswith("_reward") and c != "mean_reward"
    )

    if "hour_index" in df.columns:
        per_hour = df.groupby(["episode", "hour_index"])[rcols].mean()
    else:
        df = df.copy()
        df["_hour"] = df["step"] // steps_per_hour
        per_hour = df.groupby(["episode", "_hour"])[rcols].mean()

    daily = per_hour.groupby(level=0).sum()
    v = daily.mean().values

    total = float(v.sum())
    sq = float((v ** 2).sum())
    jain = (total ** 2) / (n_agents * sq) if sq > 0 else float("nan")
    diffs = float(np.abs(v[:, None] - v[None, :]).sum())
    gini = diffs / (2.0 * n_agents * total) if total > 0 else float("nan")
    return {"payoff": total, "jain": jain, "gini": gini}

# Style for each baseline marker on the trade-off plane.
BASELINE_STYLES = {
    "Centralized optimum": ("*", "#d62728", 320),
    "Midpoint heuristic":  ("D", "#9467bd", 110),
}