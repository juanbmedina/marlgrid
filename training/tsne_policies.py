"""
t-SNE visualization of learned MARL policies (decentralized PPO / IPPO).

This module rolls out the trained policies, records the (observation, action)
pairs each agent produces, and projects the observations to 2D with t-SNE.
A single embedding is computed once and recolored by several context
variables (agent, role, action price, hour), which is the standard and
efficient way to read a t-SNE map.

Self-contained: the checkpoint loaders are included here (copied from
evaluate.py) so there is no cross-module import. RLlib / torch / repo
imports are done lazily inside the rollout functions, so the plotting half
loads in a notebook without Ray.

Importable for notebook use:

    from tsne_policies import collect_policy_io, run_tsne, plot_tsne_panel

    X, df = collect_policy_io(checkpoint_path, num_episodes=50)
    emb, perp = run_tsne(X)
    plot_tsne_panel(emb, df, perplexity=perp, save_path="tsne_panel")

CLI / script use (mirrors evaluate.py, env-var driven). Run as a module
from the repo root so `envs` and `training` are importable:

    EVAL_EXPERIMENT_DIR=./exp_results_repro/.../energy_market_training_..._seed42_run61 \
    TSNE_NUM_EPISODES=50 \
    python -m training.tsne_policies

Control via env vars (all optional):
  EVAL_EXPERIMENT_DIR   default: ./exp_results/energy_market_training
                        may point at either a trial dir (one that directly
                        contains checkpoint_* folders) or a parent of trials.
  EVAL_CHECKPOINT_PATH  (optional override, points straight at a checkpoint_*
                        folder and skips autodetect)
  TSNE_NUM_EPISODES     default: 50
  TSNE_PERPLEXITY       default: 30
  TSNE_SEED             default: 0
  TSNE_SWEEP            e.g. "5,30,50" -> also emit a perplexity-sweep figure
                        (empty by default, which skips it)
  TSNE_SWEEP_COLOR      color variable for the sweep (default: role)

Outputs are written next to the checkpoint, under tsne_analysis/:
  obs_matrix.npy          raw observation matrix [N, obs_dim]
  policy_io_context.csv   per-row context (agent, role, action, hour, ...)
  tsne_embedding.npy      2D embedding [N, 2]
  tsne_panel.{png,pdf}    3-panel row (agent, role, action price)

When TSNE_SWEEP is set, additionally:
  tsne_embedding_perp{P}.npy   2D embedding per perplexity
  tsne_sweep_by_{color}.{png,pdf}   one variable across perplexities (overview)
  tsne_panel_perp{P}.{png,pdf}      3-panel row per perplexity (the three cases)
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


# =====================================================================
# CONFIG (edit or set env vars)
# =====================================================================
EXPERIMENT_DIR = os.environ.get("EVAL_EXPERIMENT_DIR", "./exp_results/energy_market_training")
CHECKPOINT_OVERRIDE = os.environ.get("EVAL_CHECKPOINT_PATH", "").strip()
NUM_EPISODES = int(os.environ.get("TSNE_NUM_EPISODES", "50"))
PERPLEXITY = float(os.environ.get("TSNE_PERPLEXITY", "30"))
RANDOM_STATE = int(os.environ.get("TSNE_SEED", "0"))
# Optional perplexity sweep. Set e.g. TSNE_SWEEP="5,30,50" to also emit a
# side-by-side sweep figure. Empty (default) skips it.
SWEEP_PERPLEXITIES = os.environ.get("TSNE_SWEEP", "").strip()
SWEEP_COLOR = os.environ.get("TSNE_SWEEP_COLOR", "role").strip()
# Cache-only mode: collect the rollout and save the cache, then stop (skip
# t-SNE and figures). Handy to do the heavy rollout on the Ray console and
# all the exploration later in a notebook via PolicyTSNE.from_cache().
CACHE_ONLY = os.environ.get("TSNE_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes", "y")


# Role coloring reuses the energy palette semantically (seller = generation
# green, buyer = demand red), so the t-SNE map is consistent with the rest
# of the figures. Agents use a qualitative colormap.
ROLE_COLORS = {
    "seller":  "#2ca02c",  # green  (generation)
    "buyer":   "#d62728",  # red    (demand)
    "neutral": "#7f7f7f",  # gray
}
AGENT_CMAP = "tab10"


# =====================================================================
# Checkpoint loaders (copied from evaluate.py to keep this self-contained)
# =====================================================================
def find_latest_checkpoint(experiment_dir: str) -> str:
    """Resolve a checkpoint path.

    Accepts either:
      A) a trial dir that directly contains checkpoint_* folders, or
      B) a parent dir containing several trial dirs.
    Picks the highest-index checkpoint (of the newest trial in case B).
    """
    experiment_dir = Path(os.path.abspath(experiment_dir))
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment dir does not exist: {experiment_dir}")

    # Case A: pointed straight at a trial dir holding checkpoint_* folders.
    direct = list(experiment_dir.glob("checkpoint_*"))
    if direct:
        latest = max(direct, key=lambda p: int(p.name.split("_")[-1]))
        return os.path.abspath(str(latest))

    # Case B: pointed at a parent that contains trial dirs.
    trial_dirs = [p for p in experiment_dir.iterdir() if p.is_dir()]
    valid_trials = [p for p in trial_dirs if list(p.glob("checkpoint_*"))]
    if not valid_trials:
        raise ValueError(
            f"No checkpoint_* folders found in {experiment_dir} "
            f"nor in any of its immediate subdirectories."
        )

    latest_trial = max(valid_trials, key=lambda p: p.stat().st_mtime)
    checkpoints = list(latest_trial.glob("checkpoint_*"))
    latest = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))
    return os.path.abspath(str(latest))


def load_env_config_used(checkpoint_path: str) -> dict:
    """Load env_config_used.json from the trial dir (parent of checkpoint_*).
    Falls back to {} if missing."""
    ckpt = Path(checkpoint_path).resolve()
    trial_dir = ckpt.parent
    json_path = trial_dir / "env_config_used.json"

    if not json_path.exists():
        print(f"[tsne] env_config_used.json not found at: {json_path}")
        print("[tsne] Falling back to env defaults ({}).")
        return {}

    with json_path.open("r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"[tsne] env_config_used.json is not a dict: {json_path}")

    print(f"[tsne] Loaded env_config from: {json_path}")
    return cfg


def load_rlmodules(checkpoint_path: str, env_config: dict) -> dict:
    """Load one RLModule per policy from:
      checkpoint_*/learner_group/learner/rl_module/{policy_id}
    """
    from ray.rllib.core.rl_module.rl_module import RLModule
    from envs.energy_env import P2PEnergyEnv

    base = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    if not base.exists():
        raise FileNotFoundError(f"[tsne] RLModule base path not found: {base}")

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
            raise ValueError(f"[tsne] RLModule path not found for {policy_id}: {policy_path}")
        print(f"[tsne] Loading RLModule {policy_id} from {policy_path}")
        m = RLModule.from_checkpoint(str(policy_path))
        m.eval()
        rlmodules[policy_id] = m

    return rlmodules


def deterministic_action(env, agent_id, policy_id, rl_module, obs):
    """Deterministic action from an RLModule's inference output."""
    import torch
    from ray.rllib.core.columns import Columns
    from ray.rllib.utils.numpy import convert_to_numpy

    del policy_id  # kept only for signature consistency

    obs = np.asarray(obs, dtype=np.float32)
    input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

    with torch.no_grad():
        out = rl_module.forward_inference(input_dict)

    if Columns.ACTIONS in out:
        action_np = convert_to_numpy(out[Columns.ACTIONS])[0]
    else:
        if Columns.ACTION_DIST_INPUTS not in out:
            raise KeyError(
                f"[tsne] RLModule output has neither {Columns.ACTIONS} nor "
                f"{Columns.ACTION_DIST_INPUTS}. Keys: {list(out.keys())}"
            )
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
# Agent-feature decoding
# =====================================================================
# Every observation is concatenate([global_vec, agent_features]), so the
# 17-dim agent block is always the LAST 17 entries, regardless of obs_mode.
# Layout (see energy_env._build_obs):
#   [is_seller, is_buyer, is_neutral, idx_norm, D, G, net, cap, q, p,
#    sold_p2p, bought_p2p, grid_export, grid_import, a, b, c]
_AGENT_FEAT_DIM = 17
_ROLE_NAMES = np.array(["seller", "buyer", "neutral"])


def _decode_role(obs_row: np.ndarray) -> str:
    onehot = obs_row[-_AGENT_FEAT_DIM: -_AGENT_FEAT_DIM + 3]
    return str(_ROLE_NAMES[int(np.argmax(onehot))])


def _decode_net(obs_row: np.ndarray) -> float:
    return float(obs_row[-_AGENT_FEAT_DIM + 6])


# =====================================================================
# Rollout: collect (observation, action, context)
# =====================================================================
def collect_policy_io(
    checkpoint_path: str,
    num_episodes: int = NUM_EPISODES,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Roll out the trained policies and record observations, deterministic
    actions, and context (episode, step, hour, agent_id, policy_id, role,
    net_energy, action_qty, action_price)."""
    from envs.energy_env import P2PEnergyEnv
    from training.policy_mapping import policy_mode

    env_config = load_env_config_used(checkpoint_path)
    mapping_fn = policy_mode(env_config)
    env = P2PEnergyEnv(env_config)
    rlmodules = load_rlmodules(checkpoint_path, env_config)

    obs_rows = []
    ctx_rows = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        step = 0

        while not (terminateds.get("__all__", False) or truncateds.get("__all__", False)):
            hour = int(getattr(env, "current_hour", 0))  # state that produced obs
            actions = {}

            for agent_id, agent_obs in obs.items():
                policy_id = mapping_fn(agent_id, ep, None)
                action = deterministic_action(
                    env=env,
                    agent_id=agent_id,
                    policy_id=policy_id,
                    rl_module=rlmodules[policy_id],
                    obs=agent_obs,
                )

                ov = np.asarray(agent_obs, dtype=np.float32)
                a = np.asarray(action, dtype=np.float32).reshape(-1)
                obs_rows.append(ov)
                ctx_rows.append({
                    "episode": ep + 1,
                    "step": step,
                    "hour": hour,
                    "agent_id": agent_id,
                    "policy_id": policy_id,
                    "role": _decode_role(ov),
                    "net_energy": _decode_net(ov),
                    "action_qty": float(a[0]),
                    "action_price": float(a[1]),
                })
                actions[agent_id] = action

            obs, _, terminateds, truncateds, _ = env.step(actions)
            step += 1

        print(f"[tsne] episode {ep + 1}/{num_episodes} collected")

    X = np.vstack(obs_rows).astype(np.float32)
    df = pd.DataFrame(ctx_rows)
    print(f"[tsne] collected {X.shape[0]} agent-steps | obs_dim = {X.shape[1]}")
    return X, df


# =====================================================================
# t-SNE
# =====================================================================
def run_tsne(
    X: np.ndarray,
    perplexity: float = PERPLEXITY,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, float]:
    """Standardize observations and project them to 2D with t-SNE. Returns
    the embedding and the perplexity actually used (capped below n_samples)."""
    Xs = StandardScaler().fit_transform(X)
    n = Xs.shape[0]
    perp = float(min(perplexity, max(5.0, (n - 1) / 3.0)))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    emb = tsne.fit_transform(Xs)
    return emb, perp


# =====================================================================
# Plotting
# =====================================================================
def _save_fig(fig: plt.Figure, save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    for ext in (".png", ".pdf"):
        fig.savefig(save_path.with_suffix(ext), dpi=150, bbox_inches="tight")


def plot_tsne(
    emb: np.ndarray,
    df: pd.DataFrame,
    color_by: str = "agent_id",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 7),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    point_size: float = 10.0,
    alpha: float = 0.7,
) -> plt.Axes:
    """Scatter the 2D embedding colored by one context column. Categorical
    columns (role, agent_id) get a legend; others get a colorbar."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if color_by == "role":
        for role, c in ROLE_COLORS.items():
            m = df["role"].values == role
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=point_size, c=c,
                           alpha=alpha, label=role, edgecolors="none")
        ax.legend(title="Role", frameon=False)
    elif color_by == "agent_id":
        agents = sorted(df["agent_id"].unique())
        cmap = plt.get_cmap(AGENT_CMAP)
        for i, ag in enumerate(agents):
            m = df["agent_id"].values == ag
            ax.scatter(emb[m, 0], emb[m, 1], s=point_size, color=cmap(i % 10),
                       alpha=alpha, label=ag, edgecolors="none")
        ax.legend(title="Agent", frameon=False, ncol=2)
    else:
        vals = df[color_by].values.astype(float)
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=point_size, c=vals,
                        cmap="viridis", alpha=alpha, edgecolors="none")
        fig.colorbar(sc, ax=ax, label=color_by)

    ax.set_title(title or f"t-SNE colored by {color_by}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])

    if save_path is not None:
        _save_fig(fig, save_path)
    return ax


def plot_tsne_panel(
    emb: np.ndarray,
    df: pd.DataFrame,
    perplexity: Optional[float] = None,
    figsize: Tuple[float, float] = (18, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Three views of the same embedding in a horizontal row: agent, role,
    action price. Use `figsize` to set the image size (width, height)."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_tsne(emb, df, "agent_id",     ax=axes[0], title="By agent")
    plot_tsne(emb, df, "role",         ax=axes[1], title="By role")
    plot_tsne(emb, df, "action_price", ax=axes[2], title="By action price (action[1])")

    if perplexity is not None:
        fig.suptitle(f"t-SNE of policy observations (perplexity = {perplexity:.0f})",
                     fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        _save_fig(fig, save_path)
    return fig


# =====================================================================
# Perplexity sweep
# =====================================================================
def run_tsne_sweep(
    X: np.ndarray,
    perplexities=(5, 30, 50),
    random_state: int = RANDOM_STATE,
):
    """Compute one independent t-SNE embedding per perplexity (reusing the
    already-collected observations; no extra rollouts). Returns a list of
    (perplexity_used, embedding)."""
    results = []
    for p in perplexities:
        emb, perp = run_tsne(X, perplexity=float(p), random_state=random_state)
        results.append((perp, emb))
        print(f"[tsne] sweep: requested perplexity {p} -> used {perp:.1f}")
    return results


def plot_tsne_sweep(
    results,
    df: pd.DataFrame,
    color_by: str = "role",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    point_size: float = 10.0,
    alpha: float = 0.7,
) -> plt.Figure:
    """Plot each embedding of a perplexity sweep side by side, colored by the
    same variable, so you can check which structure survives across
    perplexities. Only structure stable across all panels is trustworthy."""
    n = len(results)
    if figsize is None:
        figsize = (6.0 * n, 6.0)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (perp, emb) in zip(axes, results):
        plot_tsne(emb, df, color_by=color_by, ax=ax,
                  title=f"perplexity = {perp:.0f}",
                  point_size=point_size, alpha=alpha)

    fig.suptitle(f"t-SNE perplexity sweep (colored by {color_by})", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        _save_fig(fig, save_path)
    return fig


def save_panels_over_perplexities(
    results,
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    prefix: str = "tsne_panel_perp",
):
    """Save a full 4-coloring panel (agent, role, action price, hour) for each
    perplexity in a sweep, using perplexity-tagged filenames so nothing is
    overwritten. Reuses the sweep embeddings (no extra t-SNE runs)."""
    out_dir = Path(out_dir)
    paths = []
    for perp, emb in results:
        stem = out_dir / f"{prefix}{int(round(perp)):02d}"
        plot_tsne_panel(emb, df, perplexity=perp, save_path=stem)
        plt.close("all")  # avoid piling up open figures
        paths.append(stem)
    return paths


# =====================================================================
# Notebook API
# =====================================================================
class PolicyTSNE:
    """Notebook-friendly wrapper around this module.

    Run the (heavy) rollout once, then explore t-SNE interactively without
    recomputing it. Embeddings are cached per perplexity. Plot methods return
    the figure/axes and do not close them, so they render inline.

    Typical notebook flow (needs Ray + the repo importable):

        from tsne_policies import PolicyTSNE

        an = PolicyTSNE.from_checkpoint(
            experiment_dir="./exp_results_repro/.../seed42_run61",
            num_episodes=1,
        )
        an.save("tsne_analysis")            # optional: cache to disk

        an.plot(color_by="role")            # single scatter (inline)
        an.plot(color_by="action_price", perplexity=50)
        an.panel(perplexity=30)             # 4-coloring panel
        an.sweep([5, 30, 50])               # robustness overview

    Later, lightweight and WITHOUT Ray (re-analyze from the cache):

        an = PolicyTSNE.from_cache("tsne_analysis")
        an.panel(perplexity=30)
    """

    def __init__(self, X: np.ndarray, df: pd.DataFrame):
        self.X = np.asarray(X, dtype=np.float32)
        self.df = df.reset_index(drop=True)
        self.checkpoint_path = None
        self._embeddings = {}  # requested perplexity (float) -> embedding

    # ---- constructors ----------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint_path=None, experiment_dir=None,
                        num_episodes: int = 1) -> "PolicyTSNE":
        """Resolve a checkpoint, roll out once, hold (X, df) in memory.
        Pass either checkpoint_path (a checkpoint_* folder) or experiment_dir
        (trial dir or a parent of trials). Requires Ray."""
        from envs.register_env import register_energy_env
        register_energy_env()

        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(experiment_dir or EXPERIMENT_DIR)
        checkpoint_path = os.path.abspath(checkpoint_path)

        X, df = collect_policy_io(checkpoint_path, num_episodes)
        obj = cls(X, df)
        obj.checkpoint_path = checkpoint_path
        return obj

    @classmethod
    def from_cache(cls, out_dir) -> "PolicyTSNE":
        """Rebuild from a previous run's cache (obs_matrix.npy +
        policy_io_context.csv). Does NOT need Ray."""
        out_dir = Path(out_dir)
        X = np.load(out_dir / "obs_matrix.npy")
        df = pd.read_csv(out_dir / "policy_io_context.csv")
        return cls(X, df)

    @classmethod
    def from_arrays(cls, X: np.ndarray, df: pd.DataFrame) -> "PolicyTSNE":
        """Wrap an already-collected (X, df) pair."""
        return cls(X, df)

    # ---- core ------------------------------------------------------------
    def tsne(self, perplexity: float = 30, random_state: int = RANDOM_STATE,
             recompute: bool = False) -> np.ndarray:
        """Return the 2D embedding at a given perplexity (cached per value)."""
        key = float(perplexity)
        if recompute or key not in self._embeddings:
            emb, _ = run_tsne(self.X, perplexity=perplexity, random_state=random_state)
            self._embeddings[key] = emb
        return self._embeddings[key]

    def plot(self, color_by: str = "role", perplexity: float = 30,
             ax=None, save_path=None, **kwargs):
        """Single scatter colored by one context column. Renders inline."""
        emb = self.tsne(perplexity)
        return plot_tsne(emb, self.df, color_by=color_by, ax=ax,
                         save_path=save_path, **kwargs)

    def panel(self, perplexity: float = 30, figsize=None, save_path=None):
        """Panel of three views (agent, role, action price) in a row.
        Pass figsize=(width, height) to set the image size."""
        emb = self.tsne(perplexity)
        kwargs = {} if figsize is None else {"figsize": figsize}
        return plot_tsne_panel(emb, self.df, perplexity=perplexity,
                               save_path=save_path, **kwargs)

    def sweep(self, perplexities=(5, 30, 50), color_by: str = "role", save_path=None):
        """One variable across several perplexities, to check robustness."""
        results = [(float(p), self.tsne(p)) for p in perplexities]
        return plot_tsne_sweep(results, self.df, color_by=color_by, save_path=save_path)

    def save(self, out_dir):
        """Persist X, df and any computed embeddings for later from_cache()."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "obs_matrix.npy", self.X)
        self.df.to_csv(out_dir / "policy_io_context.csv", index=False)
        for key, emb in self._embeddings.items():
            np.save(out_dir / f"tsne_embedding_perp{int(round(key)):02d}.npy", emb)
        return out_dir


# =====================================================================
# Script entry point
# =====================================================================
def main():
    from envs.register_env import register_energy_env

    register_energy_env()

    if CHECKPOINT_OVERRIDE:
        checkpoint_path = os.path.abspath(CHECKPOINT_OVERRIDE)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"EVAL_CHECKPOINT_PATH does not exist: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(EXPERIMENT_DIR)

    out_dir = Path(checkpoint_path).resolve().parent / "tsne_analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"[tsne] checkpoint: {checkpoint_path}")
    print(f"[tsne] output dir: {out_dir}")

    X, df = collect_policy_io(checkpoint_path, NUM_EPISODES)
    np.save(out_dir / "obs_matrix.npy", X)
    df.to_csv(out_dir / "policy_io_context.csv", index=False)
    print(f"[tsne] cache saved: {out_dir / 'obs_matrix.npy'} + policy_io_context.csv")

    if CACHE_ONLY:
        print("[tsne] TSNE_CACHE_ONLY set -> skipping t-SNE and figures.")
        print(f"[tsne] load it in a notebook: PolicyTSNE.from_cache('{out_dir}')")
        return

    emb, perp = run_tsne(X)
    np.save(out_dir / "tsne_embedding.npy", emb)
    print(f"[tsne] embedding done (perplexity used = {perp:.1f})")

    plot_tsne_panel(emb, df, perplexity=perp, save_path=out_dir / "tsne_panel")
    plt.close("all")
    print(f"[tsne] panel saved: {out_dir / 'tsne_panel'}.(png|pdf)")

    if SWEEP_PERPLEXITIES:
        perps = [float(x) for x in SWEEP_PERPLEXITIES.split(",") if x.strip()]
        results = run_tsne_sweep(X, perplexities=perps)

        # cache each embedding so panels can be recolored offline
        for p_used, emb_p in results:
            np.save(out_dir / f"tsne_embedding_perp{int(round(p_used)):02d}.npy", emb_p)

        # quick overview: one variable across perplexities
        plot_tsne_sweep(results, df, color_by=SWEEP_COLOR,
                        save_path=out_dir / f"tsne_sweep_by_{SWEEP_COLOR}")
        plt.close("all")

        # full 4-coloring panel per perplexity (the three cases, no overwrite)
        panel_paths = save_panels_over_perplexities(results, df, out_dir)
        print(f"[tsne] sweep overview + per-perplexity panels saved under {out_dir}")
        for pth in panel_paths:
            print(f"        {pth}.(png|pdf)")


if __name__ == "__main__":
    main()