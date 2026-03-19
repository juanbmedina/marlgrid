"""
community_welfare_rewards.py
─────────────────────────────
Drop-in reward-shaping module for P2PEnergyEnv.

Each public function receives the raw data already available inside
`_compute_payoffs_and_metrics` (or obtainable from env state) and returns
a single float per agent that can be combined with the individual payoff.

Naming convention
    r_<name>(...)  →  per-agent scalar  (used in reward composition)
    m_<name>(...)  →  community-level scalar  (used for logging / info)

All functions are stateless; the env keeps no extra buffers.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# =====================================================================
# 1. UTILITARIAN  SOCIAL  WELFARE  (Benthamite)
#    W = (1/N) * Σ payoff_i
#    Inspiration: classical welfare economics
# =====================================================================

def r_utilitarian_welfare(
    norm_payoffs: Dict[str, float],
) -> Dict[str, float]:
    """
    Each agent receives the community average normalised payoff as its
    social component.  Maximising this encourages Pareto-improving trades.
    """
    vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
    mean_w = float(np.mean(vals)) if vals.size else 0.0
    return {aid: mean_w for aid in norm_payoffs}


# =====================================================================
# 2. EGALITARIAN  (Rawlsian  Maximin)
#    W = min_i  payoff_i
#    Inspiration: Rawls' "Theory of Justice" — the community is only
#    as well-off as its worst-off member.
# =====================================================================

def r_rawlsian_maximin(
    norm_payoffs: Dict[str, float],
) -> Dict[str, float]:
    """
    Every agent's social reward equals the payoff of the worst-off agent.
    This pushes the whole community to lift the floor.
    """
    vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
    min_w = float(np.min(vals)) if vals.size else 0.0
    return {aid: min_w for aid in norm_payoffs}


# =====================================================================
# 3. GINI  FAIRNESS  PENALTY
#    G(x) = Σ_i Σ_j |x_i - x_j| / (2 N Σ x_i)
#    reward_social = −G   (closer to 0 is better)
#    Inspiration: development economics / inequality measurement
# =====================================================================

def m_gini(values: np.ndarray) -> float:
    """Gini coefficient ∈ [0, 1].  0 = perfect equality."""
    v = np.asarray(values, dtype=np.float64)
    if v.size <= 1 or np.sum(np.abs(v)) < 1e-12:
        return 0.0
    v_shifted = v - v.min() + 1e-8          # shift to non-negative
    n = v_shifted.size
    diffs = np.abs(v_shifted[:, None] - v_shifted[None, :]).sum()
    return float(diffs / (2 * n * v_shifted.sum()))


def r_gini_fairness(
    norm_payoffs: Dict[str, float],
) -> Dict[str, float]:
    """
    Negative Gini as a shared penalty.  G = 0 → no penalty.
    The agent reward component is  -(Gini)  so policies learn to
    reduce inequality.
    """
    vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
    penalty = -m_gini(vals)
    return {aid: penalty for aid in norm_payoffs}


# =====================================================================
# 4. JAIN'S  FAIRNESS  INDEX
#    J(x) = (Σ x_i)² / (N · Σ x_i²)    ∈ [1/N, 1]
#    Inspiration: telecommunications / network resource allocation
#    (Jain, Chiu & Hawe, 1984)
# =====================================================================

def m_jain_index(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 1.0
    v_shifted = v - v.min() + 1e-8
    s = v_shifted.sum()
    ss = (v_shifted ** 2).sum()
    if ss < 1e-12:
        return 1.0
    return float(s ** 2 / (v.size * ss))


def r_jain_fairness(
    norm_payoffs: Dict[str, float],
) -> Dict[str, float]:
    """
    Jain index as a positive shared bonus.  J = 1 means perfect fairness.
    """
    vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
    j = m_jain_index(vals)
    return {aid: j for aid in norm_payoffs}


# =====================================================================
# 5. PROPORTIONAL  FAIRNESS  (Nash  Bargaining / log-welfare)
#    W = Σ ln(payoff_i + ε)
#    Inspiration: Kelly (1997) proportional fairness in networks;
#    Nash bargaining solution in cooperative game theory.
#    Maximising the sum of logs balances efficiency and equity.
# =====================================================================

def r_proportional_fairness(
    norm_payoffs: Dict[str, float],
    eps: float = 1e-4,
) -> Dict[str, float]:
    """
    Each agent receives  (1/N) Σ ln(payoff_j + eps).
    The logarithm makes marginal improvements for low-payoff agents
    worth more, naturally discouraging extreme inequality.
    """
    vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
    vals_safe = np.clip(vals, 0.0, None) + eps
    log_welfare = float(np.mean(np.log(vals_safe)))
    return {aid: log_welfare for aid in norm_payoffs}


# =====================================================================
# 6. GRID  INDEPENDENCE  RATIO  (energy-domain specific)
#    GIR = P2P_total / (P2P_total + Grid_import + Grid_export)
#    Inspiration: microgrid self-sufficiency literature
# =====================================================================

def m_grid_independence(
    total_p2p: float,
    total_import: float,
    total_export: float,
) -> float:
    denom = total_p2p + total_import + total_export
    if denom < 1e-12:
        return 0.0
    return float(total_p2p / denom)


def r_grid_independence(
    total_p2p: float,
    total_import: float,
    total_export: float,
    agent_ids: List[str],
) -> Dict[str, float]:
    """
    Shared bonus proportional to how much of the total energy flow
    was resolved inside the community (P2P) vs. the external grid.
    """
    gir = m_grid_independence(total_p2p, total_import, total_export)
    return {aid: gir for aid in agent_ids}


# =====================================================================
# 7. DEMAND  SATISFACTION  RATIO  (per-agent + community)
#    DSR_i = energy_obtained_i / demand_i
#    Inspiration: reliability engineering / QoS in power systems
# =====================================================================

def r_demand_satisfaction(
    P: np.ndarray,
    grid_import: np.ndarray,
    caps: np.ndarray,
    roles: np.ndarray,
    agent_ids: List[str],
) -> Dict[str, float]:
    """
    Buyers get DSR_i = satisfied / demanded.
    Sellers get 1.0 (their demand is met by own generation).
    Neutrals get 1.0.
    Community reward = mean(DSR_i).
    """
    n = len(agent_ids)
    dsr = np.ones(n, dtype=np.float64)

    for idx in range(n):
        if roles[idx] == -1 and caps[idx] > 1e-8:   # buyer
            bought_p2p = float(np.sum(P[:, idx]))
            bought_grid = float(grid_import[idx])
            dsr[idx] = min((bought_p2p + bought_grid) / caps[idx], 1.0)

    mean_dsr = float(np.mean(dsr))
    return {aid: mean_dsr for aid in agent_ids}


# =====================================================================
# 8. PRICE  VOLATILITY  PENALTY
#    σ_price  over all clearing prices in the P2P matrix this step.
#    Lower volatility ⇒ more predictable, stable community market.
#    Inspiration: financial-market microstructure / electricity market design.
# =====================================================================

def m_price_volatility(M: np.ndarray, P: np.ndarray) -> float:
    """Std-dev of active clearing prices. 0 = single uniform price."""
    active = P > 1e-8
    if not np.any(active):
        return 0.0
    return float(np.std(M[active]))


def r_price_stability(
    M: np.ndarray,
    P: np.ndarray,
    price_range: float,
    agent_ids: List[str],
) -> Dict[str, float]:
    """
    Normalised negative volatility.  Low σ → reward close to 0.
    High σ → negative penalty.
    """
    sigma = m_price_volatility(M, P)
    penalty = -(sigma / max(price_range, 1e-8))
    return {aid: penalty for aid in agent_ids}


# =====================================================================
# 9. ENVY-FREENESS  PENALTY  (from fair-division theory)
#    Agent i "envies" agent j when payoff_j > payoff_i.
#    Envy = (1/N) Σ_i max_j(payoff_j - payoff_i, 0)
#    Inspiration: computational social choice / fair allocation
# =====================================================================

def m_envy(norm_payoffs: Dict[str, float]) -> float:
    vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
    if vals.size <= 1:
        return 0.0
    max_val = vals.max()
    return float(np.mean(np.maximum(max_val - vals, 0.0)))


def r_envy_penalty(
    norm_payoffs: Dict[str, float],
) -> Dict[str, float]:
    """Negative mean envy shared equally."""
    penalty = -m_envy(norm_payoffs)
    return {aid: penalty for aid in norm_payoffs}


# =====================================================================
# 10. COALITION  SURPLUS  RATIO  (cooperative game theory)
#     Compares current total payoff with the "no-P2P" counterfactual
#     where everyone trades only with the grid.
#     CSR = (Σ payoff_i − Σ payoff_i^grid) / |Σ payoff_i^grid| + ε
#     Inspiration: Shapley value literature / core stability
# =====================================================================

def r_coalition_surplus(
    payoffs: Dict[str, float],
    grid_only_payoffs: Dict[str, float],
    agent_ids: List[str],
    eps: float = 1e-4,
) -> Dict[str, float]:
    """
    Shared reward = how much better the community is with P2P
    compared to pure grid trading.
    grid_only_payoffs must be computed by the env (see integration notes).
    """
    actual = sum(payoffs.values())
    baseline = sum(grid_only_payoffs.values())
    surplus = (actual - baseline) / (abs(baseline) + eps)
    return {aid: float(surplus) for aid in agent_ids}


# =====================================================================
# COMPOSITE  REWARD  BUILDER
# =====================================================================

def build_composite_reward(
    norm_payoffs: Dict[str, float],
    social_components: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    alpha_individual: float = 0.5,
) -> Dict[str, float]:
    """
    r_i = alpha * norm_payoff_i + (1-alpha) * Σ_k w_k * social_k_i

    Parameters
    ----------
    norm_payoffs :        per-agent normalised individual payoff
    social_components :   {"metric_name": {agent_id: value, ...}, ...}
    weights :             {"metric_name": weight, ...}  (should sum ≈ 1)
    alpha_individual :    balance between selfish and social (0 = pure social)
    """
    agent_ids = list(norm_payoffs.keys())
    rewards: Dict[str, float] = {}

    for aid in agent_ids:
        social = 0.0
        for name, component in social_components.items():
            w = weights.get(name, 0.0)
            social += w * component.get(aid, 0.0)

        rewards[aid] = float(
            alpha_individual * norm_payoffs[aid]
            + (1.0 - alpha_individual) * social
        )

    return rewards