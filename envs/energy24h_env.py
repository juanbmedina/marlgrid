# energy_env.py
import os
import json
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.community_welfare_rewards import (
    r_utilitarian_welfare,
    r_rawlsian_maximin,
    r_gini_fairness,
    r_jain_fairness,
    r_proportional_fairness,
    r_grid_independence,
    r_demand_satisfaction,
    r_price_stability,
    r_envy_penalty,
    build_composite_reward,
    m_gini,
    m_jain_index,
    m_grid_independence,
    m_price_volatility,
    m_envy,
)


@dataclass
class AgentProfile:
    name: str
    consumer_profile: np.ndarray
    generator_profile: np.ndarray
    a: float
    b: float
    c: float
    D: float = 0.0
    G: float = 0.0


class P2PEnergyEnv(MultiAgentEnv):
    """
    P2P market with dynamic roles.

    Key design:
    - Agent IDs are fixed over time.
    - Roles (seller / buyer / neutral) change with the hour according to G-D.
    - All agents share the same action space: Box(0,1,(2,))
    - Action meaning depends on current role:
        seller -> [absolute offer quantity fraction, absolute ask price fraction]
        buyer  -> [absolute bid quantity fraction, absolute bid price fraction]
        neutral -> action ignored
    - Full NxN matrices P and M are kept with fixed size, so observation/info shape is stable.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}

        # ---------------- numeric ----------------
        self.eps = float(config.get("eps", 1e-8))
        self.max_steps = int(config.get("max_steps", 96))
        self.step_count = 0

        # ---------------- prices ----------------
        self.pi_gb = float(config.get("pi_min", 50.0))
        self.pi_gs = float(config.get("pi_max", 100.0))

        self.lambda_buy = float(config.get("lambda_buy", self.pi_gs))
        self.lambda_sell = float(config.get("lambda_sell", self.pi_gb))

        if self.lambda_sell > self.lambda_buy + 1e-6:
            raise ValueError(
                f"lambda_sell ({self.lambda_sell}) must be <= lambda_buy ({self.lambda_buy})."
            )

        # ---------------- buyer utility ----------------
        self.lambda_u = float(config.get("lambda_u", self.pi_gs))
        self.theta_u = float(config.get("theta_u", 1.0))
        if self.theta_u <= 0:
            raise ValueError("theta_u must be > 0 for concave utility.")

        # ---------------- reward mode ----------------
        self.reward_mode = str(config.get("reward_mode", "payoff")).lower()
        if self.reward_mode not in ("payoff", "welfare"):
            raise ValueError("reward_mode must be 'payoff' or 'welfare'")

        self.beta = float(config.get("beta", 0.0))
        self.alpha = float(config.get("alpha", 0.0))
        self.training_mode = str(config.get("training_mode", "individual")).lower()

        # Welfare reward configuration
        self.welfare_mode = str(config.get("welfare_mode", "none")).lower()
        # Options: "none", "utilitarian", "rawlsian", "gini", "jain",
        #          "proportional", "grid_independence", "demand_satisfaction",
        #          "price_stability", "envy", "composite"

        self.alpha_individual = float(config.get("alpha_individual", 0.6))
        # 1.0 = pure selfish,  0.0 = pure social
        
        # For composite mode: per-metric weights
        self.welfare_weights = config.get("welfare_weights", {
            "utilitarian":        0.25,
            "gini":               0.25,
            "grid_independence":  0.25,
            "demand_satisfaction": 0.25,
        })

        # ---------------- pairwise settlement ----------------
        self.pair_pricing_rule = str(config.get("pair_pricing_rule", "ask")).lower()
        if self.pair_pricing_rule not in ("ask", "bid", "midpoint"):
            raise ValueError("pair_pricing_rule must be 'ask', 'bid', or 'midpoint'")

        # ---------------- logging ----------------
        self.enable_csv_log = bool(config.get("enable_csv_log", False))
        self.custom_log_path = config.get(
            "custom_log_path",
            "/workspace/exp_results/energy_market_training/custom_metrics.csv",
        )
        os.makedirs(os.path.dirname(self.custom_log_path), exist_ok=True)
        self.episode_id = 0

        # ---------------- profiles ----------------
        json_path = self._resolve_json_path(config.get("agents_json_path"))
        self.agents_all: List[AgentProfile] = self._load_agents(json_path)
        self.n_agents = len(self.agents_all)
        self.num_hours = self._validate_profile_horizon(self.agents_all)

        # ---------------- forecast noise ----------------
        self.profile_noise_std = float(config.get("profile_noise_std", 0.0))
        self.profile_noise_type = str(config.get("profile_noise_type", "gaussian")).lower()
        if self.profile_noise_type not in ("gaussian", "uniform"):
            raise ValueError("profile_noise_type must be 'gaussian' or 'uniform'")

        inferred_sph = max(1, self.max_steps // max(1, self.num_hours))
        self.steps_per_hour = int(config.get("steps_per_hour", inferred_sph))
        if self.steps_per_hour <= 0:
            raise ValueError("steps_per_hour must be >= 1")

        self.hour_mode = str(config.get("hour_mode", "hold_last")).lower()
        if self.hour_mode not in ("hold_last", "wrap"):
            raise ValueError("hour_mode must be 'hold_last' or 'wrap'")

        # ---------------- fixed agent IDs ----------------
        # Use the JSON keys directly: agent_0, agent_1, ...
        self.possible_agents = [ag.name for ag in self.agents_all]
        self.agents = list(self.possible_agents)
        self.agent_name_to_idx = {ag.name: i for i, ag in enumerate(self.agents_all)}

        # policies
        self.individual_policy_ids = [f"{aid}_policy" for aid in self.possible_agents]
        # Important: with dynamic roles, a single shared policy is the sane "group" option.
        self.group_policy_ids = ["shared_policy"]

        # ---------------- common action space ----------------
        common_act = gym.spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_spaces = {aid: common_act for aid in self.possible_agents}

        # ---------------- dynamic per-agent arrays ----------------
        # role: +1 seller, -1 buyer, 0 neutral
        self.current_hour = 0
        self.role = np.zeros((self.n_agents,), dtype=np.int8)
        self.cap = np.zeros((self.n_agents,), dtype=np.float32)
        self.q = np.zeros((self.n_agents,), dtype=np.float32)
        self.p = np.full((self.n_agents,), 0.5 * (self.pi_gb + self.pi_gs), dtype=np.float32)

        # current seller / buyer index lists
        self.current_seller_idx: List[int] = []
        self.current_buyer_idx: List[int] = []

        # ---------------- full fixed-size market outcomes ----------------
        # P[seller_idx, buyer_idx]
        self.P = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        self.M = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)

        self.grid_import = np.zeros((self.n_agents,), dtype=np.float32)
        self.grid_export = np.zeros((self.n_agents,), dtype=np.float32)
        # self.mu = float(0.5 * (self.pi_gb + self.pi_gs))

        # ---------------- observation spaces ----------------
        # Global:
        # [role(N), cap(N), q(N), p(N), D(N), G(N), grid_import(N), grid_export(N), mu, hour_norm, step_norm]
        global_obs_dim = 2 * self.n_agents + 2

        # Agent-local:
        # [is_seller, is_buyer, is_neutral, idx_norm, D_i, G_i, net_i, cap_i, q_i, p_i,
        #  sold_p2p_i, bought_p2p_i, grid_export_i, grid_import_i, a_i, b_i, c_i]
        self.agent_feature_dim = 17
        obs_dim = global_obs_dim + self.agent_feature_dim

        self.observation_spaces = {
            aid: gym.spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)
            for aid in self.possible_agents
        }

        # evaluation compatibility
        self.state: Dict[str, object] = {}

        # initialize noisy profiles (no noise on first call, seed not set yet)
        self._sample_noisy_profiles()

        # initialize hour 0
        self._set_hour(0, reset_quotes=True)

    # =====================================================================
    # RLlib API
    # =====================================================================

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.episode_id += 1

        self._sample_noisy_profiles()
        self._set_hour(0, reset_quotes=False)
        self._update_state()

        obs = self._build_obs()
        infos = {
            aid: {
                "hour_index": int(self.current_hour),
                "role": self._role_str_idx(self.agent_name_to_idx[aid]),
            }
            for aid in self.possible_agents
        }
        return obs, infos

    def step(self, action_dict):
        transition_hour = int(self.current_hour)

        # --------------------------------------------------
        # 1) Interpret same action space according to role
        # --------------------------------------------------
        for aid in self.possible_agents:
            idx = self.agent_name_to_idx[aid]
            a = np.asarray(
                action_dict.get(aid, np.zeros(2, dtype=np.float32)),
                dtype=np.float32
            ).reshape(2)

            a_q = float(np.clip(a[0], 0.0, 1.0))
            a_p = float(np.clip(a[1], 0.0, 1.0))

            if self.role[idx] == 1:
                # SELLER
                q_state = a_q * float(self.cap[idx])
                self.q[idx] = np.clip(q_state, 0.0, float(self.cap[idx]))

                prof = self.agents_all[idx]
                if self.q[idx] > self.eps:
                    cost = float(prof.a * (self.q[idx] ** 2) + prof.b * self.q[idx] + prof.c)
                    u_cost = cost / self.q[idx]
                    min_cost = float(max(u_cost, self.pi_gb))
                else:
                    min_cost = float(self.pi_gb)

                p_state = min_cost + a_p * (float(self.pi_gs) - min_cost)
                self.p[idx] = np.clip(p_state, min_cost, self.pi_gs)
                

            elif self.role[idx] == -1:
                # BUYER
                q_state =  a_q * float(self.cap[idx])
                p_state = self.pi_gb + a_p * (float(self.pi_gs) - float(self.pi_gb))
                self.q[idx] = np.clip(q_state, 0.0, float(self.cap[idx]))
                self.p[idx] = np.clip(p_state, self.pi_gb, self.pi_gs)

            else:
                # NEUTRAL
                self.q[idx] = 0.0
                self.p[idx] = 0.5 * (self.pi_gb + self.pi_gs)

        # --------------------------------------------------
        # 2) Clearing in full NxN matrices
        # --------------------------------------------------
        self._clear_market_full()

        # --------------------------------------------------
        # 3) Rewards
        # --------------------------------------------------
        norm_payoffs, payoffs = self._compute_payoffs_and_metrics()

        total_p2p    = float(np.sum(self.P))
        total_import = float(np.sum(self.grid_import))
        total_export = float(np.sum(self.grid_export))
        price_range  = self.pi_gs - self.pi_gb
        
        if self.welfare_mode == "none":
            # Original behaviour: payoff + optional beta-sharing
            rewards = {
                aid: float(payoffs[aid])
                for aid in self.possible_agents
            }
        
        else:
            # Single social metric mode
            METRIC_FNS = {
                "utilitarian":        lambda: r_utilitarian_welfare(norm_payoffs),
                "rawlsian":           lambda: r_rawlsian_maximin(norm_payoffs),
                "gini":               lambda: r_gini_fairness(norm_payoffs),
                "jain":               lambda: r_jain_fairness(norm_payoffs),
                "proportional":       lambda: r_proportional_fairness(norm_payoffs),
                "grid_independence":  lambda: r_grid_independence(
                    total_p2p, total_import, total_export, self.possible_agents
                ),
                "demand_satisfaction": lambda: r_demand_satisfaction(
                    self.P, self.grid_import, self.cap, self.role, self.possible_agents
                ),
                "price_stability":    lambda: r_price_stability(
                    self.M, self.P, price_range, self.possible_agents
                ),
                "envy":               lambda: r_envy_penalty(norm_payoffs),
            }
            social_r = METRIC_FNS[self.welfare_mode]()
            rewards = {
                aid: float(norm_payoffs[aid] + social_r[aid])
                for aid in self.possible_agents
            }
        
        # ── Logging social metrics in info dict (always, for analysis) ──
        welfare_metrics = {
            "gini":               m_gini(np.array(list(norm_payoffs.values()))),
            "jain":               m_jain_index(np.array(list(norm_payoffs.values()))),
            "grid_independence":  m_grid_independence(total_p2p, total_import, total_export),
            "price_volatility":   m_price_volatility(self.M, self.P),
            "mean_envy":          m_envy(norm_payoffs),
            "utilitarian_welfare": float(np.mean(list(norm_payoffs.values()))),
            "rawlsian_welfare":   float(np.min(list(norm_payoffs.values()))),
        }

        # --------------------------------------------------
        # 5) Advance env time
        # --------------------------------------------------
        next_step_count = self.step_count + 1
        done = next_step_count >= self.max_steps

        # Hour changes for NEXT observation.
        if not done:
            next_hour = self._hour_from_step(next_step_count)
            if next_hour != self.current_hour:
                # Important:
                # when roles may change, preserving q/p across hours is usually nonsense.
                # We reset quotes at each hour transition.
                self._set_hour(next_hour, reset_quotes=False)

        self.step_count = next_step_count
        self._update_state()

        terminateds = {aid: done for aid in self.possible_agents}
        truncateds = {aid: False for aid in self.possible_agents}
        terminateds["__all__"] = done
        truncateds["__all__"] = False

        obs = self._build_obs()

        infos = {
            aid: {
                # "mu": float(self.mu),
                "hour_index": int(transition_hour),
                "next_hour_index": int(self.current_hour),
                "role": self._role_str_idx(self.agent_name_to_idx[aid]),
                "payoff": payoffs,
                "total_p2p": float(total_p2p),
                "total_import": float(total_import),
                "total_export": float(total_export),
                "P": np.array(self.P, dtype=np.float32),
                "M": np.array(self.M, dtype=np.float32),
                "q": np.array(self.q, dtype=np.float32),
                "p": np.array(self.p, dtype=np.float32),
                "cap": np.array(self.cap, dtype=np.float32),
                "role_vec": np.array(self.role, dtype=np.int8),
            }
            for aid in self.possible_agents
        }

        if done:
            self._log_custom_metrics(
                {
                    "episode": int(self.episode_id),
                    "hour_index": int(transition_hour),
                    # "mu": float(self.mu),
                    "total_p2p": float(total_p2p),
                    "total_import": float(total_import),
                    "total_export": float(total_export),
                    "num_sellers": int(np.sum(self.role == 1)),
                    "num_buyers": int(np.sum(self.role == -1)),
                }
            )

        return obs, rewards, terminateds, truncateds, infos

    # =====================================================================
    # Observation
    # =====================================================================

    def _build_obs(self) -> Dict[str, np.ndarray]:
        D = np.array([ag.D for ag in self.agents_all], dtype=np.float32)
        G = np.array([ag.G for ag in self.agents_all], dtype=np.float32)

        hour_norm = 0.0 if self.num_hours <= 1 else float(self.current_hour) / float(self.num_hours - 1)
        step_norm = float(self.step_count) / float(max(1, self.max_steps))

        global_vec = np.concatenate(
            [
                # self.role.astype(np.float32),
                # self.cap,
                # self.q,
                # self.p,
                # D,
                # G,
                self.grid_import,
                self.grid_export,
                np.array([hour_norm, step_norm], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        obs: Dict[str, np.ndarray] = {}

        for idx, aid in enumerate(self.possible_agents):
            prof = self.agents_all[idx]
            sold_p2p = float(np.sum(self.P[idx, :]))
            bought_p2p = float(np.sum(self.P[:, idx]))

            is_seller = 1.0 if self.role[idx] == 1 else 0.0
            is_buyer = 1.0 if self.role[idx] == -1 else 0.0
            is_neutral = 1.0 if self.role[idx] == 0 else 0.0
            idx_norm = float(idx) / float(max(1, self.n_agents - 1))
            net = float(prof.G - prof.D)

            feat = np.array(
                [
                    is_seller,
                    is_buyer,
                    is_neutral,
                    idx_norm,
                    float(prof.D),
                    float(prof.G),
                    float(net),
                    float(self.cap[idx]),
                    float(self.q[idx]),
                    float(self.p[idx]),
                    sold_p2p,
                    bought_p2p,
                    float(self.grid_export[idx]),
                    float(self.grid_import[idx]),
                    float(prof.a),
                    float(prof.b),
                    float(prof.c),
                ],
                dtype=np.float32,
            )

            obs[aid] = np.concatenate([global_vec, feat], axis=0).astype(np.float32)

        return obs

    # =====================================================================
    # Market clearing and rewards
    # =====================================================================

    def _pair_price(self, ask: float, bid: float) -> float:
        if self.pair_pricing_rule == "ask":
            return float(ask)
        if self.pair_pricing_rule == "bid":
            return float(bid)
        return float(0.5 * (ask + bid))

    def _clear_market_full(self) -> None:
        self.P.fill(0.0)
        self.M.fill(0.0)
        self.grid_import.fill(0.0)
        self.grid_export.fill(0.0)

        s_idx = self.current_seller_idx
        b_idx = self.current_buyer_idx


        offer_q = self.q[s_idx].astype(np.float32).copy()
        ask_p = self.p[s_idx].astype(np.float32).copy()
        bid_q = self.q[b_idx].astype(np.float32).copy()
        bid_p = self.p[b_idx].astype(np.float32).copy()

        rem_s = offer_q.copy()
        rem_b = bid_q.copy()

        s_locals = list(range(len(ask_p)))
        random.shuffle(s_locals)
        s_order_local = sorted(s_locals, key=lambda i: ask_p[i])

        b_locals = list(range(len(bid_p)))
        random.shuffle(b_locals)
        b_order_local = sorted(b_locals, key=lambda i: -bid_p[i])

        for bi_local in b_order_local:
            if rem_b[bi_local] <= self.eps:
                continue

            for sj_local in s_order_local:
                if rem_s[sj_local] <= self.eps:
                    continue
                if bid_p[bi_local] + self.eps < ask_p[sj_local]:
                    continue

                qty = float(min(rem_b[bi_local], rem_s[sj_local]))
                if qty <= self.eps:
                    continue

                sj = s_idx[sj_local]
                bi = b_idx[bi_local]

                self.P[sj, bi] += qty
                self.M[sj, bi] = self._pair_price(float(ask_p[sj_local]), float(bid_p[bi_local]))

                rem_s[sj_local] -= qty
                rem_b[bi_local] -= qty

                if rem_b[bi_local] <= self.eps:
                    break

        # Slack-node settlement
        for local_i, global_i in enumerate(b_idx):
            bought_p2p = float(np.sum(self.P[:, global_i]))
            self.grid_import[global_i] = max(float(self.cap[global_i]) - bought_p2p, 0.0)

        for local_j, global_j in enumerate(s_idx):
            sold_p2p = float(np.sum(self.P[global_j, :]))
            leftover = max(float(self.cap[global_j]) - sold_p2p, 0.0)
            if leftover > self.eps:
                self.grid_export[global_j] = leftover

    def _compute_payoffs_and_metrics(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        payoffs = {aid: 0.0 for aid in self.possible_agents}
        norm_payoffs = {aid: 0.0 for aid in self.possible_agents}

        for idx, aid in enumerate(self.possible_agents):
            prof = self.agents_all[idx]

            if self.role[idx] == 1:
                # SELLER
                sold_p2p = float(np.sum(self.P[idx, :]))
                export = float(self.grid_export[idx])
                dispatched = sold_p2p + export

                cost = float(prof.a * dispatched ** 2 + prof.b * dispatched + prof.c)
                p2p_revenue = float(np.sum(self.M[idx, :] * self.P[idx, :]))
                export_revenue = float(self.lambda_sell * export)
                revenue = p2p_revenue + export_revenue
                payoff = float(revenue - cost)

                min_payoff = self.lambda_sell * self.cap[idx] - float(prof.a * (self.cap[idx] ** 2) + prof.b * self.cap[idx] + prof.c)
                max_payoff = self.cap[idx] * self.pi_gs - float(prof.a * (self.cap[idx] ** 2) + prof.b * self.cap[idx] + prof.c)
                norm_payoff = (payoff - min_payoff) / (max_payoff - min_payoff)

                payoffs[aid] = payoff
                norm_payoffs[aid] = norm_payoff

            elif self.role[idx] == -1:
                # BUYER
                bought_p2p = float(np.sum(self.P[:, idx]))
                bought_grid = float(self.grid_import[idx])
                consumed = bought_p2p + bought_grid

                p2p_payment = float(np.sum(self.M[:, idx] * self.P[:, idx]))
                grid_payment = float(self.lambda_buy * bought_grid)
                payment = p2p_payment + grid_payment
                baseline_cost = float(self.lambda_buy * self.cap[idx])

                payoff = (baseline_cost - payment)
                min_payoff = 0
                max_payoff = baseline_cost - self.pi_gb * self.cap[idx]
                norm_payoff = (payoff - min_payoff) / (max_payoff - min_payoff) 

                payoffs[aid] = payoff
                norm_payoffs[aid] = norm_payoff

            else:
                # NEUTRAL
                payoffs[aid] = 0.0
                norm_payoffs[aid] = 0.0

        return norm_payoffs, payoffs
    # =====================================================================
    # Hourly logic
    # =====================================================================

    def _resolve_json_path(self, json_path: Optional[str]) -> str:
        candidates = [] 
        if json_path:
            candidates.append(str(json_path))

        candidates.extend(
            [
                "profiles/agents_profiles.json",
                "agents_profiles.json",
                "/mnt/data/agents_profiles.json",
                "/workspace/agents_profiles.json",
            ]
        )

        for p in candidates:
            if os.path.exists(p):
                return p

        raise FileNotFoundError(f"Could not find agents JSON. Tried: {candidates}")

    def _load_agents(self, json_path: str) -> List[AgentProfile]:
        with open(json_path, "r") as f:
            data = json.load(f)

        agents: List[AgentProfile] = []
        for name, p in data.items():
            cons = np.asarray(p["consumer_profile"], dtype=np.float32).reshape(-1)
            gen = np.asarray(p["generator_profile"], dtype=np.float32).reshape(-1)

            if cons.size == 0 or gen.size == 0:
                raise ValueError(f"{name}: empty profile")
            if cons.size != gen.size:
                raise ValueError(f"{name}: consumer/generator profile lengths do not match")

            a, b, c = [float(x) for x in p["cost_params"]]

            agents.append(
                AgentProfile(
                    name=name,
                    consumer_profile=cons,
                    generator_profile=gen,
                    a=a,
                    b=b,
                    c=c,
                    D=float(cons[0]),
                    G=float(gen[0]),
                )
            )

        return agents

    def _validate_profile_horizon(self, agents: List[AgentProfile]) -> int:
        lengths = {len(ag.consumer_profile) for ag in agents} | {len(ag.generator_profile) for ag in agents}
        if len(lengths) != 1:
            raise ValueError(f"All profiles must have same horizon length. Got lengths={sorted(lengths)}")

        num_hours = int(next(iter(lengths)))
        if num_hours <= 0:
            raise ValueError("Profile horizon must be >= 1")

        return num_hours

    def _hour_from_step(self, step_idx: int) -> int:
        raw_hour = int(step_idx // self.steps_per_hour)
        if self.hour_mode == "wrap":
            return raw_hour % self.num_hours
        return min(raw_hour, self.num_hours - 1)

    def _set_hour(self, hour_idx: int, reset_quotes: bool = True) -> None:
        self.current_hour = int(hour_idx)

        self.current_seller_idx = []
        self.current_buyer_idx = []

        for idx, ag in enumerate(self.agents_all):
            ag.D = float(self._noisy_D[idx, self.current_hour])
            ag.G = float(self._noisy_G[idx, self.current_hour])

            net = ag.G - ag.D

            if net > self.eps:
                self.role[idx] = 1
                self.cap[idx] = float(net)
                self.current_seller_idx.append(idx)
            elif net < -self.eps:
                self.role[idx] = -1
                self.cap[idx] = float(-net)
                self.current_buyer_idx.append(idx)
            else:
                self.role[idx] = 0
                self.cap[idx] = 0.0

        if reset_quotes:
            # Reset market messages each hour.
            # Preserving the previous hour's quotes is usually nonsense if role/cap changed.
            self.q = 0.5 * self.cap.astype(np.float32)
            self.p.fill(0.5 * (self.pi_gb + self.pi_gs))

            # Better initial prices by role
            for idx in self.current_seller_idx:
                prof = self.agents_all[idx]
                if self.q[idx] > self.eps:
                    cost = float(prof.a * (self.q[idx] ** 2) + prof.b * self.q[idx] + prof.c)
                    min_cost = float(max(cost / self.q[idx], self.pi_gb))
                else:
                    min_cost = float(self.pi_gb)
                self.p[idx] = min_cost

            for idx in self.current_buyer_idx:
                self.p[idx] = float(self.pi_gs)

        self.P.fill(0.0)
        self.M.fill(0.0)
        self.grid_import.fill(0.0)
        self.grid_export.fill(0.0)
        # self.mu = float(0.5 * (self.pi_gb + self.pi_gs))

    # =====================================================================
    # Aux
    # =====================================================================

    def _sample_noisy_profiles(self) -> None:
        """Build per-episode noisy D/G arrays from the original profiles.

        If profile_noise_std == 0 the originals are used unchanged.
        Noise is multiplicative and relative: noisy = original * (1 + std * noise),
        then clipped to zero so values stay non-negative.
        """
        orig_D = np.array([ag.consumer_profile for ag in self.agents_all], dtype=np.float32)
        orig_G = np.array([ag.generator_profile for ag in self.agents_all], dtype=np.float32)

        if self.profile_noise_std <= 0.0:
            self._noisy_D = orig_D
            self._noisy_G = orig_G
            return

        shape = (self.n_agents, self.num_hours)
        if self.profile_noise_type == "gaussian":
            noise_D = np.random.randn(*shape).astype(np.float32)
            noise_G = np.random.randn(*shape).astype(np.float32)
        else:  # uniform
            noise_D = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            noise_G = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)

        self._noisy_D = np.clip(orig_D * (1.0 + self.profile_noise_std * noise_D), 0.0, None)
        self._noisy_G = np.clip(orig_G * (1.0 + self.profile_noise_std * noise_G), 0.0, None)

    def _role_str_idx(self, idx: int) -> str:
        if self.role[idx] == 1:
            return "seller"
        if self.role[idx] == -1:
            return "buyer"
        return "neutral"

    def _update_state(self) -> None:
        self.state = {}

        for idx, aid in enumerate(self.possible_agents):
            sold_p2p = float(np.sum(self.P[idx, :]))
            bought_p2p = float(np.sum(self.P[:, idx]))

            avg_sell_price = (
                float(np.sum(self.M[idx, :] * self.P[idx, :]) / sold_p2p)
                if sold_p2p > self.eps else 0.0
            )
            avg_buy_price = (
                float(np.sum(self.M[:, idx] * self.P[:, idx]) / bought_p2p)
                if bought_p2p > self.eps else 0.0
            )

            if self.role[idx] == 1:
                realized = sold_p2p + float(self.grid_export[idx])
                avg_price = avg_sell_price
            elif self.role[idx] == -1:
                realized = bought_p2p + float(self.grid_import[idx])
                avg_price = avg_buy_price
            else:
                realized = 0.0
                avg_price = 0.0

            self.state[aid] = [
                float(self.q[idx]),
                float(self.p[idx]),
                float(realized),
                float(avg_price),
                float(self.role[idx]),
            ]

    def _log_custom_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.enable_csv_log:
            return

        file_exists = os.path.exists(self.custom_log_path)
        with open(self.custom_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def render(self):
        print(
            f"[episode={self.episode_id}] "
            f"step={self.step_count} hour={self.current_hour} "
            f"sellers={int(np.sum(self.role == 1))} "
            f"buyers={int(np.sum(self.role == -1))} "
            # f"mu={self.mu:.3f} total_p2p={float(np.sum(self.P)):.3f}"
        )