# energy_env.py
import os
import json
import numpy as np
import gymnasium as gym
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ray.rllib.env.multi_agent_env import MultiAgentEnv


@dataclass
class AgentProfile:
    name: str
    D: float              # consumption at t=0
    G: float              # generation at t=0
    a: float              # quadratic cost
    b: float              # linear cost
    c: float              # constant cost


class P2PEnergyEnv(MultiAgentEnv):
    """
    P2P market with bids + centralized clearing (double-auction style).

    Each step is a "market round":
      - Sellers submit (quantity offer, ask price)
      - Buyers  submit (quantity demand, bid price)
      - A clearing operator matches bids/asks, produces allocations P[j,i] and a pairwise settlement matrix M[j,i].
      - A report price mu is kept only as a summary statistic (VWAP of executed bilateral prices).
      - Any unmet buyer demand is served by the grid at lambda_buy.
      - Any remaining seller supply can be exported to the grid at lambda_sell, but only if ask <= lambda_sell
        (i.e., the seller is willing to sell that energy at the export price).

    Rewards are *economic payoffs* by default:
      - Seller reward: revenue - cost
      - Buyer  reward: utility(consumption) - payment

    Notes (important for learning):
      - Internal payments are not counted as "welfare"; they are transfers.
      - If your buyer utility parameters are too small relative to prices, the trivial optimum is "buy nothing".
        Set lambda_u roughly on the same scale as prices (e.g., 50–120 for pi_min/pi_max ~ 50–100).
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}

        # ---------- numeric ----------
        self.eps = float(config.get("eps", 1e-8))
        self.max_steps = int(config.get("max_steps", 100))
        self.step_count = 0

        # ---------- action increments ----------
        # We use *incremental* actions (like your previous env) to model negotiation over multiple rounds.
        self.power_step = float(config.get("power_step", 0.1))
        self.price_step = float(config.get("price_step", 1.0))

        # ---------- price ranges ----------
        self.pi_gb = float(config.get("pi_min", 50.0))
        self.pi_gs = float(config.get("pi_max", 100.0))

        # ---------- grid (slack) prices ----------
        # lambda_buy: price to import from the grid (should be relatively high)
        # lambda_sell: price paid by grid for exports (should be relatively low)
        self.lambda_buy = float(config.get("lambda_buy", self.pi_gs))
        self.lambda_sell = float(config.get("lambda_sell", self.pi_gb))

        if self.lambda_sell > self.lambda_buy + 1e-6:
            # This is usually a mistake (would create arbitrage: buy from grid and resell for profit).
            raise ValueError(f"lambda_sell ({self.lambda_sell}) must be <= lambda_buy ({self.lambda_buy}).")

        # ---------- buyer utility (concave) ----------
        # U(q) = lambda_u * q - 0.5 * theta_u * q^2   (concave if theta_u > 0)
        # Make lambda_u comparable to price scale, otherwise buyers prefer q=0.
        self.lambda_u = float(config.get("lambda_u", self.pi_gs))
        self.theta_u = float(config.get("theta_u", 1.0))

        if self.theta_u <= 0:
            raise ValueError("theta_u must be > 0 for concave utility.")

        # ---------- reward mode ----------
        # "payoff"  -> individual economic payoffs (recommended for this market)
        # "welfare" -> each agent gets the same global welfare (team game; harder credit assignment)
        self.reward_mode = str(config.get("reward_mode", "payoff")).lower()
        if self.reward_mode not in ("payoff", "welfare"):
            raise ValueError("reward_mode must be 'payoff' or 'welfare'")
        
        self.beta = float(config.get("beta", 0.8))
        self.alpha = float(config.get("alpha", 0.8))

        self.enable_csv_log = config.get("enable_csv_log", True)

        # pairwise settlement rule for each executed trade
        #   midpoint -> 0.5 * (ask + bid)
        #   ask      -> pay-as-ask
        #   bid      -> pay-as-bid
        self.pair_pricing_rule = str(config.get("pair_pricing_rule", "midpoint")).lower()
        if self.pair_pricing_rule not in ("midpoint", "ask", "bid"):
            raise ValueError("pair_pricing_rule must be 'midpoint', 'ask', or 'bid'")

        # ---------- load agent profiles JSON ----------
        json_path = config.get("agents_json_path", "profiles/agents_profiles.json")
        if not os.path.exists(json_path):
            for alt in ["agents_profiles.json", "/mnt/data/agents_profiles.json"]:
                if os.path.exists(alt):
                    json_path = alt
                    break
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Required file not found: {json_path}")

        self.agents_all: List[AgentProfile] = self._load_agents(json_path)

        # static split at t=0
        self.sellers, self.buyers = self._split_agents_static_t0(self.agents_all)
        self.num_sellers = len(self.sellers)
        self.num_buyers = len(self.buyers)
        if self.num_sellers == 0 or self.num_buyers == 0:
            raise ValueError(
                f"Need >=1 seller and >=1 buyer at t=0; got sellers={self.num_sellers}, buyers={self.num_buyers}"
            )

        # ---------- RLlib IDs (preserve your A0 / A1 prefix pattern) ----------
        self.seller_ids = [f"A0_seller_{j}" for j in range(self.num_sellers)]
        self.buyer_ids = [f"A1_buyer_{i}" for i in range(self.num_buyers)]
        self.possible_agents = self.seller_ids + self.buyer_ids
        self.agents = list(self.possible_agents)

        self.previous_utility = {}

        # group/individual policies (preserve your structure)
        self.individual_policy_ids = []
        for agent_id in self.agents:
            self.individual_policy_ids.append(f"{agent_id}_policy")
            self.previous_utility[agent_id] = 0
        self.group_policy_ids = ["A0_policy", "A1_policy"]
 

        # ---------- static net supply/demand at t=0 ----------
        # We interpret these as *caps* for bidding quantities.
        self.supply = np.array([max(s.G - s.D, 0.0) for s in self.sellers], dtype=np.float32)
        self.demand = np.array([max(b.D - b.G, 0.0) for b in self.buyers], dtype=np.float32)
        self.target = float(min(np.sum(self.supply), np.sum(self.demand)))  # kept for compatibility/logging

        # ---------- market state (bids/offers that evolve during an episode) ----------
        self.offer_q = 0.5 * self.supply.copy()  # seller offer quantities
        self.ask_p = np.full((self.num_sellers,), 0.5 * (self.pi_gb + self.pi_gs), dtype=np.float32)

        self.bid_q = 0.5 * self.demand.copy()    # buyer bid quantities (desired consumption)
        self.bid_p = np.full((self.num_buyers,), 0.5 * (self.pi_gb + self.pi_gs), dtype=np.float32)

        # ---------- cleared outcomes (what evaluation code expects) ----------
        self.P = np.zeros((self.num_sellers, self.num_buyers), dtype=np.float32)  # p2p allocations
        self.M = np.zeros((self.num_sellers, self.num_buyers), dtype=np.float32)  # bilateral settlement prices
        self.pi = self.bid_p  # preserve name pi: buyer bid prices
        self.mu = float(0.5 * (self.pi_gb + self.pi_gs))  # report price (VWAP of executed bilateral prices)

        self.grid_import = np.zeros((self.num_buyers,), dtype=np.float32)  # import per buyer
        self.grid_export = np.zeros((self.num_sellers,), dtype=np.float32)  # export per seller

        # ---------- spaces ----------
        # Actions:
        #  - seller: [delta_q, delta_price]
        #  - buyer : [delta_q, delta_price]

        self.action_mode = config.get("action_mode", "delta")  # "delta" | "absolute"
        if self.action_mode == "absolute":
            seller_act = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
            buyer_act  = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            seller_act = gym.spaces.Box(
                low=np.array([-self.power_step, -self.price_step], dtype=np.float32),
                high=np.array([ self.power_step,  self.price_step], dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )
            buyer_act = gym.spaces.Box(
                low=np.array([-self.power_step, -self.price_step], dtype=np.float32),
                high=np.array([ self.power_step,  self.price_step], dtype=np.float32),
                shape=(2,), dtype=np.float32,
            )

        self.action_spaces = {sid: seller_act for sid in self.seller_ids}
        self.action_spaces.update({bid: buyer_act for bid in self.buyer_ids})

        # Observations: global vector + agent-specific context (Option A).
        # Global part:
        #   [P_flat, bid_p (pi), bid_q, ask_p, offer_q, mu, import_total, export_total, energy_balance, mean_cost_covering, step_norm]
        # Agent part (appended; same length for all agents):
        #   [is_seller, is_buyer, role_index_norm, D0, G0, a, b, c, cap, q_self, p_self, p2p_self, grid_self, total_self]

        global_obs_dim = (
            # self.num_sellers * self.num_buyers  # P
            # + self.num_sellers * self.num_buyers  # M
            # + self.num_buyers                  # bid_p
            # + self.num_buyers                  # bid_q
            # + self.num_sellers                 # ask_p
            # + self.num_sellers                 # offer_q
            + 3                                # scalars
        )
        obs_dim = global_obs_dim + 14
        self.observation_spaces = {
            aid: gym.spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)
            for aid in self.possible_agents
        }

        # ---------- logging ----------
        # Preserve your custom log path and function.
        self.custom_log_path = "/workspace/exp_results/energy_market_training/custom_metrics.csv"
        os.makedirs(os.path.dirname(self.custom_log_path), exist_ok=True)
        self.episode_id = 0

        # for evaluation logging (you used env.state[aid])
        self.state: Dict[str, object] = {}

    # ---------------- RLlib API ----------------
    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.episode_id += 1

        # reset bids/offers
        self.offer_q = 0.5 * self.supply.copy()
        self.ask_p[:] = 0.5 * (self.pi_gb + self.pi_gs)

        self.bid_q = 0.5 * self.demand.copy()
        self.bid_p[:] = 0.5 * (self.pi_gb + self.pi_gs)
        # self.bid_p[:] = random.uniform(self.pi_gb, self.pi_gs)
        # self.bid_p[:] =  self.pi_gs

        # reset cleared outcomes
        self.P[:] = 0.0
        self.M[:] = 0.0
        # self.mu = float(0.5 * (self.pi_gb + self.pi_gs))
        self.mu = self.pi_gb
        self.grid_import[:] = 0.0
        self.grid_export[:] = 0.0

        obs = self._build_obs(energy_balance=0.0, mean_cost_covering=0.0)
        infos = {aid: {} for aid in self.possible_agents}
        return obs, infos

    def step(self, action_dict):
        self.step_count += 1

        # 1) apply actions
        for j, sid in enumerate(self.seller_ids):
            if sid not in action_dict:
                continue

            a = np.asarray(action_dict[sid], dtype=np.float32).reshape(2)

            if self.action_mode == "absolute":
                a_q =a[0]
                a_p = a[1]

                # cantidad absoluta
                self.offer_q[j] = a_q * float(self.supply[j])

                s = self.sellers[j]
                if self.offer_q[j] > self.eps:
                    cost = float(s.a * (self.offer_q[j] ** 2) + s.b * self.offer_q[j] + s.c)
                    u_cost = cost / self.offer_q[j]   # $/kWh aprox
                    min_cost = float(max(u_cost, self.pi_gb))
                else:
                    min_cost = float(self.pi_gb)

                # precio absoluto en [min_cost, pi_gs]
                self.ask_p[j] = min_cost + a_p * (float(self.pi_gs) - min_cost)

            else:
                # tu lógica incremental actual
                dq = float(a[0]); dp = float(a[1])
                self.offer_q[j] = np.clip(self.offer_q[j] + dq, 0.0, float(self.supply[j]))
                s = self.sellers[j]
                cost = float(s.a * (self.offer_q[j] ** 2) + s.b * self.offer_q[j] + s.c)
                if self.offer_q[j] > self.eps:
                    u_cost = cost / self.offer_q[j]
                else:
                    u_cost = s.c
                min_cost = float(max(u_cost, self.pi_gb))
                self.ask_p[j] = np.clip(self.ask_p[j] + dp, min_cost, self.pi_gs)


        for i, bid in enumerate(self.buyer_ids):
            if bid not in action_dict:
                continue

            a = np.asarray(action_dict[bid], dtype=np.float32).reshape(2)

            if self.action_mode == "absolute":
                a_q =a[0]
                a_p = a[1]

                self.bid_q[i] = a_q * float(self.demand[i])


                self.bid_p[i] = float(self.pi_gb) + a_p * (float(self.pi_gs) - float(self.pi_gb))
            else:
                dq = float(a[0]); dp = float(a[1])
                self.bid_q[i] = np.clip(self.bid_q[i] + dq, 0.0, float(self.demand[i]))
                self.bid_p[i] = np.clip(self.bid_p[i] + dp, self.pi_gb, self.pi_gs)

        # keep pi as alias to bid prices (compat with your eval)
        self.pi = self.bid_p

        # 2) market clearing (compute P matrix + bilateral settlement matrix + report price + grid settlement)
        self.P, self.M, self.mu, self.grid_import, self.grid_export = self._clear_market(
            offer_q=self.offer_q,
            ask_p=self.ask_p,
            bid_q=self.bid_q,
            bid_p=self.bid_p,
        )

        # 3) rewards
        norm_payoffs, payoffs = self._compute_payoffs_and_metrics(
            P=self.P,
            M=self.M,
            mu_report=self.mu,
            offer_q=self.offer_q,
            ask_p=self.ask_p,
            bid_q=self.bid_q,
            bid_p=self.bid_p,
            grid_import=self.grid_import,
            grid_export=self.grid_export,
        )

        rewards = {}

        payoff_vals = np.array(list(norm_payoffs.values()), dtype=np.float64)
        sum_payoff_sellers = 0
        sum_payoff_buyers = 0
        for sid in self.seller_ids:
            sum_payoff_sellers += norm_payoffs[sid]
        for bid in self.buyer_ids:
            sum_payoff_buyers += norm_payoffs[bid]

        mean_payoff_sellers = sum_payoff_sellers / self.num_sellers
        mean_payoff_buyers = sum_payoff_buyers / self.num_sellers
        
        for sid in self.seller_ids:
            # rewards[sid] = (1 - self.alpha - self.beta) * payoffs[sid] + self.alpha * mean_payoff_sellers - self.beta * mean_payoff_buyers
            rewards[sid] = (1 - self.alpha) * norm_payoffs[sid] + self.alpha * mean_payoff_buyers

        for bid in self.buyer_ids:
            rewards[bid] = (1 - self.beta) * norm_payoffs[bid] + self.beta * mean_payoff_sellers

        # 4) "energy balance" metric (should be ~0 by construction)
        total_p2p = float(np.sum(self.P))
        total_import = float(np.sum(self.grid_import))
        total_export = float(np.sum(self.grid_export))
        supply_side = total_p2p + total_import
        demand_side = float(np.sum(self.demand))
        energy_balance = float(demand_side - supply_side)  # should be ~0

        # 5) termination
        terminateds = {aid: False for aid in self.possible_agents}
        truncateds = {aid: False for aid in self.possible_agents}

        if self.step_count >= self.max_steps:
            active_trades = self.P > self.eps
            executed_prices = self.M[active_trades]
            metrics = {
                "episode": int(self.episode_id),
                "mu": float(self.mu),
                "mean_trade_price": float(np.mean(executed_prices)) if executed_prices.size else 0.0,
                "std_trade_price": float(np.std(executed_prices)) if executed_prices.size else 0.0,
                "num_active_trades": float(np.sum(active_trades)),
                "total_p2p": float(total_p2p),
                "total_import": float(total_import),
                "total_export": float(total_export),
                "energy_balance": float(energy_balance),
                "mean_bid_price": float(np.mean(self.bid_p)),
                "mean_ask_price": float(np.mean(self.ask_p)),
                "sum_bid_q": float(np.sum(self.bid_q)),
                "sum_offer_q": float(np.sum(self.offer_q)),
            }
            self._log_custom_metrics(metrics)

            for aid in self.possible_agents:
                terminateds[aid] = True

        terminateds["__all__"] = all(terminateds[aid] for aid in self.possible_agents)
        truncateds["__all__"] = False

        # 6) observations + infos
        obs = self._build_obs(energy_balance=energy_balance, mean_cost_covering=0)

        # state for your evaluator (store something readable per agent)
        self.state = {}
        for j, sid in enumerate(self.seller_ids):
            sold_p2p = float(np.sum(self.P[j, :]))
            dispatched = float(sold_p2p + self.grid_export[j])
            avg_price = float(np.sum(self.M[j, :] * self.P[j, :]) / sold_p2p) if sold_p2p > self.eps else 0.0
            self.state[sid] = [float(self.offer_q[j]), float(self.ask_p[j]), dispatched, avg_price]
        for i, bid in enumerate(self.buyer_ids):
            bought_p2p = float(np.sum(self.P[:, i]))
            received = float(bought_p2p + self.grid_import[i])
            avg_price = float(np.sum(self.M[:, i] * self.P[:, i]) / bought_p2p) if bought_p2p > self.eps else 0.0
            self.state[bid] = [float(self.bid_q[i]), float(self.bid_p[i]), received, avg_price]

        infos = {
            aid: {
                "mu": float(self.mu),
                "payoff": payoffs,
                "total_p2p": float(total_p2p),
                "total_import": float(total_import),
                "total_export": float(total_export),
                "social_welfare": float(0),
                "energy_balance": float(energy_balance),
                "mean_cost_covering": float(0),
                "P": np.array(self.P.reshape(-1)),
                "M": np.array(self.M.reshape(-1)),
                "ask_q": np.array(self.offer_q),
                "ask_p": np.array(self.ask_p),
                "bid_q": np.array(self.bid_q),
                "bid_p": np.array(self.bid_p)
            }
            for aid in self.possible_agents
        }

        return obs, rewards, terminateds, truncateds, infos

    # ---------------- Clearing ----------------
    def _clear_market(
        self,
        offer_q: np.ndarray,
        ask_p: np.ndarray,
        bid_q: np.ndarray,
        bid_p: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Greedy double-auction matching (no network constraints).
        Produces:
          - P[j, i]: bilateral allocation matrix
          - M[j, i]: bilateral settlement price for each executed trade
          - mu: report price = VWAP of executed bilateral prices
          - grid_import[i]: unmet demand served by the grid
          - grid_export[j]: leftover supply exported to the grid

        Note: this keeps the current action model intact. Agents still submit a single ask/bid each.
        Pairwise prices are created by the clearing operator, not chosen directly by agents.
        """
        P = np.zeros((self.num_sellers, self.num_buyers), dtype=np.float32)
        M = np.zeros((self.num_sellers, self.num_buyers), dtype=np.float32)

        rem_s = offer_q.astype(np.float32).copy()
        rem_b = bid_q.astype(np.float32).copy()

        s_order = list(np.argsort(ask_p))
        b_order = list(np.argsort(-bid_p))

        for i in b_order:
            if rem_b[i] <= self.eps:
                continue

            for j in s_order:

                if rem_b[i] <= self.eps:
                    break
                if rem_s[j] <= self.eps:
                    continue
                if bid_p[i] + self.eps < ask_p[j]:
                    break

                x = float(min(rem_b[i], rem_s[j]))
                if x <= self.eps:
                    continue

                if self.pair_pricing_rule == "ask":
                    trade_price = float(ask_p[j])
                elif self.pair_pricing_rule == "bid":
                    trade_price = float(bid_p[i])
                else:
                    trade_price = 0.5 * (float(ask_p[j]) + float(bid_p[i]))

                P[j, i] = x
                M[j, i] = trade_price

                rem_b[i] -= x
                rem_s[j] -= x

        traded = float(np.sum(P))
        if traded > self.eps:
            mu = float(np.sum(P * M) / traded)
        else:
            best_bid = float(np.max(bid_p)) if bid_p.size else float(0.5 * (self.pi_gb + self.pi_gs))
            best_ask = float(np.min(ask_p)) if ask_p.size else float(0.5 * (self.pi_gb + self.pi_gs))
            mu = 0.5 * (best_bid + best_ask)

        mu = float(np.clip(mu, self.pi_gb, self.pi_gs))

        p2p_received = np.sum(P, axis=0).astype(np.float32)
        grid_import = np.maximum(self.demand.astype(np.float32) - p2p_received, 0.0)

        grid_export = np.zeros((self.num_sellers,), dtype=np.float32)
        for j in range(self.num_sellers):
            rem_energy = self.supply[j]-sum(P[j, :])
            if rem_energy > self.eps:
                grid_export[j] = rem_energy

        return P, M, mu, grid_import, grid_export

    # ---------------- Payoffs + metrics ----------------
    def _compute_payoffs_and_metrics(
        self,
        P: np.ndarray,
        M: np.ndarray,
        mu_report: float,
        offer_q: np.ndarray,
        ask_p: np.ndarray,
        bid_q: np.ndarray,
        bid_p: np.ndarray,
        grid_import: np.ndarray,
        grid_export: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        norm_payoffs: Dict[str, float] = {}
        payoffs: Dict[str, float] = {}

        cost_covering_sum = 0.0
        welfare = 0.0

        # Sellers
        for j, s in enumerate(self.sellers):
            sold_p2p = float(np.sum(P[j, :]))
            sold_grid = float(grid_export[j])
            dispatched = sold_p2p + sold_grid

            cost = float(s.a * (dispatched ** 2) + s.b * dispatched + s.c)
            p2p_revenue = float(np.sum(M[j, :] * P[j, :]))
            revenue = p2p_revenue + self.lambda_sell * sold_grid

            payoff = float(revenue - cost)
            min_payoff = -s.c
            max_payoff = self.supply[j] * self.pi_gs - float(s.a * (self.supply[j] ** 2) + s.b * self.supply[j] + s.c)
            norm_payoff = (payoff - min_payoff) / (max_payoff - min_payoff)

            norm_payoffs[self.seller_ids[j]] = norm_payoff
            payoffs[self.seller_ids[j]] = payoff

        # Buyers
        for i, _b in enumerate(self.buyers):
            bought_p2p = float(np.sum(P[:, i]))
            bought_grid = float(grid_import[i])

            baseline_cost = float(self.lambda_buy * self.demand[i])
            actual_cost = float(np.sum(M[:, i] * P[:, i]) + self.lambda_buy * bought_grid)

            payoff = float(baseline_cost - actual_cost)

            min_payoff = 0
            max_payoff = baseline_cost - self.pi_gb*self.demand[i]
            norm_payoff = (payoff - min_payoff) / (max_payoff - min_payoff)
            norm_payoffs[self.buyer_ids[i]] = norm_payoff
            payoffs[self.buyer_ids[i]] = payoff

        return norm_payoffs, payoffs

    # ---------------- Obs ----------------
    def _build_obs(self, energy_balance: float = 0.0, mean_cost_covering: float = 0.0) -> Dict[str, np.ndarray]:
        # ----- global market snapshot -----
        P_flat = self.P.reshape(-1)
        M_flat = self.M.reshape(-1)
        global_vec = np.concatenate(
            [
                # P_flat,
                # M_flat,
                # self.bid_p.reshape(-1),
                # self.bid_q.reshape(-1),
                # self.ask_p.reshape(-1),
                # self.offer_q.reshape(-1),
                np.array(
                    [
                        # float(self.mu),
                        float(np.sum(self.grid_import)),
                        float(np.sum(self.grid_export)),
                        # float(energy_balance),
                        # float(mean_cost_covering),
                        float(self.step_count) / float(max(1, self.max_steps)),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)

        # Backwards-compatible mode (old behavior).
        if not getattr(self, "obs_include_agent_features", True):
            return {aid: global_vec.copy() for aid in self.possible_agents}

        # ----- agent-conditioned context (fixed length) -----
        obs: Dict[str, np.ndarray] = {}

        # Sellers
        denom_s = max(1, self.num_sellers - 1)
        for j, sid in enumerate(self.seller_ids):
            prof = self.sellers[j]
            role_index_norm = float(j) / float(denom_s)
            p2p_self = float(np.sum(self.P[j, :]))            # sold to peers
            grid_self = float(self.grid_export[j])            # exported to grid
            total_self = float(p2p_self + grid_self)

            feat = np.array(
                [
                    1.0, 0.0,                              # is_seller, is_buyer
                    role_index_norm,
                    float(prof.D), float(prof.G),          # intrinsic profiles
                    float(prof.a), float(prof.b), float(prof.c),  # cost params
                    float(self.supply[j]),                 # cap
                    float(self.offer_q[j]),                # q_self
                    float(self.ask_p[j]),                  # p_self
                    p2p_self, grid_self, total_self,
                ],
                dtype=np.float32,
            )
            obs[sid] = np.concatenate([global_vec, feat], axis=0).astype(np.float32)

        # Buyers
        denom_b = max(1, self.num_buyers - 1)
        for i, bid in enumerate(self.buyer_ids):
            prof = self.buyers[i]
            role_index_norm = float(i) / float(denom_b)
            p2p_self = float(np.sum(self.P[:, i]))          # bought from peers
            grid_self = float(self.grid_import[i])          # imported from grid
            total_self = float(p2p_self + grid_self)

            feat = np.array(
                [
                    0.0, 1.0,                              # is_seller, is_buyer
                    role_index_norm,
                    float(prof.D), float(prof.G),
                    float(prof.a), float(prof.b), float(prof.c),
                    float(self.demand[i]),                 # cap
                    float(self.bid_q[i]),                  # q_self
                    float(self.bid_p[i]),                  # p_self
                    p2p_self, grid_self, total_self,
                ],
                dtype=np.float32,
            )
            obs[bid] = np.concatenate([global_vec, feat], axis=0).astype(np.float32)

        return obs

    # ---------------- JSON + split ----------------
    def _load_agents(self, json_path: str) -> List[AgentProfile]:
        with open(json_path, "r") as f:
            data = json.load(f)

        agents: List[AgentProfile] = []
        for name, p in data.items():
            D0 = float(p["consumer_profile"][0])
            G0 = float(p["generator_profile"][0])
            a, b, c = [float(x) for x in p["cost_params"]]
            agents.append(AgentProfile(name=name, D=D0, G=G0, a=a, b=b, c=c))
        return agents

    def _split_agents_static_t0(self, agents: List[AgentProfile]) -> Tuple[List[AgentProfile], List[AgentProfile]]:
        sellers, buyers = [], []
        for ag in agents:
            if ag.D <= self.eps and ag.G > self.eps:
                sellers.append(ag)
                continue
            if ag.D <= self.eps and ag.G <= self.eps:
                continue

            gdr = ag.G / (ag.D + self.eps)
            if gdr > 1.0:
                sellers.append(ag)
            elif gdr < 1.0:
                buyers.append(ag)
        return sellers, buyers
    
    def _u01(self, x: float) -> float:
        # [-1,1] -> [0,1]
        x = float(np.clip(x, -1.0, 1.0))
        return 0.5 * (x + 1.0)

    # ---------------- Logging (PRESERVED) ----------------
    def _log_custom_metrics(self, metrics: Dict[str, float]):
        """
        Logs arbitrary metrics dictionary into custom_metrics.csv.

        metrics: dict where keys are column names.
        Example:
            {
                "episode": 3,
                "mu": 1.25,
                "g_bar": -0.02,
                "alpha_mu": 0.5
            }
        """
        if not self.enable_csv_log:
            return
        
        file_exists = os.path.exists(self.custom_log_path)

        # Ensure episode is always present
        if "episode" not in metrics:
            metrics["episode"] = int(self.episode_id)

        # Convert everything to float/int safe format
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                clean_metrics[k] = float(v)
            else:
                try:
                    clean_metrics[k] = float(v)
                except (TypeError, ValueError):
                    clean_metrics[k] = str(v)

        with open(self.custom_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(clean_metrics.keys()))

            if not file_exists:
                writer.writeheader()

            writer.writerow(clean_metrics)