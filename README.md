# MARLgrid

**Multi-Agent Reinforcement Learning for Peer-to-Peer Energy Markets**

MARLgrid is a simulation framework for training and evaluating cooperative and competitive multi-agent reinforcement learning (MARL) policies in a peer-to-peer (P2P) energy trading environment. Agents represent energy prosumers — participants that can both produce and consume electricity — and learn to trade energy within a local community while interacting with an external grid.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment: P2PEnergyEnv](#environment-p2penergyenv)
  - [Agent Profiles](#agent-profiles)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Market Clearing](#market-clearing)
  - [Reward Design](#reward-design)
- [Community Welfare Rewards](#community-welfare-rewards)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiment Pipeline](#experiment-pipeline)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [License](#license)

---

## Overview

MARLgrid models a **24-hour peer-to-peer energy market** where heterogeneous agents (households, small generators, pure consumers) learn bidding and offering strategies through reinforcement learning. The framework is built on top of [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) and supports:

- **Dynamic roles**: agents switch between seller, buyer, and neutral roles each hour based on their generation-demand surplus.
- **Multiple welfare objectives**: 9+ social reward functions drawn from welfare economics, fairness theory, and energy systems literature.
- **Flexible training modes**: individual per-agent policies or a shared group policy.
- **Configurable market mechanisms**: ask-price, bid-price, or midpoint clearing rules.
- **Reproducible experiment pipelines**: end-to-end train → evaluate → collect results via Docker and SSH.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   run_experiments.sh                  │
│          (orchestrates train → eval → pull)           │
└──────────┬──────────────────────────────┬────────────┘
           │                              │
           ▼                              ▼
   ┌───────────────┐            ┌──────────────────┐
   │  train_ppo.py │            │   evaluate.py    │
   │  (Ray Tune +  │            │  (deterministic  │
   │   PPO config) │            │   rollout + CSV) │
   └───────┬───────┘            └────────┬─────────┘
           │                             │
           ▼                             ▼
   ┌──────────────────────────────────────────────┐
   │            P2PEnergyEnv (energy24h_env.py)   │
   │  ┌────────────────────────────────────────┐  │
   │  │  Agent profiles (JSON) → dynamic roles │  │
   │  │  Market clearing (NxN P2P + grid)      │  │
   │  │  Payoff computation + normalization     │  │
   │  └────────────────┬───────────────────────┘  │
   │                   │                           │
   │                   ▼                           │
   │  ┌────────────────────────────────────────┐  │
   │  │  community_welfare_rewards.py          │  │
   │  │  (social reward shaping functions)     │  │
   │  └────────────────────────────────────────┘  │
   └──────────────────────────────────────────────┘
```

---

## Environment: P2PEnergyEnv

The core environment (`energy24h_env.py`) implements `ray.rllib.env.multi_agent_env.MultiAgentEnv` and simulates a full 24-hour P2P energy market cycle.

### Agent Profiles

Agents are defined in a JSON file (e.g., `agents_profiles_24h.json`). Each agent has:

| Field | Description |
|-------|-------------|
| `consumer_profile` | 24-element array of hourly demand (kW) |
| `generator_profile` | 24-element array of hourly generation (kW) |
| `cost_params` | Quadratic cost coefficients `[a, b, c]` where `cost = a·q² + b·q + c` |

At each hour, the **net surplus** `G - D` determines an agent's role:

- **Seller** (`G - D > ε`): has energy to offer on the P2P market.
- **Buyer** (`G - D < -ε`): needs to procure energy from the market or grid.
- **Neutral** (`|G - D| ≤ ε`): does not participate in trading.

The included sample profile defines 6 agents with diverse characteristics: baseload generators, solar-like producers, commercial consumers, and a pure consumer with zero generation.

### Observation Space

Each agent receives a concatenation of global and local features:

**Global features** (shared by all agents): grid import/export per agent, normalized hour, normalized step.

**Local features** (agent-specific, 17 dimensions): one-hot role encoding (seller/buyer/neutral), agent index, demand `D`, generation `G`, net surplus, available capacity, current offered/bid quantity, current offered/bid price, P2P sold/bought quantities, grid export/import, and cost parameters `(a, b, c)`.

### Action Space

All agents share the same action space: `Box(0, 1, shape=(2,))`.

The interpretation depends on the agent's current role:

| Role | `action[0]` | `action[1]` |
|------|-------------|-------------|
| **Seller** | Fraction of surplus to offer | Price point between unit cost floor and grid sell price |
| **Buyer** | Fraction of demand to bid for | Price point between grid buy and grid sell price |
| **Neutral** | Ignored | Ignored |

### Market Clearing

The environment uses a **priority-based bilateral matching** algorithm:

1. Buyers are sorted by descending bid price (highest willingness-to-pay first).
2. Sellers are sorted by ascending ask price (cheapest offers first).
3. For each buyer, iterate over compatible sellers (where `bid ≥ ask`) and match quantities until the buyer's demand is satisfied or no compatible sellers remain.
4. Unmatched demand is satisfied by grid imports at `λ_buy`; unmatched supply is exported to the grid at `λ_sell`.

The settlement price for each bilateral trade is determined by the `pair_pricing_rule` configuration (`"ask"`, `"bid"`, or `"midpoint"`).

### Reward Design

Payoffs are computed as follows:

**Sellers**: `revenue(P2P + grid export) − generation cost(a·q² + b·q + c)`, normalized to `[0, 1]` between the worst-case (all exported at `λ_sell`) and best-case (all sold at `π_gs`) payoffs.

**Buyers**: `baseline grid cost − actual payment(P2P + grid import)`, normalized between zero savings and maximum possible savings.

When `welfare_mode != "none"`, a social reward component is added to each agent's individual payoff.

---

## Community Welfare Rewards

The module `community_welfare_rewards.py` provides 9 social reward-shaping functions, each grounded in economic or fairness theory. All are stateless and composable.

| # | Welfare Mode | Formula / Concept | Origin |
|---|-------------|-------------------|--------|
| 1 | `utilitarian` | Mean of all normalized payoffs | Benthamite welfare economics |
| 2 | `rawlsian` | Minimum payoff across all agents | Rawls' Theory of Justice |
| 3 | `gini` | Negative Gini coefficient as penalty | Development economics |
| 4 | `jain` | Jain's Fairness Index as bonus | Telecom resource allocation (Jain et al., 1984) |
| 5 | `proportional` | Mean of log-payoffs (Nash bargaining) | Kelly (1997), cooperative game theory |
| 6 | `grid_independence` | P2P volume / total energy flow | Microgrid self-sufficiency |
| 7 | `demand_satisfaction` | Mean demand satisfaction ratio | Power systems reliability |
| 8 | `price_stability` | Negative normalized price std. dev. | Market microstructure |
| 9 | `envy` | Negative mean envy penalty | Fair division theory |

Additionally, a **composite reward builder** (`build_composite_reward`) allows mixing multiple social objectives with configurable weights and an `alpha_individual` parameter that balances selfish vs. social reward.

**Naming convention**: `r_<name>(...)` returns per-agent reward dicts; `m_<name>(...)` returns scalar metrics for logging.

---

## Training

Training uses **PPO** (Proximal Policy Optimization) via Ray Tune. The entry point is `train_ppo.py`.

Key training hyperparameters (defaults):

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Discount factor (γ) | 0.95 |
| Learning rate | 3×10⁻⁴ |
| Train batch size (per learner) | 1024 |
| Minibatch size | 128 |
| SGD epochs per iteration | 6 |
| Entropy coefficient | 0.001 |
| Network architecture | 2×128 FC + ReLU, shared value head |
| Environment runners | 8 |
| GPU learners | 1 |
| Stopping criterion | 150,000 env steps |

**Training modes**:

- `"individual"`: each agent gets its own policy (e.g., `agent_0_policy`, `agent_1_policy`, ...).
- `"group"`: all agents share a single `shared_policy`.

Checkpoints are saved every 10 iterations, with the 5 most recent kept.

**Running training**:

```bash
python -m training.train_ppo
```

---

## Evaluation

The evaluation script (`evaluate.py`) performs deterministic rollouts with trained policies and saves detailed per-step, per-agent data to CSV.

Features:

- Auto-detects the latest checkpoint in the experiment directory.
- Loads `env_config_used.json` from the trial directory for reproducibility.
- Uses `RLModule.forward_inference` for deterministic action selection.
- Exports: episode, step, hour, role, payoffs, P2P volumes, grid flows, and all welfare metrics.

**Running evaluation**:

```bash
python -m training.evaluate
```

**Environment variables** (optional overrides):

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_EXPERIMENT_DIR` | `./exp_results/energy_market_training` | Path to experiment directory |
| `EVAL_NUM_EPISODES` | 50 | Number of evaluation episodes |
| `EVAL_OUTPUT_CSV` | `evaluation_agent_states.csv` | Output CSV filename |
| `EVAL_CHECKPOINT_PATH` | *(auto-detect)* | Override specific checkpoint path |

---

## Experiment Pipeline

`run_experiments.sh` provides an end-to-end pipeline that:

1. **Syncs** the local project to a remote GPU server via `rsync`.
2. **Trains** inside a Docker container (`marlgrid` image) with GPU support.
3. **Evaluates** the best checkpoint in a fresh container.
4. **Organizes** results (renames trial folders, moves custom metrics).
5. **Pulls** results back to the local machine (light mode: CSVs only; full mode: all files including checkpoints).

**Supported algorithms**: PPO, SAC, APPO (configurable via the `ALGO` variable).

**Usage**:

```bash
# Light pull (CSVs and config only)
./run_experiments.sh light

# Full pull (all files including checkpoints)
./run_experiments.sh all
```

---

## Configuration Reference

The environment is configured via a dictionary passed to `P2PEnergyEnv`. Key parameters:

### Market Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pi_min` | 50.0 | Grid buy price floor (π_gb) |
| `pi_max` | 100.0 | Grid sell price ceiling (π_gs) |
| `lambda_buy` | π_gs | Price agents pay to import from grid |
| `lambda_sell` | π_gb | Price agents receive for grid exports |
| `pair_pricing_rule` | `"ask"` | Settlement rule: `"ask"`, `"bid"`, or `"midpoint"` |

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 96 | Total steps per episode |
| `steps_per_hour` | *inferred* | Steps per hour (e.g., 4 for 96 steps / 24 hours) |
| `hour_mode` | `"hold_last"` | What to do after last hour: `"hold_last"` or `"wrap"` |
| `profile_noise_std` | 0.0 | Multiplicative noise on demand/generation profiles |
| `profile_noise_type` | `"gaussian"` | Noise distribution: `"gaussian"` or `"uniform"` |

### Reward Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_mode` | `"payoff"` | `"payoff"` or `"welfare"` |
| `welfare_mode` | `"none"` | Social reward function (see table above) |
| `alpha_individual` | 0.6 | Balance between individual (1.0) and social (0.0) reward |
| `welfare_weights` | *see code* | Per-metric weights for `"composite"` mode |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_mode` | `"individual"` | `"individual"` (per-agent policies) or `"group"` (shared policy) |
| `agents_json_path` | *auto-detected* | Path to agent profiles JSON |

---

## Project Structure

```
marlgrid/
├── energy24h_env.py              # Core multi-agent environment
├── community_welfare_rewards.py  # Social welfare reward functions
├── train_ppo.py                  # PPO training script (Ray Tune)
├── evaluate.py                   # Deterministic evaluation + CSV export
├── run_experiments.sh            # End-to-end experiment pipeline
├── agents_profiles_24h.json      # Sample 6-agent profiles (24h)
└── reward_analysis.ipynb         # Jupyter notebook for reward analysis
```

---

## Requirements

- **Python** ≥ 3.9
- **Ray** (with RLlib) ≥ 2.x
- **PyTorch**
- **NumPy**, **Pandas**
- **Gymnasium**
- **Docker** (for the experiment pipeline)

Install core dependencies:

```bash
pip install ray[rllib] torch numpy pandas gymnasium
```

---

## Getting Started

### 1. Quick training run

```python
from energy24h_env import P2PEnergyEnv

env_config = {
    "max_steps": 96,
    "steps_per_hour": 4,
    "pi_min": 60.0,
    "pi_max": 100.0,
    "lambda_sell": 50,
    "lambda_buy": 110,
    "welfare_mode": "none",
    "training_mode": "individual",
    "agents_json_path": "agents_profiles_24h.json",
}

env = P2PEnergyEnv(env_config)
obs, info = env.reset()

# Random agent loop
for _ in range(96):
    actions = {aid: env.action_spaces[aid].sample() for aid in env.possible_agents}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    if terminateds["__all__"]:
        break
```

### 2. Train with PPO

Edit the `ENV_CONFIG` in `train_ppo.py` to choose your welfare mode and hyperparameters, then run:

```bash
python -m training.train_ppo
```

### 3. Evaluate trained policies

```bash
EVAL_NUM_EPISODES=100 python -m training.evaluate
```

### 4. Analyze results

Open `reward_analysis.ipynb` to visualize payoff distributions, welfare metrics over time, price convergence, and P2P vs. grid trade volumes.

---

## License

*To be specified.*