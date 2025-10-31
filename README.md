# ⚡ MARLGrid: Multi-Agent Reinforcement Learning Framework for Peer-to-Peer Energy Markets

MARLGrid is a custom reinforcement learning (RL) framework designed for **multi-agent energy trading** in **peer-to-peer (P2P) electricity markets**.  
It extends **PettingZoo**, **Gym**, and **MARLlib** to simulate interactions between multiple **buyers** and **sellers**, enabling research on **cooperative** and **competitive** MARL algorithms in energy systems.

---

## 🧩 Project Structure

```
MARLGrid/
│
├── energy_agent.py        # Defines the EnergyAgent class (sellers & buyers)
├── energy_env.py          # Implements the PettingZoo-based energy market environment
├── energy_wrapper.py      # Integrates the environment with MARLlib and RLlib
├── train_energy.py        # Training script using MARLlib (MAA2C example)
└── profiles/
    └── agents_profiles.json  # Example file defining generator/consumer profiles
```

---

## ⚙️ 1. EnergyAgent (`energy_agent.py`)

The `EnergyAgent` class models each participant (seller or buyer) in the market.

### Main Features
- Defines **consumption and generation profiles** for each agent.
- Automatically determines whether the agent acts as a **seller (S)** or **buyer (B)** at each timestep.
- Maintains state variables like **power**, **price**, and **net energy**.
- Computes:
  - **Utility** (based on consumption satisfaction)
  - **Generation cost** using quadratic coefficients *(a, b, c)*
  - **Rewards** for sellers and buyers
  - **Wellness function** combining utility, cost, and competition
  - **Constraint checking** to ensure valid power/price limits

---

## 🌐 2. P2PEnergyEnv (`energy_env.py`)

Implements the **multi-agent energy market environment** using the **PettingZoo Parallel API**.

### Overview
Each episode simulates a **market operation** where sellers allocate power and buyers adjust prices to maximize their wellness while satisfying system constraints.

### Core Features
- Automatic **agent initialization** from `agents_profiles.json`
- Dynamic **state and action spaces**
- Reward system includes:
  - **Wellness-based reward**
  - **Constraint satisfaction bonus**
  - **Cost fairness penalty**
- CSV logging for each episode (`market_log.csv`)

---

## 🧱 3. RLlibEnergyEnv Wrapper (`energy_wrapper.py`)

This module wraps `P2PEnergyEnv` for **integration with MARLlib** and **Ray RLlib**.

- Adapts observation and action spaces for MARLlib.
- Registers environment in MARLlib under `"p2p_energy"`.
- Defines policy mapping between **sellers** and **buyers**.

---

## 🤖 4. Training Script (`train_energy.py`)

Demonstrates how to train the environment using **MARLlib**’s **MAA2C** algorithm.

### Workflow
1. Create environment
2. Load algorithm and model
3. Train and save policies

---

## 📦 Installation

```bash
git clone https://github.com/juanbmedina/MARLGrid.git
cd MARLGrid
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies:
```
numpy
gym
pettingzoo
ray[rllib]
marllib
matplotlib
```

---

## 🚀 Run Experiment

```bash
python train_energy.py
```

Outputs are stored in:
```
exp_results/
market_log.csv
trained_policies/
```

---

## 🧠 Future Work

- Constrained RL for better welfare optimization  
- Renewable generation uncertainty  
- Communication between agents  
- Benchmark with QMIX, MADDPG, VDN  

---

## 🖋️ Citation

```
@software{MARLGrid2025,
  author = {juan B. Medina},
  title = {MARLGrid: Multi-Agent Reinforcement Learning Framework for Peer-to-Peer Energy Markets},
  year = {2025},
  url = {https://github.com/juanbmedina/MARLGrid}
}
```

**Author:** Juan B. Medina
**License:** MIT  
**Keywords:** Multi-Agent Reinforcement Learning, Transactive Energy, PettingZoo, MARLlib, Energy Markets