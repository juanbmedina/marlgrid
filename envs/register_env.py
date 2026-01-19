# envs/register_env.py
from ray.tune.registry import register_env
from envs.energy_env import P2PEnergyEnv

def register_energy_env():
    register_env(
        "energy_market_ma",
        lambda config: P2PEnergyEnv(config)
    )
