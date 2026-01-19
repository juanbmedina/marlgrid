# training/rl_modules.py
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

def get_energy_rl_module_spec():
    return MultiRLModuleSpec(
        rl_module_specs={
            "seller_policy": RLModuleSpec(),
            "buyer_policy": RLModuleSpec(),
        }
    )
