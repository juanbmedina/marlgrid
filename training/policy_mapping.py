# training/policy_mapping.py

def energy_policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id.startswith("seller"):
        return "seller_policy"
    elif agent_id.startswith("buyer"):
        return "buyer_policy"
