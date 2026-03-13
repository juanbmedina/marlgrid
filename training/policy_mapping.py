# training/policy_mapping.py
def policy_mode(params):

    def energy_policy_mapping_fn(agent_id, *args, **kwargs):
        
        mode = params.get("training_mode", "group")

        if mode == "group":
            if agent_id.startswith("A0_"): 
                return "A0_policy" 
            if agent_id.startswith("A1_"): 
                return "A1_policy"

        elif mode == "individual":
            return f"{agent_id}_policy"

        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    return energy_policy_mapping_fn

