# training/experiments_registry.py
"""
Central registry of experiments.

Each entry defines OVERRIDES on top of the base ENV_CONFIG that lives in
train_ppo.py. To add a new experiment, add a new key here and that is the
ONLY file you need to touch — no edits to train_ppo.py, no edits to
energy24h_env.py, no edits to any shell script.

The remote runner iterates over EXPERIMENTS x SEEDS, so to launch a sweep
you just pass EXPERIMENTS="base,partial,shuffle" to submit_job.sh.

Convention:
  - "env_config":   dict of ENV_CONFIG keys to override
  - "notes":        free-text description, kept for reproducibility audit
                    (also recorded in run_meta.json per run)

Special name:
  - "default":      sentinel meaning "use train_ppo.py's hardcoded ENV_CONFIG
                    unchanged". DO NOT add an entry for "default" here.
"""

EXPERIMENTS = {

    # ============================================================
    # ISGT 2026 — three main scenarios
    # ============================================================
    # NOTE: keys like "observation_mode" and "shuffle_order_book" must
    # exist as config.get(...) calls in energy24h_env.py for these to
    # actually do anything. Adapt the keys below to match whatever your
    # env exposes today.

    "R1-S1": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "none",
            "norm_reward": True,
            "obs_mode": "total",
        },
        "notes": "Fully observable, none index",
    },

    "R1-S2": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "none",
            "norm_reward": True,
            "obs_mode": "partial",
        },
        "notes": "Partial observable, none index",
    },

    "R1-S3": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "none",
            "norm_reward": True,
            "obs_mode": "local",
        },
        "notes": "Partial observable, none index",
    },


    "R2-S1": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain",
            "obs_mode": "total",
        },
        "notes": "Fully observable, jain index",
    },

    "R2-S2": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain",
            "obs_mode": "partial",
        },
        "notes": "Partial observable, jain index",
    },

    "R2-S3": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain",
            "obs_mode": "local",
        },
        "notes": "Partial observable, jain index",
    },


    "R3-S1": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain-only",
            "obs_mode": "total",
        },
        "notes": "Fully observable, jain index",
    },

    "R3-S2": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain-only",
            "obs_mode": "partial",
        },
        "notes": "Partial observable, jain index",
    },

    "R3-S3": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain-only",
            "obs_mode": "local",
        },
        "notes": "Partial observable, jain index",
    },

    # ============================================================
    # Forecast-to-reality robustness (oracle / naive / dr)
    # Same cell as R2-S3 (local obs + Jain). All three share the same
    # reality at a given seed; they differ only in what training sees.
    # ============================================================

    "fr_oracle": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain",
            "norm_reward": True,
            "obs_mode": "local",
            "train_regime": "oracle",  # entrena en la reality, despliega en la reality
            "sigma_reality_gen": {"agent_0": 0.03, "agent_1": 0.05, "agent_2": 0.05,
                                  "agent_3": 0.15, "agent_4": 0.15, "agent_5": 0.0},
            "sigma_reality_dem": {"agent_0": 0.05, "agent_1": 0.09, "agent_2": 0.09,
                                  "agent_3": 0.09, "agent_4": 0.09, "agent_5": 0.09},
            "sigma_train_gen":   {"agent_0": 0.03, "agent_1": 0.05, "agent_2": 0.05,
                                  "agent_3": 0.15, "agent_4": 0.15, "agent_5": 0.0},
            "sigma_train_dem":   {"agent_0": 0.05, "agent_1": 0.09, "agent_2": 0.09,
                                  "agent_3": 0.09, "agent_4": 0.09, "agent_5": 0.09},
        },
        "notes": "Forecast-reality: clairvoyant ceiling (train on the realized day)",
    },

    "fr_naive": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain",
            "norm_reward": True,
            "obs_mode": "local",
            "train_regime": "naive",   # entrena en el pronóstico, despliega en la reality
            "sigma_reality_gen": {"agent_0": 0.03, "agent_1": 0.05, "agent_2": 0.05,
                                  "agent_3": 0.15, "agent_4": 0.15, "agent_5": 0.0},
            "sigma_reality_dem": {"agent_0": 0.05, "agent_1": 0.09, "agent_2": 0.09,
                                  "agent_3": 0.09, "agent_4": 0.09, "agent_5": 0.09},
            # sigma_train_* no se usan en 'naive' (entrena sobre el pronóstico
            # sin ruido); se dejan por simetria y para cambiar de regimen sin editar.
            "sigma_train_gen":   {"agent_0": 0.03, "agent_1": 0.05, "agent_2": 0.05,
                                  "agent_3": 0.15, "agent_4": 0.15, "agent_5": 0.0},
            "sigma_train_dem":   {"agent_0": 0.05, "agent_1": 0.09, "agent_2": 0.09,
                                  "agent_3": 0.09, "agent_4": 0.09, "agent_5": 0.09},
        },
        "notes": "Forecast-reality: floor (train on forecast, deploy on reality)",
    },

    "fr_dr": {
        "env_config": {
            "enable_csv_log": False,
            "max_steps": 96,           # 24 horas * 4 pasos por hora
            "steps_per_hour": 4,
            "hour_mode": "hold_last",
            "action_mode": "absolute", # ya no delta
            "pi_min": 60.0,
            "pi_max": 100.0,
            "lambda_sell": 50,
            "lambda_buy": 110,
            "training_mode": "individual",   # o "group" con shared_policy
            "pair_pricing_rule": "midpoint",
            "agents_json_path": "profiles/agents_profiles_24h.json",
            "welfare_mode": "jain",
            "norm_reward": True,
            "obs_mode": "local",
            "train_regime": "dr",      # entrena resampleando la nube, despliega en la reality
            "sigma_reality_gen": {"agent_0": 0.03, "agent_1": 0.05, "agent_2": 0.05,
                                  "agent_3": 0.15, "agent_4": 0.15, "agent_5": 0.0},
            "sigma_reality_dem": {"agent_0": 0.05, "agent_1": 0.09, "agent_2": 0.09,
                                  "agent_3": 0.09, "agent_4": 0.09, "agent_5": 0.09},
            "sigma_train_gen":   {"agent_0": 0.03, "agent_1": 0.05, "agent_2": 0.05,
                                  "agent_3": 0.15, "agent_4": 0.15, "agent_5": 0.0},
            "sigma_train_dem":   {"agent_0": 0.05, "agent_1": 0.09, "agent_2": 0.09,
                                  "agent_3": 0.09, "agent_4": 0.09, "agent_5": 0.09},
        },
        "notes": "Forecast-reality: domain randomization at central cloud width",
    },



}


def get_experiment(name: str) -> dict:
    """Look up an experiment config, with a helpful error if the name is unknown."""
    if name not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment '{name}'. "
            f"Available: {sorted(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]


def list_experiments() -> list:
    """Return the sorted list of registered experiment names. Used by submit_job.sh."""
    return sorted(EXPERIMENTS.keys())