from marllib import marl
from train_utils import clean_folder, rename_and_move_result, copy_config_file
import energy_wrapper
import os

FOLDER_TO_CLEAN = 'exp_results'
TRAINING_OUTPUT_DIR = 'exp_results/maa2c_mlp_energy_market'  # where results are generated
DESTINATION_ROOT = '/workspace/energy_marl_project/trained_policies/maa2c_mlp_energy_market'  # Where experiments are stored
EXP_NAME = 'energy_market_simple'  # Change this to your experiment name
CONFIG_FILE = 'p2p_energy.yaml'
CONFIG_DIR = 'config/env_config'

clean_folder(FOLDER_TO_CLEAN)
copy_config_file(CONFIG_FILE, config_dir=CONFIG_DIR)

# Step 1: Create your custom environment
env = marl.make_env(environment_name="p2p_energy", map_name="P2PEnergyEnv", force_coop=False)

#Step 2: Initialize the algorithm and load hyperparameters
algorithm = marl.algos.maa2c(hyperparam_source="common")

#customize model
model = marl.build_model(env, algorithm, {"core_arch": "mlp", "encode_layer": "128-128"})

#start learning
algorithm.fit(env, 
          model, 
          stop={'timesteps_total': 200000}, 
          local_mode=False, 
          num_gpus=1,
          num_workers=10,
          share_policy='all', 
          checkpoint_freq=50)


destination_path = os.path.join(DESTINATION_ROOT, EXP_NAME)

rename_and_move_result(TRAINING_OUTPUT_DIR, DESTINATION_ROOT, EXP_NAME)