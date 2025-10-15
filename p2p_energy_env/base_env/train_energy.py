from marllib import marl
from train_utils import clean_folder, rename_and_move_result, copy_config_file, copy_hyperparams_file
import energy_wrapper
import os

FOLDER_TO_CLEAN = 'exp_results'
TRAINING_OUTPUT_DIR = 'exp_results/maa2c_lstm_P2PEnergyEnv'  # where results are generated
DESTINATION_ROOT = '/workspace/marlgrid/trained_policies/maa2c_lstm_P2PEnergyEnv'  # Where experiments are stored
EXP_NAME = 'energy_market'  # Change this to your experiment name
CONFIG_FILE = 'p2p_energy.yaml'
CONFIG_DIR = 'config/env_config'

HYPER_FILE = 'maa2c.yaml'
HYPER_DIR = 'hyperparams'
env_name = "p2p_energy"

clean_folder(FOLDER_TO_CLEAN)
copy_config_file(CONFIG_FILE, config_dir=CONFIG_DIR)
copy_hyperparams_file(env_name, HYPER_FILE, HYPER_DIR)

# Step 1: Create your custom environment
env = marl.make_env(environment_name="p2p_energy", map_name="P2PEnergyEnv", force_coop=True)

#Step 2: Initialize the algorithm and load hyperparameters
algorithm = marl.algos.maa2c(hyperparam_source="p2p_energy")

#customize model
model = marl.build_model(env, algorithm, {"core_arch": "lstm"})

#start learning
algorithm.fit(env, 
          model, 
          stop={'episode_reward_mean': 6e6,'timesteps_total': 1e6}, 
          local_mode=False, 
          num_gpus=1,
          num_workers=10,
          share_policy='individual', 
          checkpoint_freq=10)


destination_path = os.path.join(DESTINATION_ROOT, EXP_NAME)

rename_and_move_result(TRAINING_OUTPUT_DIR, DESTINATION_ROOT, EXP_NAME)