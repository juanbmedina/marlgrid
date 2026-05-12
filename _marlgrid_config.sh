# _marlgrid_config.sh
# Shared config for submit_job.sh / check_status.sh / fetch_results.sh.
# Source it from each of those scripts; do NOT execute directly.

# --- SSH ---
REMOTE_USER=insuasti
REMOTE_HOST=10.10.10.76
REMOTE_PORT=2222

# --- Docker ---
IMAGE_NAME=marlgrid
CONTAINER_NAME=marlgrid-repro-container

# --- Server paths ---
REMOTE_BASE=/home/insuasti/juan_medina/marlgrid
REMOTE_RESULTS=${REMOTE_BASE}/exp_results_repro
TRAINING_SUBDIR=energy_market_training
CONTAINER_WORKDIR=/workspace

# --- Local paths ---
# Convention: scripts must be invoked from the project root.
LOCAL_BASE=$(pwd)
LOCAL_PROJECT=${LOCAL_BASE}
LOCAL_RESULTS=${LOCAL_BASE}/exp_results_repro

# --- Training ---
ALGO="PPO"
TRAIN_MODULE="training.train_ppo"

# --- tmux / sentinel files ---
TMUX_SESSION=marlgrid
DONE_FILE=${REMOTE_BASE}/.run_done
LOG_FILE=${REMOTE_BASE}/run.log