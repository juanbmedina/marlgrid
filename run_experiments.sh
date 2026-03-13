#!/bin/bash
set -e

############################
# CONFIG
############################
REMOTE_USER=insuasti
REMOTE_HOST=10.10.10.76
REMOTE_PORT=2222

IMAGE_NAME=marlgrid
CONTAINER_NAME=marlgrid-container

# Server paths
REMOTE_BASE=/home/insuasti/juan_medina/marlgrid
REMOTE_RESULTS=${REMOTE_BASE}/exp_results
TRAINING_SUBDIR=energy_market_training
TRAINING_DIR=${REMOTE_RESULTS}/${TRAINING_SUBDIR}

# Container mount
CONTAINER_WORKDIR=/workspace

# Local paths
LOCAL_BASE=$(pwd)
LOCAL_PROJECT=${LOCAL_BASE}
LOCAL_RESULTS=${LOCAL_BASE}/exp_results

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOCAL_RESULTS_RUN=${LOCAL_RESULTS}/exp_results_${TIMESTAMP}

############################
# ALGORITHM-SPECIFIC PARAMS
# Change only this block
############################
ALGO="PPO"   # PPO | SAC | APPO

case "${ALGO}" in
  PPO)
    TRAIN_MODULE="training.train_ppo"
    EVAL_MODULE="training.evaluate"
    TRIAL_GLOB="PPO_*"
    CONSISTENT_FOLDER_NAME="PPO_energy_market_run"
    ;;
  SAC)
    TRAIN_MODULE="training.train_sac"
    EVAL_MODULE="training.evaluate"
    TRIAL_GLOB="SAC_*"
    CONSISTENT_FOLDER_NAME="SAC_energy_market_run"
    ;;
  APPO)
    TRAIN_MODULE="training.train_appo"
    # If your APPO evaluation reuses PPO code, keep evaluate_ppo.
    # If you have evaluate_appo.py, change this single line only.
    EVAL_MODULE="training.evaluate"
    TRIAL_GLOB="APPO_*"
    CONSISTENT_FOLDER_NAME="APPO_energy_market_run"
    ;;
  *)
    echo "ERROR: Unsupported ALGO='${ALGO}'. Use PPO, SAC, or APPO."
    exit 1
    ;;
esac

############################
# SYNC PROJECT → SERVER
############################
echo "=== Syncing project to server ==="

rsync -avz --delete \
  --exclude-from=.rsyncignore \
  -e "ssh -p ${REMOTE_PORT}" \
  "${LOCAL_PROJECT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}"

############################
# RUN TRAINING IN CONTAINER
############################
echo "=== Running training in Docker container ==="
echo "=== Algorithm: ${ALGO} ==="
echo "=== Train module: ${TRAIN_MODULE} ==="
echo "=== Eval module: ${EVAL_MODULE} ==="
echo "=== Trial glob: ${TRIAL_GLOB} ==="
echo "=== Container name: ${CONTAINER_NAME} ==="
echo "=== Press Ctrl+C to stop training early ==="
echo ""

# Preserve original flow: continue after training even if interrupted/failed.
set +e
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  REMOTE_BASE="${REMOTE_BASE}" \
  CONTAINER_WORKDIR="${CONTAINER_WORKDIR}" \
  IMAGE_NAME="${IMAGE_NAME}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  TRAIN_MODULE="${TRAIN_MODULE}" \
  'bash -s' <<'EOF'
set -e

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run --gpus all \
  --rm \
  --shm-size=24g \
  --user root \
  --name "${CONTAINER_NAME}" \
  -v "${REMOTE_BASE}:/workspace" \
  "${IMAGE_NAME}" \
  bash -lc "
    rm -rf /tmp/ray
    cd ${CONTAINER_WORKDIR}
    python3 -m ${TRAIN_MODULE}
  "
EOF
TRAIN_EXIT=$?
set -e

echo ""
if [ ${TRAIN_EXIT} -eq 0 ]; then
  echo "=== Training completed successfully ==="
else
  echo "=== Training was interrupted or failed (exit code: ${TRAIN_EXIT}) ==="
fi

############################
# RUN EVALUATION IN NEW CONTAINER
############################
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  REMOTE_BASE="${REMOTE_BASE}" \
  REMOTE_RESULTS="${REMOTE_RESULTS}" \
  TRAINING_SUBDIR="${TRAINING_SUBDIR}" \
  TRAINING_DIR="${TRAINING_DIR}" \
  CONTAINER_WORKDIR="${CONTAINER_WORKDIR}" \
  IMAGE_NAME="${IMAGE_NAME}" \
  CONTAINER_NAME="${CONTAINER_NAME}" \
  EVAL_MODULE="${EVAL_MODULE}" \
  TRIAL_GLOB="${TRIAL_GLOB}" \
  'bash -s' <<'EOF'
set -e

find_latest_trial_dir() {
  local base_dir="$1"
  local pattern="$2"

  find "${base_dir}" -maxdepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

TRIAL_DIR=$(find_latest_trial_dir "${TRAINING_DIR}" "${TRIAL_GLOB}")

if [ -z "${TRIAL_DIR}" ] || [ ! -d "${TRIAL_DIR}" ]; then
  echo "ERROR: No trial folder found with pattern ${TRIAL_GLOB} in ${TRAINING_DIR}"
  ls -la "${TRAINING_DIR}" || true
  exit 1
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

TRIAL_BASENAME=$(basename "${TRIAL_DIR}")
echo "=== Detected trial folder: ${TRIAL_BASENAME} ==="

# Move env_config_used.json into detected trial folder.
docker run --rm --user root \
  -v "${REMOTE_BASE}:/workspace" \
  "${IMAGE_NAME}" \
  bash -lc "
    set -e
    SRC=/workspace/exp_results/env_config_used.json
    DST_DIR=/workspace/exp_results/${TRAINING_SUBDIR}/${TRIAL_BASENAME}
    DST=\${DST_DIR}/env_config_used.json

    echo '=== (container) SRC:' \$SRC
    echo '=== (container) DST_DIR:' \$DST_DIR
    echo '=== (container) DST:' \$DST

    mkdir -p \"\$DST_DIR\"

    if [ -f \"\$SRC\" ]; then
      echo '=== Moving env_config_used.json into detected trial folder ==='
      mv -f \"\$SRC\" \"\$DST\"
      ls -la \"\$DST\" || true
    elif [ -f \"\$DST\" ]; then
      echo '=== env_config_used.json already inside trial folder (ok) ==='
    else
      echo '=== ERROR: env_config_used.json not found in SRC nor DST ==='
      ls -la /workspace/exp_results || true
      ls -la \"\$DST_DIR\" || true
      exit 1
    fi
  "

CHECKPOINT_COUNT=$(find "${TRAINING_DIR}" -name 'checkpoint_*' -type d 2>/dev/null | wc -l | tr -d ' ')
echo "=== Found ${CHECKPOINT_COUNT} checkpoint(s) ==="

if [ "${CHECKPOINT_COUNT}" -le 0 ]; then
  echo "ERROR: No checkpoints found, cannot evaluate."
  exit 1
fi

docker stop "${CONTAINER_NAME}" 2>/dev/null || true

docker run --rm --gpus all \
  --shm-size=24g \
  --user root \
  --name "${CONTAINER_NAME}" \
  -v "${REMOTE_BASE}:/workspace" \
  "${IMAGE_NAME}" \
  bash -lc "
    set -e
    cd ${CONTAINER_WORKDIR}
    python3 -m ${EVAL_MODULE}
  "

echo "=== Evaluation completed ==="
EOF

############################
# RENAME FOLDER + ORGANIZE CUSTOM METRICS
############################
echo ""
echo "=== Renaming trial folder and organizing custom metrics ==="

ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  REMOTE_BASE="${REMOTE_BASE}" \
  TRAINING_DIR="${TRAINING_DIR}" \
  TRAINING_SUBDIR="${TRAINING_SUBDIR}" \
  CONSISTENT_FOLDER_NAME="${CONSISTENT_FOLDER_NAME}" \
  IMAGE_NAME="${IMAGE_NAME}" \
  TRIAL_GLOB="${TRIAL_GLOB}" \
  'bash -s' <<'EOF'
set -e

find_latest_trial_dir() {
  local base_dir="$1"
  local pattern="$2"

  find "${base_dir}" -maxdepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

if [ -d "${TRAINING_DIR}" ]; then
  TRIAL_DIR=$(find_latest_trial_dir "${TRAINING_DIR}" "${TRIAL_GLOB}")

  if [ -n "${TRIAL_DIR}" ] && [ -d "${TRIAL_DIR}" ]; then
    TRIAL_BASENAME=$(basename "${TRIAL_DIR}")
    echo "=== Found folder: ${TRIAL_BASENAME} ==="

    docker run --rm \
      --user root \
      -v "${REMOTE_BASE}:/workspace" \
      "${IMAGE_NAME}" \
      bash -lc "
        set -e
        cd /workspace

        SRC_DIR='exp_results/${TRAINING_SUBDIR}/${TRIAL_BASENAME}'
        DST_DIR='exp_results/${TRAINING_SUBDIR}/${CONSISTENT_FOLDER_NAME}'
        METRICS='exp_results/${TRAINING_SUBDIR}/custom_metrics.csv'

        if [ \"\${SRC_DIR}\" != \"\${DST_DIR}\" ]; then
          rm -rf \"\${DST_DIR}\"
          echo '=== Renaming' \"\${SRC_DIR}\" 'to' \"\${DST_DIR}\" '==='
          mv \"\${SRC_DIR}\" \"\${DST_DIR}\"
        else
          echo '=== Source and destination folder names are the same; skipping rename ==='
        fi

        if [ -f \"\${METRICS}\" ]; then
          mv \"\${METRICS}\" \"\${DST_DIR}\"
          echo '=== custom_metrics.csv moved into run folder ==='
        else
          echo '=== custom_metrics.csv not found at top level; skipping move ==='
        fi
      "

    echo "=== Files organized in: ${TRAINING_DIR}/${CONSISTENT_FOLDER_NAME} ==="
  else
    echo "=== No trial folder found to rename ==="
  fi
else
  echo "=== Training directory not found ==="
fi

docker run --rm \
  --user root \
  -v "${REMOTE_BASE}:/workspace" \
  "${IMAGE_NAME}" \
  bash -c "rm -rf /tmp/ray" 2>/dev/null || true
EOF

############################
# PULL RESULTS TO LOCAL
############################
# ------------------------------------------
# Configuration
# ------------------------------------------

PULL_MODE=${1:-light}   # options: all | light

REMOTE_PATH="${REMOTE_RESULTS}/energy_market_training/${ALGO}_energy_market_run"
LOCAL_PATH="${LOCAL_RESULTS_RUN}/energy_market_training/${ALGO}_energy_market_run"

mkdir -p "${LOCAL_PATH}"

echo ""
echo "=== Pulling results to local machine ==="
echo "Mode: ${PULL_MODE}"

if [ "$PULL_MODE" = "light" ]; then

    echo "Downloading only key files..."

    rsync -avz --progress \
      -e "ssh -p ${REMOTE_PORT}" \
      --include="*/" \
      --include="env_config_used.json" \
      --include="progress.csv" \
      --include="evaluation_agent_states.csv" \
      --exclude="*" \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
      "${LOCAL_PATH}"

else

    echo "Downloading full experiment folder..."

    rsync -avz --progress \
      -e "ssh -p ${REMOTE_PORT}" \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
      "${LOCAL_PATH}"

fi

echo ""
echo "=== Experiment completed ==="
echo "Results saved in: ${LOCAL_PATH}"