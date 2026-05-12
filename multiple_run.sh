#!/bin/bash
# run_experiments_repro.sh
#
# Variant of run_experiments.sh designed for REPRODUCIBILITY ANALYSIS.
#
# Differences vs. run_experiments.sh:
#   * Creates ONE persistent Docker container and runs N seeds inside it
#     sequentially. This eliminates ASLR/memory-layout variability between
#     runs (which run_experiments.sh introduces by using --rm containers).
#   * Skips evaluation entirely.
#   * Pulls only progress.csv + env_config_used.json from each run.
#
# Usage:
#   bash run_experiments_repro.sh               # 3 seeds (42, 43, 44)
#   SEEDS="42 42 42" bash run_experiments_repro.sh   # same seed x3: TRUE
#                                                     # reproducibility test
#   SEEDS="42 43 44 45 46" bash run_experiments_repro.sh
#
# For a reproducibility check, use `SEEDS="42 42"` and compare the two
# progress.csv files — they should match bit-for-bit if the same-container
# hypothesis holds.

set -e

############################
# CONFIG
############################
REMOTE_USER=insuasti
REMOTE_HOST=10.10.10.76
REMOTE_PORT=2222

IMAGE_NAME=marlgrid
CONTAINER_NAME=marlgrid-repro-container

# Server paths
REMOTE_BASE=/home/insuasti/juan_medina/marlgrid
REMOTE_RESULTS=${REMOTE_BASE}/exp_results_repro
TRAINING_SUBDIR=energy_market_training

# Container mount
CONTAINER_WORKDIR=/workspace

# Local paths
LOCAL_BASE=$(pwd)
LOCAL_PROJECT=${LOCAL_BASE}
LOCAL_RESULTS=${LOCAL_BASE}/exp_results_repro

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOCAL_RESULTS_RUN=${LOCAL_RESULTS}/exp_results_${TIMESTAMP}

# Seeds to run. Default is 3 sequential seeds. For a pure reproducibility
# check, override with SEEDS="42 42" (same seed twice in same container).
SEEDS=${SEEDS:-"42 43 44"}

ALGO="PPO"
TRAIN_MODULE="training.train_ppo"
CONSISTENT_FOLDER_NAME="PPO_energy_market_run"

############################
# SYNC PROJECT -> SERVER
############################
echo "=== Syncing project to server ==="

rsync -avz --delete \
  --exclude-from=.rsyncignore \
  --exclude='runs_archive/' \
  --exclude='exp_results/' \
  --exclude='exp_results_repro/' \
  -e "ssh -p ${REMOTE_PORT}" \
  "${LOCAL_PROJECT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}"

############################
# RUN ALL SEEDS IN ONE CONTAINER
############################
############################
# WIPE STALE RESULTS ON SERVER (using a temp container with root)
############################
echo ""
echo "=== Wiping stale exp_results/runs_archive on server (via root container) ==="
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  'bash -s' -- "${REMOTE_BASE}" "${IMAGE_NAME}" <<'EOF_WIPE'
REMOTE_BASE="$1"
IMAGE_NAME="$2"
docker run --rm --user root \
  -v "${REMOTE_BASE}:/workspace" \
  "${IMAGE_NAME}" \
  bash -c "rm -rf /workspace/exp_results /workspace/runs_archive && \
           echo 'Server cleaned.' && \
           ls -la /workspace/ | head -20"
EOF_WIPE

echo ""
echo "=== Launching persistent container and running seeds: ${SEEDS} ==="
echo ""

# Export SEEDS so the heredoc on the remote side can see it.
# Also pass the other vars it needs.
#
# IMPORTANT: SSH does NOT preserve the quoting of arguments containing
# spaces. `ssh host bash -s -- "42 42"` arrives at the remote as
# `bash -s -- 42 42` (split into two args). To work around this, we
# encode SEEDS with commas locally and decode them back to spaces remotely.
SEEDS_CSV=$(echo "${SEEDS}" | tr ' ' ',')

set +e
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  'bash -s' -- \
    "${REMOTE_BASE}" \
    "${REMOTE_RESULTS}" \
    "${TRAINING_SUBDIR}" \
    "${CONTAINER_WORKDIR}" \
    "${IMAGE_NAME}" \
    "${CONTAINER_NAME}" \
    "${TRAIN_MODULE}" \
    "${SEEDS_CSV}" \
  <<'EOF'
set -e

# Read positional args sent from the local script.
REMOTE_BASE="$1"
REMOTE_RESULTS="$2"
TRAINING_SUBDIR="$3"
CONTAINER_WORKDIR="$4"
IMAGE_NAME="$5"
CONTAINER_NAME="$6"
TRAIN_MODULE="$7"
# Decode comma-separated seeds back into space-separated form.
SEEDS=$(echo "$8" | tr ',' ' ')

echo "  (remote) SEEDS='${SEEDS}'"
echo "  (remote) TRAIN_MODULE='${TRAIN_MODULE}'"

# Remove any leftover container from a previous invocation
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Make sure the results folder exists and is writable by the container user
mkdir -p "${REMOTE_RESULTS}"

# Single `docker run` that stays alive for the duration of ALL seeds.
# `--rm` is kept so the container is deleted when the script exits, but
# while the script runs the container persists in memory across seeds,
# which preserves ASLR and the memory allocator's state across runs.
docker run --gpus all \
  --rm \
  --shm-size=24g \
  --user root \
  --name "${CONTAINER_NAME}" \
  -v "${REMOTE_BASE}:/workspace" \
  -e "SEEDS=${SEEDS}" \
  -e "TRAIN_MODULE=${TRAIN_MODULE}" \
  -e "REMOTE_RESULTS=${REMOTE_RESULTS}" \
  -e "TRAINING_SUBDIR=${TRAINING_SUBDIR}" \
  "${IMAGE_NAME}" \
  bash -lc '
    set -u    # undefined vars are errors (catch typos fast)
    rm -rf /tmp/ray
    cd /workspace

    # Diagnostic: confirm what we received from the outer script.
    echo ""
    echo "============================================================"
    echo "  DIAGNOSTIC — values arrived to container:"
    echo "    SEEDS=\"${SEEDS}\""
    echo "    TRAIN_MODULE=\"${TRAIN_MODULE}\""
    echo ""
    echo "  Counting tokens in SEEDS (this is what the for-loop will iterate over):"
    n=0; for s in $SEEDS; do n=$((n+1)); echo "    token #$n = \"$s\""; done
    echo "    => $n total tokens"
    echo "============================================================"
    echo ""

    if [ "$n" -lt 1 ]; then
      echo "FATAL: SEEDS produced no tokens. Aborting."
      exit 1
    fi

    # Critical: wipe BOTH exp_results and runs_archive at the start. Without
    # this, accumulated PPO_* trial folders from prior runs get mixed in with
    # the current run output, and the final rsync may pull progress.csv files
    # from old trials with stale hyperparameters. We saw this in practice:
    # 7 PPO_* folders from different days inside a single run folder.
    echo "Wiping previous run outputs..."
    rm -rf /workspace/exp_results /workspace/runs_archive
    mkdir -p /workspace/exp_results
    mkdir -p /workspace/runs_archive
    ARCHIVE="/workspace/runs_archive"
    echo "  Cleaned. Starting fresh."
    echo "  /workspace/exp_results contents after wipe:"
    ls -la /workspace/exp_results/ || true
    echo ""

    run_idx=0
    for seed in $SEEDS; do
      run_idx=$((run_idx + 1))
      echo ""
      echo "==========================================================="
      echo "  Run #${run_idx}: seed=$seed  at $(date +%H:%M:%S)"
      echo "==========================================================="

      # train_ppo.py reads MARL_SEED and MARL_RUN_TAG from env.
      # We deliberately do NOT use `set -e` around the python call so that
      # if one seed crashes, we still try the remaining seeds.
      if MARL_SEED="$seed" MARL_RUN_TAG="seed${seed}_run${run_idx}" \
           python3 -m "$TRAIN_MODULE"; then
        echo "  Training of seed=$seed run=$run_idx finished OK"
      else
        rc=$?
        echo "  !!! Training of seed=$seed run=$run_idx FAILED (exit $rc) !!!"
        echo "  Continuing with next seed anyway."
        continue
      fi

      # Find what train_ppo.py produced. Tune uses the `name=` we pass it,
      # so the folder is /workspace/exp_results/energy_market_training_<tag>/.
      RUN_TAG="seed${seed}_run${run_idx}"
      PRODUCED="/workspace/exp_results/energy_market_training_${RUN_TAG}"
      if [ ! -d "$PRODUCED" ]; then
        echo "  WARN: expected folder not found: $PRODUCED"
        echo "  Listing /workspace/exp_results/ for diagnosis:"
        ls -la /workspace/exp_results/ || true
        continue
      fi

      # Pick the most recently modified PPO_* trial folder. With a clean
      # start (we wipe at script begin) there should be exactly one, but
      # we use mtime ordering as a safety net.
      TRIAL_COUNT=$(find "$PRODUCED" -maxdepth 1 -type d -name "PPO_*" | wc -l)
      if [ "$TRIAL_COUNT" -gt 1 ]; then
        echo "  WARN: found $TRIAL_COUNT PPO_* folders in $PRODUCED, expected 1."
        echo "  Folders found:"
        find "$PRODUCED" -maxdepth 1 -type d -name "PPO_*" -printf "    %T@ %p\n"
      fi
      TRIAL=$(find "$PRODUCED" -maxdepth 1 -type d -name "PPO_*" -printf "%T@ %p\n" \
              | sort -nr | head -n 1 | cut -d" " -f2-)
      if [ -n "$TRIAL" ]; then
        DEST="$PRODUCED/PPO_energy_market_run"
        if [ "$TRIAL" != "$DEST" ]; then
          # Remove any stale destination from prior failed attempts
          rm -rf "$DEST"
          mv "$TRIAL" "$DEST"
        fi
      fi

      # Move the entire run folder into the archive so the next iteration
      # cannot possibly overwrite or delete it.
      mv "$PRODUCED" "$ARCHIVE/"
      echo "  Archived run -> $ARCHIVE/$(basename $PRODUCED)"
      echo "  Done run #${run_idx} (seed=$seed) at $(date +%H:%M:%S)"
    done

    echo ""
    echo "==========================================================="
    echo "  All ${run_idx} runs done. Restoring to exp_results/..."
    echo "==========================================================="

    # Copy everything back to /workspace/exp_results so the rsync pull
    # at the end sees it all.
    rm -rf /workspace/exp_results
    mkdir -p /workspace/exp_results
    if [ -d "$ARCHIVE" ] && [ "$(ls -A $ARCHIVE)" ]; then
      cp -a "$ARCHIVE"/* /workspace/exp_results/
    fi

    echo ""
    echo "Final contents of /workspace/exp_results/:"
    ls -la /workspace/exp_results/ || true
    echo ""
    echo "progress.csv files produced:"
    find /workspace/exp_results -name "progress.csv" | sort

    # Chown all generated files back to the host user so that subsequent
    # rsync calls from the local machine can read/delete them. We detect
    # the host user by looking at the owner of /workspace itself (which is
    # the bind-mounted /home/insuasti/juan_medina/marlgrid directory).
    HOST_UID=$(stat -c "%u" /workspace)
    HOST_GID=$(stat -c "%g" /workspace)
    echo ""
    echo "Chowning generated files to UID=$HOST_UID GID=$HOST_GID ..."
    chown -R "$HOST_UID:$HOST_GID" /workspace/exp_results /workspace/runs_archive 2>/dev/null || true
    echo "Done."
  '
EOF
TRAIN_EXIT=$?
set -e

echo ""
if [ ${TRAIN_EXIT} -eq 0 ]; then
  echo "=== All trainings completed successfully ==="
else
  echo "=== Trainings interrupted or failed (exit code: ${TRAIN_EXIT}) ==="
fi

############################
# PULL ONLY REWARD-ANALYSIS FILES TO LOCAL
############################
mkdir -p "${LOCAL_RESULTS_RUN}"

echo ""
echo "=== Pulling progress.csv and env_config_used.json for each seed ==="

# train_ppo.py writes to /workspace/exp_results/energy_market_training_seed<N>/
# We pull that entire subtree but filter to only the two files we need for
# reward plotting/analysis.
REMOTE_EXP_DIR="${REMOTE_BASE}/exp_results"

rsync -avz --progress \
  -e "ssh -p ${REMOTE_PORT}" \
  --include="*/" \
  --include="progress.csv" \
  --include="env_config_used.json" \
  --include="run_meta.json" \
  --exclude="*" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EXP_DIR}/" \
  "${LOCAL_RESULTS_RUN}/"

echo ""
echo "=== Experiment completed ==="
echo "Results pulled to: ${LOCAL_RESULTS_RUN}"
echo ""
echo "Tree:"
find "${LOCAL_RESULTS_RUN}" -name "progress.csv" | sort