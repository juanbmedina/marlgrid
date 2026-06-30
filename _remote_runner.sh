#!/bin/bash
# _remote_runner.sh
#
# Runs ON THE SERVER, inside a detached tmux session launched by submit_job.sh.
# Receives:
#   $1 = SEEDS_CSV       (e.g. "42,43,44")
#   $2 = EXPERIMENTS_CSV (e.g. "base,partial,shuffle"; "default" = no override)
#   $3 = TRAIN_MODULE    (e.g. "training.train_ppo")
#
# Responsibilities:
#   1. Wipe stale results.
#   2. Launch a single persistent docker container that runs all
#      (experiment, seed) combinations sequentially, and EVALUATES each
#      run immediately after its training finishes.
#   3. Always write a sentinel file ${REMOTE_BASE}/.run_done containing
#      the exit code, even on failure, so check_status.sh can detect
#      completion.

set -u

############################
# Args
############################
SEEDS_CSV="${1:-}"
EXPERIMENTS_CSV="${2:-default}"
TRAIN_MODULE="${3:-training.train_ppo}"

if [ -z "${SEEDS_CSV}" ]; then
  echo "FATAL: SEEDS_CSV (arg 1) is empty."
  exit 2
fi

# Decode CSV back into space-separated.
SEEDS=$(echo "${SEEDS_CSV}" | tr ',' ' ')
EXPERIMENTS=$(echo "${EXPERIMENTS_CSV}" | tr ',' ' ')

############################
# Config (must match _marlgrid_config.sh)
############################
REMOTE_BASE=/home/insuasti/juan_medina/marlgrid
REMOTE_RESULTS=${REMOTE_BASE}/exp_results_repro
TRAINING_SUBDIR=energy_market_training
IMAGE_NAME=marlgrid
CONTAINER_NAME=marlgrid-repro-container
DONE_FILE=${REMOTE_BASE}/.run_done

# Eval module + episodes count (override via env if needed).
EVAL_MODULE="${EVAL_MODULE:-training.evaluate}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-50}"

# Count total expected runs (for the header).
N_SEEDS=$(echo "$SEEDS" | wc -w)
N_EXPS=$(echo "$EXPERIMENTS" | wc -w)
N_TOTAL=$((N_SEEDS * N_EXPS))

echo "================================================================"
echo "  MARLGRID REMOTE RUNNER"
echo "  Started at:    $(date)"
echo "  Hostname:      $(hostname)"
echo "  SEEDS:         ${SEEDS}      (${N_SEEDS})"
echo "  EXPERIMENTS:   ${EXPERIMENTS}  (${N_EXPS})"
echo "  TOTAL RUNS:    ${N_TOTAL}"
echo "  TRAIN_MODULE:  ${TRAIN_MODULE}"
echo "  EVAL_MODULE:   ${EVAL_MODULE}"
echo "  EVAL_NUM_EPS:  ${EVAL_NUM_EPISODES}"
echo "================================================================"

# Remove stale sentinel from a previous run (if any).
rm -f "${DONE_FILE}"

############################
# 1. Wipe stale results on server (via root container)
############################
echo ""
echo "=== Wiping stale exp_results / runs_archive ==="
docker run --rm --user root \
  -v "${REMOTE_BASE}:/workspace" \
  "${IMAGE_NAME}" \
  bash -c "rm -rf /workspace/exp_results /workspace/runs_archive && \
           echo 'Server cleaned.' && \
           ls -la /workspace/ | head -20"

############################
# 2. Launch persistent training container
############################
echo ""
echo "=== Launching persistent container ==="

# Remove leftover container from previous invocation, if any.
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

mkdir -p "${REMOTE_RESULTS}"

# Important: do NOT 'set -e' before docker run — we want to always
# write the sentinel file, even on docker failure.
set +e
docker run --gpus all \
  --rm \
  --shm-size=24g \
  --user root \
  --name "${CONTAINER_NAME}" \
  -v "${REMOTE_BASE}:/workspace" \
  -e "SEEDS=${SEEDS}" \
  -e "EXPERIMENTS=${EXPERIMENTS}" \
  -e "TRAIN_MODULE=${TRAIN_MODULE}" \
  -e "EVAL_MODULE=${EVAL_MODULE}" \
  -e "EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES}" \
  -e "REMOTE_RESULTS=${REMOTE_RESULTS}" \
  -e "TRAINING_SUBDIR=${TRAINING_SUBDIR}" \
  "${IMAGE_NAME}" \
  bash -lc '
    set -u
    rm -rf /tmp/ray
    cd /workspace

    echo ""
    echo "============================================================"
    echo "  DIAGNOSTIC — values arrived to container:"
    echo "    SEEDS=\"${SEEDS}\""
    echo "    EXPERIMENTS=\"${EXPERIMENTS}\""
    echo "    TRAIN_MODULE=\"${TRAIN_MODULE}\""
    echo "    EVAL_MODULE=\"${EVAL_MODULE}\""
    echo "    EVAL_NUM_EPISODES=\"${EVAL_NUM_EPISODES}\""
    echo ""
    echo "  Tokens in SEEDS:"
    n_seeds=0; for s in $SEEDS; do n_seeds=$((n_seeds+1)); echo "    seed #$n_seeds = \"$s\""; done
    echo "  Tokens in EXPERIMENTS:"
    n_exps=0; for e in $EXPERIMENTS; do n_exps=$((n_exps+1)); echo "    exp  #$n_exps = \"$e\""; done
    echo "    => $((n_seeds * n_exps)) total (experiment, seed) runs"
    echo "============================================================"
    echo ""

    if [ "$n_seeds" -lt 1 ] || [ "$n_exps" -lt 1 ]; then
      echo "FATAL: SEEDS or EXPERIMENTS produced no tokens. Aborting."
      exit 1
    fi

    # Wipe BOTH exp_results and runs_archive before starting.
    echo "Wiping previous run outputs..."
    rm -rf /workspace/exp_results /workspace/runs_archive
    mkdir -p /workspace/exp_results
    mkdir -p /workspace/runs_archive
    ARCHIVE="/workspace/runs_archive"
    echo "  Cleaned. Starting fresh."
    echo ""

    # Counters for the per-run status report at the end.
    train_ok=0; train_fail=0
    eval_ok=0;  eval_fail=0; eval_skip=0

    run_idx=0
    # =====================================================================
    # DOUBLE LOOP: experiments x seeds
    # Outer loop = experiment (so all seeds of one experiment run together,
    # which is friendlier for monitoring than interleaving).
    # =====================================================================
    for exp in $EXPERIMENTS; do
      for seed in $SEEDS; do
        run_idx=$((run_idx + 1))

        # Build a tag that is unique per (experiment, seed, repeat).
        # "default" sentinel = no experiment override; keep the old tag style.
        if [ "$exp" = "default" ]; then
          RUN_TAG="seed${seed}_run${run_idx}"
        else
          RUN_TAG="${exp}_seed${seed}_run${run_idx}"
        fi

        echo ""
        echo "==========================================================="
        echo "  Run #${run_idx}/${n_exps}x${n_seeds}: exp=${exp}  seed=${seed}"
        echo "  Tag: ${RUN_TAG}    at $(date +%H:%M:%S)"
        echo "==========================================================="

        # ---------- TRAIN ----------
        # train_ppo.py reads MARL_SEED, MARL_RUN_TAG, MARL_EXPERIMENT_NAME from env.
        # Do NOT use set -e here: if one run crashes we still try the rest.
        if MARL_EXPERIMENT_NAME="$exp" \
           MARL_SEED="$seed" \
           MARL_RUN_TAG="$RUN_TAG" \
             python3 -m "$TRAIN_MODULE"; then
          echo "  Training of ${RUN_TAG} finished OK"
          train_ok=$((train_ok + 1))
        else
          rc=$?
          echo "  !!! Training of ${RUN_TAG} FAILED (exit $rc) !!!"
          echo "  Continuing with next run anyway."
          train_fail=$((train_fail + 1))
          continue
        fi

        PRODUCED="/workspace/exp_results/energy_market_training_${RUN_TAG}"
        if [ ! -d "$PRODUCED" ]; then
          echo "  WARN: expected folder not found: $PRODUCED"
          ls -la /workspace/exp_results/ || true
          eval_skip=$((eval_skip + 1))
          continue
        fi

        # ---------- RENAME PPO_xxxx -> PPO_energy_market_run ----------
        TRIAL_COUNT=$(find "$PRODUCED" -maxdepth 1 -type d -name "PPO_*" | wc -l)
        if [ "$TRIAL_COUNT" -gt 1 ]; then
          echo "  WARN: found $TRIAL_COUNT PPO_* folders in $PRODUCED, expected 1."
          find "$PRODUCED" -maxdepth 1 -type d -name "PPO_*" -printf "    %T@ %p\n"
        fi
        TRIAL=$(find "$PRODUCED" -maxdepth 1 -type d -name "PPO_*" -printf "%T@ %p\n" \
                | sort -nr | head -n 1 | cut -d" " -f2-)
        if [ -n "$TRIAL" ]; then
          DEST="$PRODUCED/PPO_energy_market_run"
          if [ "$TRIAL" != "$DEST" ]; then
            rm -rf "$DEST"
            mv "$TRIAL" "$DEST"
          fi
        fi

        # ---------- PRESERVE env_config + run_meta INTO run dir ----------
        # train_ppo.py writes these at /workspace/exp_results/ (root level),
        # so each run overwrites the previous one. Copy into the run folder
        # before evaluating/archiving so every run keeps its exact config.
        if [ -f /workspace/exp_results/env_config_used.json ]; then
          cp /workspace/exp_results/env_config_used.json "$PRODUCED/"
          if [ -d "$PRODUCED/PPO_energy_market_run" ]; then
            cp /workspace/exp_results/env_config_used.json \
               "$PRODUCED/PPO_energy_market_run/"
          fi
          echo "  Preserved env_config_used.json into $PRODUCED/"
        else
          echo "  WARN: /workspace/exp_results/env_config_used.json not found "
          echo "        — evaluation will fall back to env defaults."
        fi

        if [ -f /workspace/exp_results/run_meta.json ]; then
          cp /workspace/exp_results/run_meta.json "$PRODUCED/"
          if [ -d "$PRODUCED/PPO_energy_market_run" ]; then
            cp /workspace/exp_results/run_meta.json \
               "$PRODUCED/PPO_energy_market_run/"
          fi
          echo "  Preserved run_meta.json into $PRODUCED/"
        else
          echo "  WARN: /workspace/exp_results/run_meta.json not found."
        fi

        # ---------- EVALUATE just-trained run ----------
        LATEST_CKPT=$(find "$PRODUCED/PPO_energy_market_run" -maxdepth 1 \
                        -type d -name "checkpoint_*" -printf "%T@ %p\n" 2>/dev/null \
                      | sort -nr | head -n 1 | cut -d" " -f2-)

        if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
          echo ""
          echo "  --- Evaluating ${RUN_TAG} at $(date +%H:%M:%S) ---"
          echo "  Checkpoint: $LATEST_CKPT"
          echo "  Episodes:   $EVAL_NUM_EPISODES"

          if EVAL_CHECKPOINT_PATH="$LATEST_CKPT" \
             EVAL_NUM_EPISODES="$EVAL_NUM_EPISODES" \
               python3 -m "$EVAL_MODULE"; then
            echo "  Evaluation of ${RUN_TAG} finished OK"
            eval_ok=$((eval_ok + 1))
          else
            rc=$?
            echo "  !!! Evaluation of ${RUN_TAG} FAILED (exit $rc) !!!"
            echo "  Training results are still preserved; continuing."
            eval_fail=$((eval_fail + 1))
          fi
        else
          echo "  WARN: no checkpoint_* found in $PRODUCED/PPO_energy_market_run"
          echo "        — skipping evaluation for ${RUN_TAG}."
          eval_skip=$((eval_skip + 1))
        fi

        # ---------- ARCHIVE ----------
        # Move full run folder (now including evaluation CSV) into archive
        # so the next iteration cannot overwrite or delete it.
        mv "$PRODUCED" "$ARCHIVE/"
        echo "  Archived run -> $ARCHIVE/$(basename $PRODUCED)"
        echo "  Done run #${run_idx} (${RUN_TAG}) at $(date +%H:%M:%S)"
      done   # end seeds loop
    done     # end experiments loop

    echo ""
    echo "==========================================================="
    echo "  All ${run_idx} runs done. Restoring to exp_results/..."
    echo "==========================================================="

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
    echo ""
    echo "evaluation_agent_states.csv files produced:"
    find /workspace/exp_results -name "evaluation_agent_states.csv" | sort
    echo ""
    echo "env_config_used.json files preserved per run:"
    find /workspace/exp_results -name "env_config_used.json" | sort
    echo ""
    echo "run_meta.json files preserved per run:"
    find /workspace/exp_results -name "run_meta.json" | sort

    # ---------- Summary ----------
    echo ""
    echo "==========================================================="
    echo "  PER-RUN SUMMARY  (experiments x seeds = ${n_exps} x ${n_seeds} = $((n_exps*n_seeds)))"
    echo "    training:    OK=$train_ok   FAIL=$train_fail"
    echo "    evaluation:  OK=$eval_ok    FAIL=$eval_fail   SKIP=$eval_skip"
    echo "==========================================================="

    # Chown so host user can rsync/delete results from the host side.
    HOST_UID=$(stat -c "%u" /workspace)
    HOST_GID=$(stat -c "%g" /workspace)
    echo ""
    echo "Chowning generated files to UID=$HOST_UID GID=$HOST_GID ..."
    chown -R "$HOST_UID:$HOST_GID" /workspace/exp_results /workspace/runs_archive 2>/dev/null || true
    echo "Done."
  '
DOCKER_EXIT=$?
set -e

############################
# 3. Always write sentinel
############################
echo ""
echo "================================================================"
echo "  RUNNER FINISHED at: $(date)"
echo "  docker exit code: ${DOCKER_EXIT}"
echo "================================================================"

echo "${DOCKER_EXIT}" > "${DONE_FILE}"
echo "Wrote ${DONE_FILE} with exit code ${DOCKER_EXIT}"

exit ${DOCKER_EXIT}