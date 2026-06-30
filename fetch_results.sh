#!/bin/bash
# fetch_results.sh
#
# Pull experiment results from the server.
#
# By DEFAULT pulls the FULL run folders (CSV files + model checkpoints +
# env_runner / learner_group state). A downloaded experiment is then
# fully self-contained: you can re-run evaluate.py / evaluate_legacy.py,
# inspect weights, restart Tune from the checkpoint, etc., without
# having to go back to the server (which is wiped on the next run).
#
# Set MODE=light to pull ONLY the analysis files (progress.csv,
# evaluation_agent_states.csv, env_config_used.json, run_meta.json).
# Much smaller and faster, but no model weights, so you cannot
# re-evaluate without retraining.
#
# Set SRC=archive to pull from ~/juan_medina/marlgrid/runs_archive
# (seeds that finished while training is still running). This skips
# the tmux-alive guard automatically.
#
# Refuses to run if the tmux session is still live (unless FORCE=1
# or SRC=archive).
#
# Usage:
#   bash fetch_results.sh                       # full pull from exp_results
#   MODE=light bash fetch_results.sh            # only CSVs + config JSONs
#   FORCE=1 bash fetch_results.sh               # pull even if job is running
#   SRC=archive bash fetch_results.sh           # pull finished seeds from archive
#   SRC=archive MODE=light bash fetch_results.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_marlgrid_config.sh"

MODE="${MODE:-full}"
case "$MODE" in
  full|light) ;;
  *) echo "ERROR: invalid MODE='$MODE' (use 'full' or 'light')."; exit 2 ;;
esac

SRC="${SRC:-results}"
case "$SRC" in
  results|archive) ;;
  *) echo "ERROR: invalid SRC='$SRC' (use 'results' or 'archive')."; exit 2 ;;
esac

############################
# 1. Verify state
############################
SESSION_ALIVE=$(ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "tmux has-session -t ${TMUX_SESSION} 2>/dev/null && echo yes || echo no")

DONE_EXISTS=$(ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "[ -f ${DONE_FILE} ] && echo yes || echo no")

echo "Remote state:"
echo "  tmux session alive: ${SESSION_ALIVE}"
echo "  .run_done exists:   ${DONE_EXISTS}"
echo "  pull mode:          ${MODE}"
echo "  pull source:        ${SRC}"
echo ""

if [ "${SRC}" = "archive" ]; then
  # Archive mode: designed for mid-training pulls — skip tmux guard.
  if [ "${SESSION_ALIVE}" = "yes" ]; then
    echo "INFO: training still running — pulling finished seeds from runs_archive."
  fi

  # List what's available in the archive before pulling.
  echo "=== Contents of runs_archive on server ==="
  ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "if [ -d ${REMOTE_BASE}/runs_archive ]; then
       ls -1d ${REMOTE_BASE}/runs_archive/*/ 2>/dev/null \
         | while read d; do echo \"  \$(basename \$d)\"; done
       echo ''
       echo 'progress.csv files in archive:'
       find ${REMOTE_BASE}/runs_archive -name 'progress.csv' 2>/dev/null | sort
       echo ''
       echo 'evaluation_agent_states.csv files in archive:'
       find ${REMOTE_BASE}/runs_archive -name 'evaluation_agent_states.csv' 2>/dev/null | sort
     else
       echo '  (runs_archive does not exist yet — no seeds have finished)'
     fi"
  echo ""

  # Verify the archive actually exists and is non-empty.
  ARCHIVE_EXISTS=$(ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "[ -d ${REMOTE_BASE}/runs_archive ] && [ \"\$(ls -A ${REMOTE_BASE}/runs_archive 2>/dev/null)\" ] \
     && echo yes || echo no")

  if [ "${ARCHIVE_EXISTS}" = "no" ]; then
    echo "ERROR: runs_archive is empty or does not exist on server."
    echo "       No seeds have finished yet. Try again later."
    exit 1
  fi

else
  # Default (results) mode: original guards.
  if [ "${SESSION_ALIVE}" = "yes" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
      echo "WARN: session still running, but FORCE=1 — pulling partial results."
    else
      echo "Job is still running. Refusing to pull (use FORCE=1 or SRC=archive to override)."
      echo "Run 'bash check_status.sh' to see progress."
      exit 1
    fi
  fi

  if [ "${DONE_EXISTS}" = "no" ] && [ "${SESSION_ALIVE}" = "no" ]; then
    echo "ERROR: no tmux session and no .run_done sentinel."
    echo "       The job either never ran or was killed before writing the sentinel."
    exit 1
  fi

  if [ "${DONE_EXISTS}" = "yes" ]; then
    EXITCODE=$(ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "cat ${DONE_FILE}")
    if [ "${EXITCODE}" != "0" ]; then
      echo "WARN: remote exit code was ${EXITCODE} — pulling whatever is there."
    fi
  fi
fi

############################
# 2. Pull files
############################
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOCAL_RESULTS_RUN=${LOCAL_RESULTS}/exp_results_${TIMESTAMP}
mkdir -p "${LOCAL_RESULTS_RUN}"

if [ "$SRC" = "archive" ]; then
  REMOTE_EXP_DIR="${REMOTE_BASE}/runs_archive"
else
  REMOTE_EXP_DIR="${REMOTE_BASE}/exp_results"
fi

if [ "$MODE" = "light" ]; then
  echo "=== LIGHT pull: only analysis files ==="
  echo "    progress.csv"
  echo "    env_config_used.json"
  echo "    run_meta.json"
  echo "    evaluation_agent_states.csv"
  echo ""
  echo "    NOTE: model weights are NOT pulled in light mode."
  echo "          Use the default (full) mode if you may want to re-evaluate"
  echo "          or inspect models locally."
  echo ""
  echo "    from: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EXP_DIR}/"
  echo "    to:   ${LOCAL_RESULTS_RUN}/"
  echo ""

  rsync -avz --progress \
    -e "ssh -p ${REMOTE_PORT}" \
    --include="*/" \
    --include="progress.csv" \
    --include="env_config_used.json" \
    --include="run_meta.json" \
    --include="evaluation_agent_states.csv" \
    --exclude="*" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EXP_DIR}/" \
    "${LOCAL_RESULTS_RUN}/"

else
  echo "=== FULL pull: entire run folders (including model checkpoints) ==="
  echo ""
  echo "    from: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EXP_DIR}/"
  echo "    to:   ${LOCAL_RESULTS_RUN}/"
  echo ""

  rsync -avz --progress \
    -e "ssh -p ${REMOTE_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_EXP_DIR}/" \
    "${LOCAL_RESULTS_RUN}/"
fi

############################
# 3. Pull the run.log too (small, useful for debugging)
############################
echo ""
echo "=== Pulling run.log ==="
rsync -avz \
  -e "ssh -p ${REMOTE_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${LOG_FILE}" \
  "${LOCAL_RESULTS_RUN}/run.log" 2>/dev/null \
  || echo "(no run.log on server)"

############################
# 4. Summary
############################
echo ""
echo "=== Done ==="
echo "Pull source:      ${SRC}"
echo "Pull mode:        ${MODE}"
echo "Results saved to: ${LOCAL_RESULTS_RUN}"
echo ""
echo "progress.csv files pulled:"
find "${LOCAL_RESULTS_RUN}" -name "progress.csv" | sort
echo ""
echo "evaluation_agent_states.csv files pulled:"
find "${LOCAL_RESULTS_RUN}" -name "evaluation_agent_states.csv" | sort

if [ "$MODE" = "full" ]; then
  echo ""
  echo "checkpoint folders pulled (latest per seed):"
  find "${LOCAL_RESULTS_RUN}" -maxdepth 6 -type d -name "checkpoint_*" \
    -printf "  %p\n" | sort
  echo ""
  echo "Sanity check — looking for non-empty rl_module/agent_X_policy dirs:"
  find "${LOCAL_RESULTS_RUN}" -type d -name "agent_*_policy" \
    | while read d; do
        n_files=$(find "$d" -type f | wc -l)
        echo "  $d  ($n_files files)"
      done
fi