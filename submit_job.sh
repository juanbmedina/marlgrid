#!/bin/bash
# submit_job.sh
#
# Sync project to server, then launch the training job inside a detached
# tmux session. Returns immediately so you can disconnect your laptop.
#
# Usage:
#   bash submit_job.sh                          # default seeds (42 43 44)
#   SEEDS="42 43 44 45 46" bash submit_job.sh
#   SEEDS="42 42" bash submit_job.sh            # repro check

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_marlgrid_config.sh"

# Seeds (override via env)
SEEDS=${SEEDS:-"42 43 44"}
# SSH does not preserve quoting of space-separated args; encode as CSV.
SEEDS_CSV=$(echo "${SEEDS}" | tr ' ' ',')

############################
# 0. Pre-flight: don't trample a running session
############################
echo "=== Checking for existing tmux session '${TMUX_SESSION}' on server ==="
if ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
     "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
  echo ""
  echo "ERROR: a tmux session named '${TMUX_SESSION}' already exists on server."
  echo "       Aborting to avoid clobbering it."
  echo ""
  echo "Options:"
  echo "  - Watch:  ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} -t 'tmux attach -t ${TMUX_SESSION}'"
  echo "    (detach with Ctrl+b then d)"
  echo "  - Kill:   ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} 'tmux kill-session -t ${TMUX_SESSION}'"
  exit 1
fi
echo "  OK, no live session."

############################
# 1. Sync project to server
############################
echo ""
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
# 2. Make remote runner executable
############################
echo ""
echo "=== Preparing remote runner ==="
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "chmod +x ${REMOTE_BASE}/_remote_runner.sh && \
   echo '  remote runner ready at ${REMOTE_BASE}/_remote_runner.sh'"

############################
# 3. Launch in detached tmux session
############################
echo ""
echo "=== Launching tmux session '${TMUX_SESSION}' on server ==="
echo "  SEEDS:        ${SEEDS}"
echo "  TRAIN_MODULE: ${TRAIN_MODULE}"
echo "  LOG:          ${LOG_FILE}"
echo ""

# We pass SEEDS as CSV (no spaces) so the inner shell quoting cannot break it.
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "tmux new-session -d -s ${TMUX_SESSION} \
     'cd ${REMOTE_BASE} && \
      bash ${REMOTE_BASE}/_remote_runner.sh ${SEEDS_CSV} ${TRAIN_MODULE} \
        > ${LOG_FILE} 2>&1'"

# Verify it actually started
sleep 2
if ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" \
     "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
  echo "  tmux session '${TMUX_SESSION}' is alive."
else
  echo "  WARN: tmux session not found after launch. Check ${LOG_FILE} on server."
  exit 1
fi

cat <<EOM

================================================================
  Job launched.  You can close your laptop now.
================================================================

Useful commands (from this same project dir):

  bash check_status.sh        Quick status (running / done / failed)
  bash fetch_results.sh       Pull progress.csv files when done

Watch live (re-attach to the tmux session):

  ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} -t 'tmux attach -t ${TMUX_SESSION}'
  (detach with Ctrl+b then d  --  do NOT type 'exit', that kills the run)

Tail the log without attaching:

  ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ${LOG_FILE}'

EOM