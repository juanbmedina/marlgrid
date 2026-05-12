#!/bin/bash
# check_status.sh
#
# Quick remote-status check, no rsync. Tells you whether the training
# is RUNNING, DONE (and exit code), or NOT FOUND.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_marlgrid_config.sh"

echo "=== Remote status of MARLGrid job ==="
echo "    server: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "    tmux:   ${TMUX_SESSION}"
echo ""

ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "
  set -u
  if tmux has-session -t ${TMUX_SESSION} 2>/dev/null; then
    echo 'STATE: RUNNING'
    echo ''
    echo '--- Last 25 lines of log (${LOG_FILE}) ---'
    if [ -f ${LOG_FILE} ]; then
      tail -25 ${LOG_FILE}
    else
      echo '(no log file yet)'
    fi
    echo ''
    echo '--- Docker container ---'
    docker ps --filter name=${CONTAINER_NAME} \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  elif [ -f ${DONE_FILE} ]; then
    EXITCODE=\$(cat ${DONE_FILE})
    if [ \"\$EXITCODE\" = \"0\" ]; then
      echo \"STATE: DONE (exit 0) — ready to fetch_results.sh\"
    else
      echo \"STATE: DONE (exit \$EXITCODE) — something failed\"
      echo ''
      echo '--- Last 40 lines of log ---'
      [ -f ${LOG_FILE} ] && tail -40 ${LOG_FILE}
    fi
    echo ''
    echo '--- progress.csv files on server ---'
    find ${REMOTE_BASE}/exp_results -name 'progress.csv' 2>/dev/null | sort || true
  else
    echo 'STATE: NOT FOUND'
    echo '  No tmux session, no .run_done sentinel.'
    echo '  Either the job was never submitted, or it was killed externally.'
  fi
"