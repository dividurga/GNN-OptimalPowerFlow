#!/usr/bin/env zsh
# Train PI-GNN (MPN + mixed MSE/PowerImbalance loss) on 14Bus + 57Bus
# outage datasets (500 epochs, patience 40), matching run_training_14_57.sh.
# Launch:  nohup caffeinate -s ./run_training_14_57_pi.sh </dev/null >logs/nohup_pi.out 2>&1 &
# Monitor: tail -f logs/master_pi.log
# Summary: cat logs/summary_pi.log

set -o pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

eval "$(conda shell.zsh hook)"
conda activate cos324

LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_pi.log"
SUMMARY_LOG="$LOG_DIR/summary_pi.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

> "$MASTER_LOG"
> "$SUMMARY_LOG"

log "===== PI-GNN training: 14Bus + 57Bus (MPN + mixed loss, 500 epochs, patience 40) ====="
log "Machine: $(uname -m) | Python: $(python --version 2>&1) | PWD: $PROJ_DIR"
log ""

TOTAL=0
PASSED=0
FAILED=0

for bus in 14 57; do
  TOTAL=$((TOTAL + 1))
  RUN_NAME="${bus}Bus_MPN_mixed_outages_lessepoch"
  RUN_LOG="$LOG_DIR/${RUN_NAME}.log"

  log "--- [$TOTAL/2] Starting $RUN_NAME ---"

  if [[ -d "Results/${RUN_NAME}" ]] && ls "Results/${RUN_NAME}/"*.pt 1>/dev/null 2>&1; then
    PASSED=$((PASSED + 1))
    log "    SKIPPED (already completed)"
    echo "SKIP  ${RUN_NAME}  -" >> "$SUMMARY_LOG"
    continue
  fi

  START_TIME=$(date +%s)

  python -u train.py \
    --bus "$bus" \
    --gnn_type MPN \
    --train_loss mixed \
    --epochs 500 \
    --patience 40 \
    --dataset_dir "Datasets/${bus}Bus_outages" \
    --output_dir "Results/${RUN_NAME}" \
    > "$RUN_LOG" 2>&1

  EXIT_CODE=$?
  END_TIME=$(date +%s)
  ELAPSED=$(( END_TIME - START_TIME ))
  MINS=$(( ELAPSED / 60 ))
  SECS=$(( ELAPSED % 60 ))

  if [[ $EXIT_CODE -eq 0 ]]; then
    PASSED=$((PASSED + 1))
    STATUS="OK"
    log "    PASSED in ${MINS}m ${SECS}s"
  else
    FAILED=$((FAILED + 1))
    STATUS="FAIL (exit $EXIT_CODE)"
    log "    FAILED (exit $EXIT_CODE) after ${MINS}m ${SECS}s — see $RUN_LOG"
  fi

  echo "${STATUS}  ${RUN_NAME}  ${MINS}m${SECS}s" >> "$SUMMARY_LOG"
done

log ""
log "===== PI-GNN training session finished ====="
log "Results: $PASSED passed, $FAILED failed out of $TOTAL runs"
