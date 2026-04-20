#!/usr/bin/env zsh
# Train 14Bus + 57Bus × 5 GNN types on outage datasets (500 epochs, patience 40)
# Launch: nohup caffeinate -s ./run_training_14_57.sh </dev/null >logs/nohup.out 2>&1 &
# Monitor: tail -f logs/master.log
# Summary: cat logs/summary.log

set -o pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

eval "$(conda shell.zsh hook)"
conda activate cos324

LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master.log"
SUMMARY_LOG="$LOG_DIR/summary.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

> "$MASTER_LOG"
> "$SUMMARY_LOG"

log "===== Training session: 14Bus + 57Bus (500 epochs, patience 40) ====="
log "Machine: $(uname -m) | Python: $(python --version 2>&1) | PWD: $PROJ_DIR"
log ""

TOTAL=0
PASSED=0
FAILED=0

for bus in 14 57; do
  for gnn in GCN GATConv GraphConv SAGEConv ChebConv; do
    TOTAL=$((TOTAL + 1))
    RUN_NAME="${bus}Bus_${gnn}_outages_lessepoch"
    RUN_LOG="$LOG_DIR/${RUN_NAME}.log"

    # GATConv crashes on MPS — force CPU
    if [[ "$gnn" == "GATConv" ]]; then
      DEV_FLAG="--device cpu"
    else
      DEV_FLAG=""
    fi

    log "--- [$TOTAL/10] Starting $RUN_NAME ${DEV_FLAG:+(${DEV_FLAG})} ---"

    # Skip if model already saved from a previous run
    if [[ -d "Results/${RUN_NAME}" ]] && ls "Results/${RUN_NAME}/"*.pt 1>/dev/null 2>&1; then
      PASSED=$((PASSED + 1))
      log "    SKIPPED (already completed)"
      echo "SKIP  ${RUN_NAME}  -" >> "$SUMMARY_LOG"
      continue
    fi

    START_TIME=$(date +%s)

    python -u train.py \
      --bus "$bus" \
      --gnn_type "$gnn" \
      --epochs 500 \
      --patience 40 \
      ${=DEV_FLAG} \
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
done

log ""
log "===== Training session finished ====="
log "Results: $PASSED passed, $FAILED failed out of $TOTAL runs"
