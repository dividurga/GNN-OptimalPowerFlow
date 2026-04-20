#!/usr/bin/env zsh
# Train PI-GNN (MPN + mixed MSE/PowerImbalance loss) on 30Bus + 118Bus
# outage datasets (500 epochs, patience 40), matching run_training_14_57_pi.sh.
#
# IMPORTANT preflight for the partner's machine:
#   1. Activate the env:                conda activate gnn   (or your local env name)
#   2. Confirm datasets exist:          ls Datasets/30Bus_outages Datasets/118Bus_outages
#      If missing, regenerate with:
#        python dataset_generation/generate_dataset.py --bus 30  --num_datasets 20 --samples 2000 --with_outages
#        python dataset_generation/generate_dataset.py --bus 118 --num_datasets 20 --samples 2000 --with_outages
#   3. On Apple Silicon the MPS device is auto-detected; CUDA works identically.
#
# Launch:  nohup caffeinate -s ./run_training_30_118_pi.sh </dev/null >logs/nohup_30_118_pi.out 2>&1 &
# Monitor: tail -f logs/master_30_118_pi.log
# Summary: cat logs/summary_30_118_pi.log
#
# Expected wall-clock on Apple Silicon MPS: 30-bus ~10h, 118-bus ~15-20h.
# Use CUDA if available for a large speedup.

set -o pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# Adjust this line to match the local conda env name if it's not "gnn"
eval "$(conda shell.zsh hook)"
conda activate gnn

LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_30_118_pi.log"
SUMMARY_LOG="$LOG_DIR/summary_30_118_pi.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

> "$MASTER_LOG"
> "$SUMMARY_LOG"

log "===== PI-GNN training: 30Bus + 118Bus (MPN + mixed loss, 500 epochs, patience 40) ====="
log "Machine: $(uname -m) | Python: $(python --version 2>&1) | PWD: $PROJ_DIR"
log ""

TOTAL=0
PASSED=0
FAILED=0

for bus in 30 118; do
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
