#!/bin/bash
# =============================================================================
# slurm_all_experiments.slurm
#
# Full experiment suite: train all GNN models on train_normal + train_n,
# evaluate on all generalisation axes, generate metric plots.
#
# Each array task handles ONE bus system.  Resource sizing is conservative
# for the largest system (118-bus); smaller ones finish well within limits.
#
# ── Submission examples ───────────────────────────────────────────────────
#
#   All 4 buses in parallel (one job per bus):
#     sbatch --array=0-3 slurm_all_experiments.slurm
#
#   Single bus (e.g. 14-bus only):
#     sbatch --array=0 slurm_all_experiments.slurm
#
#   Two specific buses (14 and 118):
#     sbatch --array=0,3 slurm_all_experiments.slurm
#
#   Override which models to run:
#     MODELS="GraphConv MPN" sbatch --array=0-3 slurm_all_experiments.slurm
#
#   Smoke-test (fast, for debugging):
#     EPOCHS=50 MODELS="GraphConv MPN" sbatch --array=0 slurm_all_experiments.slurm
#
#   Re-plot only (skip training):
#     sbatch --array=0-3 slurm_all_experiments.slurm --skip_training
#
# ── Bus → array index mapping ──────────────────────────────────────────────
#   index 0 → 14-bus
#   index 1 → 30-bus
#   index 2 → 57-bus
#   index 3 → 118-bus
#
# =============================================================================

#SBATCH --job-name=gnn_opf
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --partition=gpu                  # change to your partition (run `sinfo`)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                        # sized for 118-bus; smaller buses use less
#SBATCH --time=24:00:00                  # 14-bus ~2h, 30-bus ~6h, 57-bus ~10h, 118-bus ~20h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=durgadivija@gmail.com

set -euo pipefail

mkdir -p logs Results

# ── Environment ───────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.6 || module load anaconda3 || true
# Conda's activation script references $PS1 which isn't set in batch shells;
# the outer `set -u` would then abort. Disable nounset around conda activation.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn        # ← change if your conda env has a different name
set -u

# ── Bus selection ─────────────────────────────────────────────────────────
BUS_LIST=(14 30 57 118)
BUS=${BUS_LIST[$SLURM_ARRAY_TASK_ID]}

# ── Configurable overrides (set as env vars at submission time) ───────────
#   MODELS     — space-separated list, default: all six
#   EPOCHS     — training epochs, default: 500
#   PATIENCE   — early-stopping patience, default: 100
#   EXTRA_ARGS — any additional flags passed verbatim to run_all_experiments.py
MODELS="${MODELS:-GCN GraphConv SAGEConv GATConv ChebConv MPN}"
EPOCHS="${EPOCHS:-500}"
PATIENCE="${PATIENCE:-100}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "======================================================================"
echo " Job        : $SLURM_JOB_NAME  (array task $SLURM_ARRAY_TASK_ID)"
echo " Host       : $(hostname)"
echo " Start      : $(date)"
echo " Bus        : IEEE ${BUS}-bus"
echo " Models     : $MODELS"
echo " Epochs     : $EPOCHS  |  Patience: $PATIENCE"
echo " GPU        : ${CUDA_VISIBLE_DEVICES:-none assigned}"
echo " Results dir: Results/"
echo "======================================================================"

# ── Skip if this bus already completed ────────────────────────────────────
# Check for the presence of at least one checkpoint from a trainNormal run.
DONE_MARKER="Results/${BUS}Bus_GraphConv_trainNormal/train_stats.pt"
if [[ -f "$DONE_MARKER" ]] && [[ "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "[skip] Checkpoints already exist for ${BUS}-bus."
  echo "       Set FORCE_RERUN=1 to override, or use --skip_training to re-plot."
  exit 0
fi

# ── Validate datasets ─────────────────────────────────────────────────────
for SPLIT in train_normal train_n; do
  DIR="Datasets/${BUS}Bus/${SPLIT}"
  if [[ ! -d "$DIR" ]]; then
    echo "[error] Missing dataset: $DIR"
    echo "        Generate it first with:"
    echo "          bash generate_all_datasets.sh $BUS"
    exit 1
  fi
done

# ── Run experiments ───────────────────────────────────────────────────────
python -u run_all_experiments.py \
    --buses     "$BUS" \
    --models    $MODELS \
    --epochs    "$EPOCHS" \
    --patience  "$PATIENCE" \
    --device    cuda \
    $EXTRA_ARGS

echo "======================================================================"
echo " End        : $(date)"
echo " Exit code  : $?"
echo "======================================================================"
