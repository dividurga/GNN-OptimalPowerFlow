#!/bin/bash
# =============================================================================
# slurm_run_experiments.slurm
#
# Runs run_all_experiments.py for one bus system per array task.
#
# ── Submission examples ───────────────────────────────────────────────────────
#
#   All 4 buses in parallel:
#     sbatch --array=0-3 slurm_run_experiments.slurm
#
#   Single bus (e.g. 14-bus only):
#     sbatch --array=0 slurm_run_experiments.slurm
#
#   Specific models only:
#     MODELS="GraphConv MPN" sbatch --array=0-3 slurm_run_experiments.slurm
#
#   Smoke-test (fast, for debugging):
#     EPOCHS=50 MODELS="GraphConv" sbatch --array=0 slurm_run_experiments.slurm
#
#   Skip training, re-plot from existing checkpoints:
#     sbatch --array=0-3 slurm_run_experiments.slurm --skip_training
#
# ── Bus → array index mapping ─────────────────────────────────────────────────
#   index 0 → 14-bus
#   index 1 → 30-bus
#   index 2 → 57-bus
#   index 3 → 118-bus
#
# =============================================================================

#SBATCH --job-name=gnn_exp
#SBATCH --output=logs/exp_%x_%A_%a.out
#SBATCH --error=logs/exp_%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=durgadivija@gmail.com

set -euo pipefail

mkdir -p logs Results

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.6 || module load anaconda3 || true
# Conda's activation script references $PS1 which isn't set in batch shells;
# the outer `set -u` would then abort. Disable nounset around conda activation.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn
set -u

# ── Bus selection ─────────────────────────────────────────────────────────────
BUS_LIST=(14 30 57 118)
BUS=${BUS_LIST[$SLURM_ARRAY_TASK_ID]}

# ── Configurable overrides ────────────────────────────────────────────────────
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
echo "======================================================================"

# ── Validate datasets ─────────────────────────────────────────────────────────
for SPLIT in train_normal train_n; do
    DIR="Datasets/${BUS}Bus/${SPLIT}"
    if [[ ! -d "$DIR" ]]; then
        echo "[error] Missing dataset: $DIR"
        echo "        Generate it first with: bash generate_all_datasets.sh $BUS"
        exit 1
    fi
done

# ── Run experiments ───────────────────────────────────────────────────────────
python -u run_all_experiments.py \
    --buses       "$BUS" \
    --models      $MODELS \
    --epochs      "$EPOCHS" \
    --patience    "$PATIENCE" \
    --datasets_dir Datasets \
    --results_dir  Results \
    --device       cuda \
    $EXTRA_ARGS

echo "======================================================================"
echo " End        : $(date)"
echo " Exit code  : $?"
echo "======================================================================"
