#!/bin/bash
#SBATCH --job-name=pignn_30_118
#SBATCH --output=logs/slurm_%x_%A_%a.out     # %A=jobID, %a=arrayIdx
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --array=0-1                          # one task per bus system
#SBATCH --partition=gpu                      # change to `class` or similar if you don't have gpu partition access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00                      # 30-bus usually < 6h; 118-bus can take 12-24h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@princeton.edu # ← change me

# =============================================================================
# Train PI-GNN (MPN + mixed MSE/PowerImbalance loss) on 30Bus + 118Bus.
# Submitted as a job array — index 0 = 30-bus, index 1 = 118-bus. Both tasks
# run in parallel if the scheduler has capacity; otherwise they queue.
#
# Submit:   sbatch slurm_pignn_30_118.slurm
# Check:    squeue -u $USER
# Cancel:   scancel <jobid>           # all array tasks
#           scancel <jobid>_0         # just the 30-bus task
#
# Before first submission, edit the `--mail-user` line above and confirm the
# `--partition` / `--gres` lines match what you have access to on Adroit
# (run `sinfo` and `sshare` to see available partitions and quotas).
# =============================================================================

set -euo pipefail

# ---- workspace ---------------------------------------------------------------
PROJ_DIR="$(pwd)"
mkdir -p logs Results

# ---- environment -------------------------------------------------------------
# Adroit uses Lmod. Adjust the conda module name if yours differs
# (run `module avail anaconda` on the login node to see available versions).
module purge
module load anaconda3/2024.6 || module load anaconda3 || true

# Activate the project's conda env. Change `gnn` if your env name differs.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn

# ---- pick bus system from the array index ------------------------------------
BUS_LIST=(30 118)
BUS=${BUS_LIST[$SLURM_ARRAY_TASK_ID]}

RUN_NAME="${BUS}Bus_MPN_mixed_outages_lessepoch"
DATASET_DIR="Datasets/${BUS}Bus_outages"
OUTPUT_DIR="Results/${RUN_NAME}"

echo "======================================================================"
echo " Job:        $SLURM_JOB_NAME  (task $SLURM_ARRAY_TASK_ID)"
echo " Host:       $(hostname)"
echo " Start:      $(date)"
echo " Bus:        $BUS"
echo " Dataset:    $DATASET_DIR"
echo " Output:     $OUTPUT_DIR"
echo " GPU:        ${CUDA_VISIBLE_DEVICES:-none assigned}"
echo " PyTorch device will be auto-detected (cuda > mps > cpu)"
echo "======================================================================"

# Skip if already completed (lets you resubmit the whole array safely).
if [[ -d "$OUTPUT_DIR" ]] && ls "$OUTPUT_DIR"/*.pt 1>/dev/null 2>&1; then
  echo "[skip] checkpoint already exists at $OUTPUT_DIR — nothing to do."
  exit 0
fi

# Sanity check that the dataset is present
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[error] $DATASET_DIR missing. Regenerate with:"
  echo "   python dataset_generation/generate_dataset.py --bus $BUS --num_datasets 20 --samples 2000 --with_outages"
  exit 1
fi

# ---- train -------------------------------------------------------------------
python -u train.py \
    --bus "$BUS" \
    --gnn_type MPN \
    --train_loss mixed \
    --epochs 500 \
    --patience 40 \
    --dataset_dir "$DATASET_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --device cuda

echo "======================================================================"
echo " End:        $(date)"
echo " Exit code:  $?"
echo "======================================================================"
