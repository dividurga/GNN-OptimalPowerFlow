#!/usr/bin/env bash
# =============================================================================
# generate_all_datasets.sh
#
# Generates every train / val / test dataset variant needed for the
# generalisation experiments.  Run locally or wrap in a SLURM job.
#
# Directory layout produced under Datasets/<N>Bus/:
#
#   train_normal/       Files 1-7   — i.i.d. ±40 % loads, Gaussian N-k, train-partition lines
#                       File  8     — same loads/outages, val-partition lines (topology-disjoint)
#   train_n/            Files 1-8   — i.i.d. ±40 % loads, NO outages (full network)
#
#   test_indist/        Files 1-4   — same distrib as train_normal  (baseline)
#   test_topo/          Files 1-4   — unseen line indices (test partition), ±40 %
#   test_load_tail/     Files 1-4   — train lines, load annulus 40 %–80 %
#   test_load_extreme/  Files 1-4   — train lines, load annulus 80 %–120 %
#   test_nk1/           Files 1-4   — exactly 1 line outage  (N-1)
#   test_nk2/           Files 1-4   — exactly 2 line outages (N-2)
#   test_nk3/           Files 1-4   — exactly 3 line outages (N-3)
#   test_corr/          Files 1-4   — train lines, spatially correlated loads, ±40 %
#
# N-k experiment: train a model on train_n/ (no outages), then evaluate on
# test_nk1/2/3/ to measure zero-shot generalisation to contingency scenarios.
#
# Usage:
#   bash generate_all_datasets.sh             # all four bus systems
#   bash generate_all_datasets.sh 14          # single bus system
#   BUSES="30 118" bash generate_all_datasets.sh
#
# SLURM (example — adjust partition / mem as needed):
#   sbatch --partition=cpu --mem=8G --time=12:00:00 \
#          --output=logs/gen_%j.out generate_all_datasets.sh
# =============================================================================

set -euo pipefail

# ---- configurable knobs ------------------------------------------------------
BUSES="${BUSES:-${1:-14 30 57 118}}"   # override with env var or positional arg
SAMPLES=2000                            # samples per Excel file
TRAIN_FILES=7                           # files 1-7   → train partition
VAL_FILES=1                             # file 8      → val partition (topology-disjoint)
TEST_FILES=4                            # files 1-4   per test set
TOPO_SEED=42

GEN="python dataset_generation/generate_dataset.py"
TOPO_SPLIT="python dataset_generation/make_topology_splits.py"

# ---- helpers -----------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# run_gen_outages: generates with --with_outages (for train_normal + most test sets)
run_gen() {
    local bus=$1 subdir=$2; shift 2
    local out_dir="Datasets/${bus}Bus/${subdir}"
    log "  → ${out_dir}"
    $GEN --bus "$bus" --samples "$SAMPLES" --with_outages \
         --output_dir "$out_dir" "$@"
}

# run_gen_no_outages: generates WITHOUT outages (for train_n and test_indist_n)
run_gen_no_outages() {
    local bus=$1 subdir=$2; shift 2
    local out_dir="Datasets/${bus}Bus/${subdir}"
    log "  → ${out_dir}"
    $GEN --bus "$bus" --samples "$SAMPLES" \
         --output_dir "$out_dir" "$@"
}

# ==============================================================================
mkdir -p logs topology_splits

for BUS in $BUSES; do
    log "================================================================"
    log " IEEE ${BUS}-bus — generating all dataset splits"
    log "================================================================"

    SPLIT_FILE="topology_splits/${BUS}Bus_topology_split.json"

    # ------------------------------------------------------------------
    # Step 0 — topology split (idempotent: same seed → same split)
    # ------------------------------------------------------------------
    log "Step 0 — topology split"
    $TOPO_SPLIT --bus "$BUS" --seed $TOPO_SEED
    log "  saved: $SPLIT_FILE"

    # ------------------------------------------------------------------
    # Step 1 — train_normal  (i.i.d. ±40 %, Gaussian N-k, train lines)
    #          Files 1-7: train topology partition
    #          File  8:   val topology partition  (disjoint from train)
    # ------------------------------------------------------------------
    log "Step 1 — train_normal (files 1-${TRAIN_FILES}, train partition)"
    run_gen "$BUS" train_normal \
        --num_datasets "$TRAIN_FILES" \
        --variation 0.4 \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition train

    log "Step 1b — train_normal (files $((TRAIN_FILES+1))-$((TRAIN_FILES+VAL_FILES)), val partition)"
    run_gen "$BUS" train_normal \
        --num_datasets "$VAL_FILES" \
        --start_index $((TRAIN_FILES+1)) \
        --variation 0.4 \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition val

    # ------------------------------------------------------------------
    # Step 2 — train_n  (full network, NO outages)
    #          No topology partition needed — no outages means no topology leakage.
    #          Files 1-7 train, 8 val (same distribution, no outages either way).
    # ------------------------------------------------------------------
    log "Step 2 — train_n  (no outages, files 1-$((TRAIN_FILES+VAL_FILES)))"
    run_gen_no_outages "$BUS" train_n \
        --num_datasets $((TRAIN_FILES+VAL_FILES)) \
        --variation 0.4

    # ------------------------------------------------------------------
    # Step 3 — test_indist  (same distribution as train_normal — baseline)
    # ------------------------------------------------------------------
    log "Step 3 — test_indist"
    run_gen "$BUS" test_indist \
        --num_datasets "$TEST_FILES" \
        --variation 0.4 \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition train

    # ------------------------------------------------------------------
    # Step 4 — test_topo  (unseen lines; isolates topology axis)
    # ------------------------------------------------------------------
    log "Step 4 — test_topo"
    run_gen "$BUS" test_topo \
        --num_datasets "$TEST_FILES" \
        --variation 0.4 \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition test

    # ------------------------------------------------------------------
    # Step 5 — test_load_tail  (loads in annulus 40 %–80 %; isolates load axis)
    # ------------------------------------------------------------------
    log "Step 5 — test_load_tail"
    run_gen "$BUS" test_load_tail \
        --num_datasets "$TEST_FILES" \
        --variation 0.8 --variation_min 0.4 \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition train

    # ------------------------------------------------------------------
    # Step 6 — test_load_extreme  (loads in annulus 80 %–120 %)
    # ------------------------------------------------------------------
    log "Step 6 — test_load_extreme"
    run_gen "$BUS" test_load_extreme \
        --num_datasets "$TEST_FILES" \
        --variation 1.2 --variation_min 0.8 \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition train

    # ------------------------------------------------------------------
    # Steps 7-9 — test_nk1 / test_nk2 / test_nk3
    #   Exactly 1 / 2 / 3 simultaneous line outages, any line in the network.
    #   No topology partition: train_n has no outages so nothing to leak.
    # ------------------------------------------------------------------
    log "Step 7 — test_nk1  (N-1: exactly 1 outage)"
    run_gen "$BUS" test_nk1 \
        --num_datasets "$TEST_FILES" \
        --variation 0.4 \
        --outage_exact 1

    log "Step 8 — test_nk2  (N-2: exactly 2 outages)"
    run_gen "$BUS" test_nk2 \
        --num_datasets "$TEST_FILES" \
        --variation 0.4 \
        --outage_exact 2

    log "Step 9 — test_nk3  (N-3: exactly 3 outages)"
    run_gen "$BUS" test_nk3 \
        --num_datasets "$TEST_FILES" \
        --variation 0.4 \
        --outage_exact 3

    # ------------------------------------------------------------------
    # Step 10 — test_corr  (correlated loads; isolates spatial axis)
    # ------------------------------------------------------------------
    log "Step 10 — test_corr"
    run_gen "$BUS" test_corr \
        --num_datasets "$TEST_FILES" \
        --variation 0.4 \
        --correlated_loads \
        --topology_split_file "$SPLIT_FILE" \
        --topology_partition train

    log "Done — IEEE ${BUS}-bus  ✓"
    log ""
done

log "All buses complete."
log ""
log "Dataset summary:"
log "  train_normal/      — Files 1-${TRAIN_FILES} (train partition lines)"
log "                       Files $((TRAIN_FILES+1))-$((TRAIN_FILES+VAL_FILES)) (val partition lines, topology-disjoint)"
log "                       i.i.d. ±40 %% loads, Gaussian N-k outages"
log "  train_n/           — Files 1-$((TRAIN_FILES+VAL_FILES)), no outages (full network)"
log ""
log "  test_indist/       — baseline (same distrib as train_normal)"
log "  test_topo/         — unseen line indices (20 %% reserved test-partition lines)"
log "  test_load_tail/    — OOD loads in [40 %%, 80 %%]"
log "  test_load_extreme/ — OOD loads in [80 %%, 120 %%]"
log "  test_nk1/          — exactly N-1 (1 line out)"
log "  test_nk2/          — exactly N-2 (2 lines out)"
log "  test_nk3/          — exactly N-3 (3 lines out)"
log "  test_corr/         — spatially correlated load shifts"
log ""
log "N-k experiment:  train on train_n/  → evaluate on test_nk1/, test_nk2/, test_nk3/"
log "Other axes:      train on train_normal/ → evaluate on test_indist/, test_topo/,"
log "                 test_load_tail/, test_load_extreme/, test_corr/"
