# For Noah — All Tests We Should Run

## New / Modified Files

| File | Type | Purpose |
|---|---|---|
| `dataset_generation/make_topology_splits.py` | NEW | Partitions each bus system's line indices 80/20 into train and test sets, saved as a JSON. Ensures lines removed during test were never removed during training. |
| `dataset_generation/generate_dataset.py` | MODIFIED | Added 5 new flags: `--variation_min`, `--outage_exact`, `--topology_split_file`, `--topology_partition`, `--correlated_loads` |
| `generate_all_datasets.sh` | NEW | Master script — runs everything in order and produces all 10 dataset directories per bus system |

---

## Generalisation Axes and Datasets

**Two training sets:**

| Directory | Description | Use for |
|---|---|---|
| `train_normal/` | ±40% loads, Gaussian N-k outages, train-partition lines | All axes except N-k |
| `train_n/` | ±40% loads, **no outages** (full network) | N-k experiment only |

**Test sets — each isolates one axis, holding all others fixed:**

| Directory | What varies | What's fixed | Train set to pair with |
|---|---|---|---|
| `test_indist/` | Nothing (baseline) | ±40%, same N-k, train lines | `train_normal` |
| `test_topo/` | **Line indices** (20% reserved, never seen in training) | ±40%, Gaussian N-k depth | `train_normal` |
| `test_load_tail/` | **Load magnitude** [40%–80%] | Train lines, same N-k | `train_normal` |
| `test_load_extreme/` | **Load magnitude** [80%–120%] | Train lines, same N-k | `train_normal` |
| `test_nk1/` | **N-k depth** (exactly 1 outage) | ±40%, any line | `train_n` |
| `test_nk2/` | **N-k depth** (exactly 2 outages) | ±40%, any line | `train_n` |
| `test_nk3/` | **N-k depth** (exactly 3 outages) | ±40%, any line | `train_n` |
| `test_corr/` | **Load spatial structure** (all buses correlated) | Train lines, same N-k | `train_normal` |

---

## Reproduction Instructions

### Step 1 — Generate all datasets

```bash
# From the project root. Runs all bus systems (14, 30, 57, 118).
bash generate_all_datasets.sh

# Single bus only:
bash generate_all_datasets.sh 14

# Specific buses via env var:
BUSES="30 118" bash generate_all_datasets.sh
```

On SLURM (CPU node, ~8–12h for all 4 buses):
```bash
sbatch --partition=cpu --mem=8G --time=12:00:00 \
       --output=logs/gen_%j.out generate_all_datasets.sh
```

After this, your `Datasets/` folder looks like:
```
Datasets/
  14Bus/
    train_normal/   train_n/
    test_indist/    test_topo/
    test_load_tail/ test_load_extreme/
    test_nk1/       test_nk2/   test_nk3/
    test_corr/
  30Bus/  57Bus/  118Bus/   (same structure)
topology_splits/
  14Bus_topology_split.json  ...
```

---

### Step 2 — Train models

**For topology / load / corr experiments** (train on `train_normal`):
```bash
python train.py --bus 14 --gnn_type GraphConv \
    --dataset_dir Datasets/14Bus/train_normal \
    --train_files 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --val_files 17 18
```

**For the N-k experiment** (train on `train_n`, full network only):
```bash
python train.py --bus 14 --gnn_type GraphConv \
    --dataset_dir Datasets/14Bus/train_n \
    --train_files 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
    --val_files 17 18
```

Repeat for each `--gnn_type` (GraphConv, GCN, GAT, SAGEConv, MPN) and each `--bus`.

---

### Step 3 — Evaluate on each test set

Point your evaluation script at a test directory with files 1–4:

```bash
# Axis: baseline
python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv/best_model.pt \
    --dataset_dir Datasets/14Bus/test_indist --test_files 1 2 3 4

# Axis: topology generalisation
python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv/best_model.pt \
    --dataset_dir Datasets/14Bus/test_topo --test_files 1 2 3 4

# Axis: load tail
python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv/best_model.pt \
    --dataset_dir Datasets/14Bus/test_load_tail --test_files 1 2 3 4

# Axis: load extreme
python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv/best_model.pt \
    --dataset_dir Datasets/14Bus/test_load_extreme --test_files 1 2 3 4

# Axis: correlated loads
python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv/best_model.pt \
    --dataset_dir Datasets/14Bus/test_corr --test_files 1 2 3 4

# Axis: N-k (use the train_n checkpoint here)
python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv_trainN/best_model.pt \
    --dataset_dir Datasets/14Bus/test_nk1 --test_files 1 2 3 4

python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv_trainN/best_model.pt \
    --dataset_dir Datasets/14Bus/test_nk2 --test_files 1 2 3 4

python evaluate.py --bus 14 --checkpoint Results/14Bus_GraphConv_trainN/best_model.pt \
    --dataset_dir Datasets/14Bus/test_nk3 --test_files 1 2 3 4
```

---

## What Each Experiment Answers

| Experiment | Question |
|---|---|
| `train_normal` → `test_indist` | In-distribution baseline — sanity check |
| `train_normal` → `test_topo` | Does the model generalise to line failures it has never seen? |
| `train_normal` → `test_load_tail` | Does accuracy degrade for loads 40–80% away from nominal? |
| `train_normal` → `test_load_extreme` | How bad is it at 80–120% load deviation? |
| `train_normal` → `test_corr` | Does correlated regional demand break the model vs. i.i.d. per-bus noise? |
| `train_n` → `test_nk1/2/3` | Can a model trained on the full network predict power flow after N-1, N-2, N-3 contingencies it has never encountered? Does this degrade gracefully or cliff? |
