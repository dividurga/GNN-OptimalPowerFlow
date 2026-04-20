# GNN-OptimalPowerFlow

Graph Neural Networks for efficient prediction of AC Power Flow solutions on IEEE standard test systems (14, 30, 57, 118 bus). The repo benchmarks six GNN variants — **five standard** (GCN, GraphConv, SAGEConv, GATConv, ChebConv) and **one physics-informed** (MPN trained with a power-imbalance auxiliary loss, ported and adapted from [PowerFlowNet](https://github.com/StavrosOrf/PoweFlowNet)) — against Newton-Raphson ground truth.

## What's new in this version

- **PI-GNN (MPN + physics loss)**: an additional model that combines a TAGConv-based message-passing network with a loss term that enforces AC power balance (Kirchhoff's laws) at load buses. See [gnn/mpn.py](gnn/mpn.py) and [gnn/physics_loss.py](gnn/physics_loss.py).
- **Edge attributes (R, X)**: the data pipeline now attaches per-unit line resistance and reactance to every `Data` object as `edge_attr`. Existing baselines ignore this field; the MPN uses it.
- **Transformer-aware graph**: the graph now includes transformer branches (via `net._ppc['branch']`) in addition to lines. Previously only `net.line` edges were loaded, which left several buses (6, 7 in case14) effectively isolated.
- **Outage-aware edge attributes**: per-sample edge_attr tracks which lines are in/out of service, mirroring the existing per-sample edge_index logic.
- **Three training loss modes**: `--train_loss {mse, physics, mixed}` (default `mse`).

## Architectures

| Model       | File                     | Input                 | Edge attrs? | Default loss     |
|-------------|--------------------------|-----------------------|-------------|------------------|
| GCN         | [gnn/model.py](gnn/model.py) | 7 features/bus        | No          | MSE              |
| GraphConv   | [gnn/model.py](gnn/model.py) | 7 features/bus        | No          | MSE              |
| SAGEConv    | [gnn/model.py](gnn/model.py) | 7 features/bus        | No          | MSE              |
| GATConv     | [gnn/model.py](gnn/model.py) | 7 features/bus        | No          | MSE              |
| ChebConv    | [gnn/model.py](gnn/model.py) | 7 features/bus        | No          | MSE              |
| **MPN**     | [gnn/mpn.py](gnn/mpn.py)   | 7 feat/bus + R,X/edge | **Yes**     | MSE / mixed / physics |

Node features (7): `P`, `Q`, `V`, `δ`, `is_pv`, `is_pq`, `is_slack`.
Targets (2): `V`, `δ`.

### Physics loss

The `PowerImbalance` loss ([gnn/physics_loss.py](gnn/physics_loss.py)) computes the squared KCL residual `(ΔP² + ΔQ²)` at each bus using the AC power flow equations, applied only at pure-load buses (generator/slack buses are masked because the raw dataset doesn't record generator P, Q output — only load demand).

`MixedMSEPowerImbalance` combines plain MSE with the physics term (default `alpha=0.9`, `physics_scale=0.02`), matching the PowerFlowNet paper's formulation.

## Datasets

Pre-generated datasets live under `Datasets/` (gitignored — too large to check in). Each `PF_Dataset_N.xlsx` contains Newton-Raphson ground-truth samples with ±40% random load perturbations. Two flavors per bus system:

- `Datasets/{14,30,57,118}Bus/` — fixed topology
- `Datasets/{14,30,57,118}Bus_outages/` — per-sample random line outages (1 outage on average for 14/30-bus, more for larger systems)

Regenerate from scratch:

```bash
python dataset_generation/generate_dataset.py --bus 14 --num_datasets 20 --samples 2000
python dataset_generation/generate_dataset.py --bus 14 --num_datasets 20 --samples 2000 --with_outages
```

## Installation

```bash
conda env create -f environment.yml
conda activate gnn
```

Key dependencies: `torch==2.2.2`, `torch-geometric==2.7.0`, `pandapower`, `numpy==1.26.4`, `pandas`, `matplotlib`, `openpyxl`.

## Training

### Single model

```bash
# Plain MSE baseline (any of the 5 standard types)
python train.py --bus 14 --gnn_type SAGEConv --dataset_dir Datasets/14Bus_outages

# PI-GNN: MPN with mixed MSE + physics loss
python train.py --bus 14 --gnn_type MPN --train_loss mixed --dataset_dir Datasets/14Bus_outages

# MPN with plain MSE (architecture-only comparison, no physics term)
python train.py --bus 14 --gnn_type MPN --train_loss mse --dataset_dir Datasets/14Bus_outages
```

Key flags:

| Flag                 | Default      | Notes                                                 |
|----------------------|--------------|-------------------------------------------------------|
| `--bus`              | `14`         | `14`, `30`, `57`, or `118`                            |
| `--gnn_type`         | `GraphConv`  | `GCN`, `GraphConv`, `SAGEConv`, `GATConv`, `ChebConv`, `MPN` |
| `--train_loss`       | `mse`        | `mse`, `physics`, `mixed` (mixed is the PI-GNN default) |
| `--mixed_alpha`      | `0.9`        | MSE weight in the mixed loss                          |
| `--physics_scale`    | `0.02`       | Scale on the physics term (matches PowerFlowNet)      |
| `--mpn_hidden`       | `129`        | MPN hidden dim                                        |
| `--mpn_layers`       | `4`          | Number of TAGConv layers                              |
| `--mpn_K`            | `3`          | TAGConv filter order                                  |
| `--epochs`           | `800`        |                                                        |
| `--patience`         | `100`        | Early-stopping patience                               |
| `--dataset_dir`      | auto         | Defaults to `PerturbedDatasets/<N>Bus/`               |
| `--device`           | auto         | `cpu`, `cuda`, `mps`                                   |

Validation/test loss is always reported as **denormalised MSE** — safe to compare across `--train_loss` modes.

### Batch training (multiple configs)

Two helper scripts train an entire suite and stream logs to `logs/`:

```bash
# 5 baselines × {14, 57} on outage datasets (500 epochs, patience 40)
nohup caffeinate -s ./run_training_14_57.sh </dev/null >logs/nohup.out 2>&1 &

# PI-GNN (MPN + mixed loss) × {14, 57}
nohup caffeinate -s ./run_training_14_57_pi.sh </dev/null >logs/nohup_pi.out 2>&1 &

# Monitor
tail -f logs/master.log       # baselines
tail -f logs/master_pi.log    # PI-GNN
```

### Training the remaining two bus systems (30 + 118)

Two ready-to-go launchers are provided; pick the one that matches your machine.

#### A. Local macOS / Linux workstation

Use [run_training_30_118_pi.sh](run_training_30_118_pi.sh) — same pattern as `run_training_14_57_pi.sh`:

```bash
# macOS
nohup caffeinate -s ./run_training_30_118_pi.sh </dev/null >logs/nohup_30_118_pi.out 2>&1 &

# Linux (no caffeinate needed)
nohup ./run_training_30_118_pi.sh </dev/null >logs/nohup_30_118_pi.out 2>&1 &

tail -f logs/master_30_118_pi.log
```

The shell script expects a conda env named `gnn` and auto-detects the device (CUDA > MPS > CPU). Expect ~6h for 30-bus and ~15–20h for 118-bus on Apple Silicon MPS; much faster on CUDA.

#### B. Princeton Adroit (SLURM / HPC cluster)

Login nodes don't run long jobs — use [slurm_pignn_30_118.slurm](slurm_pignn_30_118.slurm). It submits a **2-task job array** (one per bus) onto GPU nodes.

```bash
# 1. Edit the script once: set --mail-user to your Princeton email, and confirm
#    the --partition / --gres lines match your Adroit access (run `sinfo` to see).

# 2. Submit both 30-bus and 118-bus runs in parallel:
sbatch slurm_pignn_30_118.slurm

# 3. Monitor:
squeue -u $USER                       # queue status
tail -f logs/slurm_pignn_30_118_*.out # per-task stdout once running
tail -f logs/30Bus_MPN_mixed_outages_lessepoch.log   # train.py's own log

# 4. Cancel if needed:
scancel <jobid>        # cancels the whole array
scancel <jobid>_0      # cancels just the 30-bus task
```

The script `module load`s anaconda, activates the `gnn` env, and checks for `Datasets/{30,118}Bus_outages/` before launching — if a dataset is missing it prints the regeneration command and exits cleanly. If a checkpoint already exists at `Results/<run_name>/` the task no-ops, so you can safely resubmit after failures without overwriting work.

> Same pattern works for the 5 baselines — copy the `.slurm` file and swap the `python train.py` line to use `--gnn_type {GCN|GraphConv|...} --train_loss mse`.

Keep datasets in the same location: `Datasets/{30,118}Bus_outages/` (regenerate with the dataset-generation command above if missing).

## Repo layout

```text
gnn/
  data.py           # dataset loading, edge R/X extraction, normalization
  model.py          # GNNPowerFlow wrapper over 5 standard conv types
  mpn.py            # MPN model (ported + adapted from PowerFlowNet)
  physics_loss.py   # PowerImbalance + MixedMSEPowerImbalance
  metrics.py        # NRMSE, R², etc.
train.py            # main training script — single model, any loss
evaluate.py         # checkpoint evaluation on held-out test set
dataset_generation/ # synthetic PF dataset generator (pandapower + NR)
run_training_14_57.sh       # batch run: 5 baselines × 14, 57-bus (local)
run_training_14_57_pi.sh    # batch run: MPN+mixed × 14, 57-bus (local)
run_training_30_118_pi.sh   # batch run: MPN+mixed × 30, 118-bus (local)
slurm_pignn_30_118.slurm    # Adroit HPC job-array version of the 30/118-bus run
environment.yml             # conda env spec
```

## Physics-informed model — design notes

The power-imbalance loss does *not* assume the dataset contains generator output — it only checks KCL at load buses. Key implementation choices:

1. **Per-unit conversion**: raw dataset P, Q are in MW/MVAr; loss divides by `sn_mva` (100 for IEEE cases) to convert to per-unit before computing imbalance.
2. **Degrees → radians**: dataset stores voltage angle in degrees (verified empirically); loss applies `π/180` internally.
3. **Transformer approximation**: transformer branches are treated as lines (tap ratio = 1) in the imbalance computation. This introduces a small residual at transformer-connected buses (~0.02 on ground truth for 14-bus) but is adequate as a soft auxiliary signal.
4. **Load-bus masking**: imbalance is averaged only over buses *without* generators (derived from `net.gen` + `net.ext_grid`). Without this mask the loss is dominated by constant errors at generator buses.

## Results

### Baselines (pre–edge-feature pipeline)

Historical numbers from `Results/*` reflect training before the data pipeline included transformer edges. These are not directly comparable to the new MPN runs — plan to rerun after the current PI-GNN job finishes.

### Early PI-GNN results

| Config                         | Best val MSE (denorm.) | Epochs | Notes                          |
|--------------------------------|------------------------|--------|--------------------------------|
| 14-bus outages, MPN + mixed    | **8.7e-5** (epoch 492) | 500    | ~8h on Apple Silicon MPS       |
| 57-bus outages, MPN + mixed    | *in progress*          | 500    | see logs/57Bus_MPN_mixed_outages_lessepoch.log |

## Reference

The physics-informed MPN model and loss are adapted from:

> Lin et al., *PowerFlowNet: Power flow approximation using message-passing Graph Neural Networks*, International Journal of Electrical Power & Energy Systems, vol. 160, 2024. [doi](https://doi.org/10.1016/j.ijepes.2024.110112) · [original repo](https://github.com/StavrosOrf/PoweFlowNet)

The baseline GNN variants and dataset-generation methodology follow the original NCSU ECE592 project report (see `document/`).

## Acknowledgments

Originally developed as an ECE592 (Advanced Topics in Deep Learning, NCSU) project under Dr. Kaixiong Zhou; extended with the PI-GNN integration.
