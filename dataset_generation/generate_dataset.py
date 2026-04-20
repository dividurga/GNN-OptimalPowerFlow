"""
Dataset generation script for GNN-OptimalPowerFlow.

Generates AC power flow datasets (Newton-Raphson ground truth) with:
  - Random ±40% load variations (matching the paper's methodology)
  - Random line outages sampled from a Gaussian centred at 1 outage

New flags for generalisation experiments:
  --variation_min       sample load factors from annulus [variation_min, variation]
                        (absolute value) to generate tail / extreme load test sets
  --outage_exact        force exactly this many simultaneous outages per sample
  --topology_split_file path to JSON produced by make_topology_splits.py
  --topology_partition  'train' or 'test' — which line-index partition to use
  --correlated_loads    all buses share a global factor + small per-bus noise

Usage:
    python dataset_generation/generate_dataset.py --bus 14 --num_datasets 10 --samples 2000
    # tail-load test set (loads between ±40 % and ±80 %):
    python dataset_generation/generate_dataset.py --bus 14 --num_datasets 4 \\
        --variation 0.8 --variation_min 0.4 --with_outages \\
        --topology_split_file topology_splits/14Bus_topology_split.json \\
        --topology_partition train --output_dir Datasets/14Bus/test_load_tail
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw


NETWORK_MAP = {
    "14":  nw.case14,
    "30":  nw.case30,
    "57":  nw.case57,
    "118": nw.case118,
}

MAX_ITERATIONS = {
    "14":  20,
    "30":  20,
    "57":  30,
    "118": 35,
}

OUTAGE_MEANS = {
    "14":  1.0,
    "30":  1.0,
    "57":  1.5,
    "118": 2.0,
}


def sample_n_outages(n_available: int, mean: float = 1.0) -> int:
    """Sample the number of line outages from N(mean, 1), clipped to [1, n_available]."""
    return int(np.clip(np.random.normal(loc=mean, scale=1), 1, n_available))


def apply_line_outages(net, n_outages: int, allowed_indices: list[int] | None = None) -> list[int]:
    """
    Randomly disable n_outages lines from allowed_indices (all lines if None).
    Returns the list of affected line indices.
    """
    pool = allowed_indices if allowed_indices is not None else list(range(len(net.line)))
    n_outages = min(n_outages, len(pool))
    indices = random.sample(pool, n_outages)
    for idx in indices:
        net.line.at[idx, 'in_service'] = False
    return indices


def sample_load_factor(variation: float, variation_min: float | None) -> float:
    """
    If variation_min is None: uniform in [-variation, variation].
    Otherwise:  magnitude drawn from [variation_min, variation], random sign.
    Used to produce tail / extreme out-of-distribution load test sets.
    """
    if variation_min is not None:
        mag = random.uniform(variation_min, variation)
        return mag * random.choice([-1, 1])
    return random.uniform(-variation, variation)


def generate_datasets(
    bus: str,
    num_datasets: int,
    samples_per_dataset: int,
    variation_range: float,
    output_dir: str,
    with_outages: bool,
    variation_min: float | None = None,
    outage_exact: int | None = None,
    allowed_line_indices: list[int] | None = None,
    correlated_loads: bool = False,
    start_index: int = 1,
) -> None:
    net = NETWORK_MAP[bus]()
    max_iter = MAX_ITERATIONS[bus]
    outage_mean = OUTAGE_MEANS[bus]
    pool_size = len(allowed_line_indices) if allowed_line_indices is not None else len(net.line)

    os.makedirs(output_dir, exist_ok=True)

    bus_ids  = net.bus.index
    line_ids = net.line.index
    columns  = []
    for b in bus_ids:
        columns.extend([f"P_{b + 1} (PQ)", f"Q_{b + 1} (PQ)", f"V_{b + 1}", f"d_{b + 1}"])
    for l in line_ids:
        columns.append(f"line_{l}_in_service")

    original_p = net.load['p_mw'].copy()
    original_q = net.load['q_mvar'].copy()
    original_line_service = net.line['in_service'].copy()

    for dataset_n in range(start_index, start_index + num_datasets):
        print(f"Generating dataset {dataset_n}/{num_datasets}...")
        data = []
        skipped = 0

        for sample in range(1, samples_per_dataset + 1):
            # Reset loads and line states
            net.load['p_mw'] = original_p.copy()
            net.load['q_mvar'] = original_q.copy()
            net.line['in_service'] = original_line_service.copy()

            # Apply load perturbations
            if correlated_loads:
                # All buses share a global shift; per-bus noise is 20 % of that range.
                global_factor = sample_load_factor(variation_range, variation_min)
                noise_scale   = variation_range * 0.2
                for load in net.load.index:
                    noise = random.uniform(-noise_scale, noise_scale)
                    net.load.at[load, 'p_mw']    = original_p[load] * (1 + global_factor + noise)
                    net.load.at[load, 'q_mvar']  = original_q[load] * (1 + global_factor + noise)
            else:
                for load in net.load.index:
                    net.load.at[load, 'p_mw']   = original_p[load] * (
                        1 + sample_load_factor(variation_range, variation_min)
                    )
                    net.load.at[load, 'q_mvar'] = original_q[load] * (
                        1 + sample_load_factor(variation_range, variation_min)
                    )

            # Optionally apply line outages
            if with_outages:
                n_out = outage_exact if outage_exact is not None else sample_n_outages(pool_size, mean=outage_mean)
                apply_line_outages(net, n_out, allowed_line_indices)

            # Run Newton-Raphson power flow
            try:
                pp.runpp(net, algorithm='nr', max_iteration=max_iter)
            except pp.LoadflowNotConverged:
                skipped += 1
                continue

            # Collect per-bus P, Q, V, δ
            row = []
            for b in bus_ids:
                if b in net.load['bus'].values:
                    p_load = net.res_load.loc[
                        net.load[net.load['bus'] == b].index, 'p_mw'
                    ].sum()
                    q_load = net.res_load.loc[
                        net.load[net.load['bus'] == b].index, 'q_mvar'
                    ].sum()
                else:
                    p_load, q_load = 0.0, 0.0

                v_mag = net.res_bus.at[b, 'vm_pu']
                v_ang = net.res_bus.at[b, 'va_degree']
                row.extend([p_load, q_load, v_mag, v_ang])

            # Append per-line in-service status (1=active, 0=out)
            row.extend(net.line['in_service'].astype(int).values.tolist())

            data.append(row)

        df = pd.DataFrame(data, columns=columns)
        df.insert(0, "Data", [f"Data {i + 1}" for i in range(len(df))])
        df.insert(0, "Dataset", [f"PF Dataset_{dataset_n}"] + [""] * (len(df) - 1))

        out_path = os.path.join(output_dir, f"PF_Dataset_{dataset_n}.xlsx")
        df.to_excel(out_path, index=False)
        print(f"  Saved {len(data)} samples to {out_path} ({skipped} skipped)")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Generate OPF datasets with Newton-Raphson.")
    parser.add_argument("--bus", choices=["14", "30", "57", "118"], default="14",
                        help="IEEE bus system to use (default: 14)")
    parser.add_argument("--num_datasets", type=int, default=20,
                        help="Number of dataset files to generate (default: 20)")
    parser.add_argument("--start_index", type=int, default=1,
                        help="Starting file index for output files (default: 1 → PF_Dataset_1.xlsx)")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Samples per dataset file (default: 2000)")
    parser.add_argument("--variation", type=float, default=0.4,
                        help="Load variation range as a fraction (default: 0.4 → ±40%%)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ../Datasets/<N>Bus_outages/)")
    parser.add_argument("--with_outages", action="store_true",
                        help="Enable random line outages per sample")
    # ---- generalisation experiment flags ----
    parser.add_argument("--variation_min", type=float, default=None,
                        help="If set, load perturbations are sampled from the annulus "
                             "[variation_min, variation] in absolute value. "
                             "Use for tail (0.4–0.8) and extreme (0.8–1.2) test sets.")
    parser.add_argument("--outage_exact", type=int, default=None,
                        help="Force exactly this many simultaneous line outages per sample "
                             "(requires --with_outages). Overrides Gaussian sampling.")
    parser.add_argument("--topology_split_file", type=str, default=None,
                        help="Path to JSON produced by make_topology_splits.py. "
                             "Restricts outages to a partition of line indices so that "
                             "train and test topologies are disjoint.")
    parser.add_argument("--topology_partition", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Which partition to use from --topology_split_file.")
    parser.add_argument("--correlated_loads", action="store_true",
                        help="All buses share a global load factor; individual noise is "
                             "±20%% of variation_range. Tests generalisation to structured "
                             "regional demand shifts.")
    args = parser.parse_args()

    # ---- resolve topology partition ----
    allowed_line_indices = None
    if args.topology_split_file is not None:
        if args.topology_partition is None:
            raise ValueError("--topology_partition (train|test) is required with --topology_split_file")
        with open(args.topology_split_file) as fh:
            split = json.load(fh)
        key = f"{args.topology_partition}_line_indices"
        allowed_line_indices = split[key]
        print(f"Topology partition : {args.topology_partition}  "
              f"({len(allowed_line_indices)} / {split['n_lines']} lines available for outage)")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "Datasets",
        f"{args.bus}Bus{'_outages' if args.with_outages else ''}"
    )

    variation_desc = (
        f"annulus [{int(args.variation_min * 100)}%–{int(args.variation * 100)}%]"
        if args.variation_min is not None
        else f"±{int(args.variation * 100)}%"
    )
    outage_desc = (
        f"exact {args.outage_exact}" if args.outage_exact is not None
        else ("Gaussian" if args.with_outages else "none")
    )
    print(f"Bus system    : IEEE {args.bus}-bus")
    print(f"Datasets      : {args.num_datasets}")
    print(f"Samples each  : {args.samples}")
    print(f"Load variation: {variation_desc}")
    print(f"Correlated    : {'yes' if args.correlated_loads else 'no'}")
    print(f"Line outages  : {outage_desc}")
    print(f"Output dir    : {output_dir}\n")

    generate_datasets(
        bus=args.bus,
        num_datasets=args.num_datasets,
        samples_per_dataset=args.samples,
        variation_range=args.variation,
        output_dir=output_dir,
        with_outages=args.with_outages,
        variation_min=args.variation_min,
        outage_exact=args.outage_exact,
        allowed_line_indices=allowed_line_indices,
        correlated_loads=args.correlated_loads,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
