"""
Dataset generation script for GNN-OptimalPowerFlow.

Generates AC power flow datasets (Newton-Raphson ground truth) with:
  - Random ±40% load variations (matching the paper's methodology)
  - Random line outages sampled from a Gaussian centred at 1 outage

Usage:
    python dataset_generation/generate_dataset.py --bus 14 --num_datasets 10 --samples 2000
"""

import argparse
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


def sample_n_outages(n_lines: int, mean: float = 1.0) -> int:
    """Sample the number of line outages from N(mean, 1), clipped to [1, n_lines]."""
    return int(np.clip(np.random.normal(loc=mean, scale=1), 1, n_lines))


def apply_line_outages(net, n_outages: int) -> list[int]:
    """Randomly disable n_outages lines. Returns the list of affected line indices."""
    indices = np.random.choice(len(net.line), size=n_outages, replace=False).tolist()
    for idx in indices:
        net.line.at[idx, 'in_service'] = False
    return indices


def generate_datasets(
    bus: str,
    num_datasets: int,
    samples_per_dataset: int,
    variation_range: float,
    output_dir: str,
    with_outages: bool,
) -> None:
    net = NETWORK_MAP[bus]()
    max_iter = MAX_ITERATIONS[bus]
    outage_mean = OUTAGE_MEANS[bus]

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

    for dataset_n in range(1, num_datasets + 1):
        print(f"Generating dataset {dataset_n}/{num_datasets}...")
        data = []
        skipped = 0

        for sample in range(1, samples_per_dataset + 1):
            # Reset loads and line states
            net.load['p_mw'] = original_p.copy()
            net.load['q_mvar'] = original_q.copy()
            net.line['in_service'] = original_line_service.copy()

            # Apply ±variation_range random load perturbations
            for load in net.load.index:
                net.load.at[load, 'p_mw'] = original_p[load] * (
                    1 + random.uniform(-variation_range, variation_range)
                )
                net.load.at[load, 'q_mvar'] = original_q[load] * (
                    1 + random.uniform(-variation_range, variation_range)
                )

            # Optionally apply line outages
            if with_outages:
                n_out = sample_n_outages(len(net.line), mean=outage_mean)
                apply_line_outages(net, n_out)

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
    parser.add_argument("--samples", type=int, default=2000,
                        help="Samples per dataset file (default: 2000)")
    parser.add_argument("--variation", type=float, default=0.4,
                        help="Load variation range as a fraction (default: 0.4 → ±40%%)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ../Datasets/<N>Bus_outages/)")
    parser.add_argument("--with_outages", action="store_true",
                        help="Enable random line outages per sample")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "Datasets",
        f"{args.bus}Bus{'_outages' if args.with_outages else ''}"
    )

    print(f"Bus system   : IEEE {args.bus}-bus")
    print(f"Datasets     : {args.num_datasets}")
    print(f"Samples each : {args.samples}")
    print(f"Load variation: ±{int(args.variation * 100)}%")
    print(f"Line outages : {'yes' if args.with_outages else 'no'}")
    print(f"Output dir   : {output_dir}\n")

    generate_datasets(
        bus=args.bus,
        num_datasets=args.num_datasets,
        samples_per_dataset=args.samples,
        variation_range=args.variation,
        output_dir=output_dir,
        with_outages=args.with_outages,
    )


if __name__ == "__main__":
    main()
