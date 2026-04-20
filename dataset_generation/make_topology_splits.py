"""
Partitions line indices into train and test sets for topology-disjoint dataset generation.

For each IEEE bus system, assigns 80% of line indices to the train partition and 20% to
the test partition. The resulting JSON is consumed by generate_dataset.py via
--topology_split_file / --topology_partition to ensure that lines removed during test
evaluation were *never* removed during training.

Usage:
    python dataset_generation/make_topology_splits.py --bus 14
    python dataset_generation/make_topology_splits.py --bus 14 --test_frac 0.2 --seed 42
    # Run for all buses at once:
    for BUS in 14 30 57 118; do
        python dataset_generation/make_topology_splits.py --bus $BUS
    done
"""

import argparse
import json
import os
import random

import pandapower.networks as nw

NETWORK_MAP = {
    "14":  nw.case14,
    "30":  nw.case30,
    "57":  nw.case57,
    "118": nw.case118,
}


def main():
    parser = argparse.ArgumentParser(
        description="Partition line indices into train/test sets for topology-disjoint evaluation."
    )
    parser.add_argument("--bus", choices=["14", "30", "57", "118"], default="14",
                        help="IEEE bus system (default: 14)")
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Fraction of line indices reserved for the test partition (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write the JSON file (default: ../topology_splits/)")
    args = parser.parse_args()

    random.seed(args.seed)

    net = NETWORK_MAP[args.bus]()
    n_lines = len(net.line)

    indices = list(range(n_lines))
    random.shuffle(indices)

    n_test = max(1, int(round(n_lines * args.test_frac)))
    test_indices  = sorted(indices[:n_test])
    train_indices = sorted(indices[n_test:])

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "topology_splits"
    )
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{args.bus}Bus_topology_split.json")
    payload = {
        "bus": args.bus,
        "n_lines": n_lines,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "n_train_lines": len(train_indices),
        "n_test_lines": len(test_indices),
        "train_line_indices": train_indices,
        "test_line_indices": test_indices,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"IEEE {args.bus}-bus  ({n_lines} lines total)")
    print(f"  train partition : {len(train_indices)} lines  → {train_indices}")
    print(f"  test  partition : {len(test_indices)}  lines  → {test_indices}")
    print(f"  saved to        : {out_path}")


if __name__ == "__main__":
    main()
