"""Build v2 System1/System2 transition datasets from Multi-Square files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .multisquare import load_high_trajectories, load_low_trajectories
from .transitions import build_transition_datasets, write_transition_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    high = load_high_trajectories(input_dir / "expert_high-data.json")
    low = load_low_trajectories(input_dir / "expert_low-data.json")
    datasets = build_transition_datasets(
        high,
        low,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_transition_datasets(datasets, args.output_dir)
    print(json.dumps(datasets.manifest.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
