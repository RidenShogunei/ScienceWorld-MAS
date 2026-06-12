"""Download the Multi-Square ScienceWorld expert dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "sangeun-park/Multi-Square"
FILES = (
    "ScienceWorld/expert_high-data.json",
    "ScienceWorld/expert_low-data.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=args.revision,
        )
        destination = output_dir / Path(filename).name
        shutil.copy2(downloaded, destination)
        print(f"[download] {filename} -> {destination}")


if __name__ == "__main__":
    main()

