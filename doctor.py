"""Check whether this checkout can reproduce the ScienceWorld-MAS workflow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_DATA = {
    "expert_high-data.json": "59cf6b2e78445fae67032b17ff391314c7e52ac7db8d5078c4ac1d1322e9e441",
    "expert_low-data.json": "da9c35f82d1dae5c363b13c53572d50af07b453fd78457a9dc45f381ec39d29b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/multisquare/ScienceWorld")
    parser.add_argument("--model", default=None, help="Optional local model path or HF model ID.")
    parser.add_argument("--smoke-environment", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    checks = {}
    checks["python_supported"] = sys.version_info[:2] in {(3, 10), (3, 11)}
    checks["java_available"] = shutil.which("java") is not None
    checks["packages"] = {
        name: package_version(name)
        for name in (
            "scienceworld",
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "huggingface_hub",
        )
    }
    checks["required_packages_installed"] = all(checks["packages"].values())

    data_checks = {}
    for filename, expected in EXPECTED_DATA.items():
        path = data_dir / filename
        actual = sha256(path) if path.exists() else None
        data_checks[filename] = {
            "exists": path.exists(),
            "sha256": actual,
            "matches_expected": actual == expected,
        }
    checks["data"] = data_checks
    checks["data_valid"] = all(item["matches_expected"] for item in data_checks.values())

    if args.model:
        model_path = Path(args.model)
        checks["model"] = {
            "value": args.model,
            "local_path_exists": model_path.exists(),
            "looks_like_local_model": (model_path / "config.json").exists() if model_path.exists() else False,
        }

    if checks["java_available"]:
        java = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        checks["java_version"] = (java.stderr or java.stdout).splitlines()[0]

    smoke = None
    if args.smoke_environment and checks["packages"]["scienceworld"]:
        try:
            from scienceworld_env import EpisodeSpec, ScienceWorldRunner

            runner = ScienceWorldRunner(step_limit=5)
            spec = EpisodeSpec("boil", 0, "train")
            observation, task, _ = runner.reset(spec)
            _, reward, done, info, action_valid = runner.step("look around")
            runner.close()
            smoke = {
                "ok": True,
                "task_count": len(runner.task_names),
                "task": task,
                "initial_observation_length": len(observation),
                "look_reward": reward,
                "look_done": done,
                "look_score": info.get("score"),
                "look_action_valid": action_valid,
            }
        except Exception as exc:
            smoke = {"ok": False, "error": repr(exc)}
    checks["environment_smoke"] = smoke

    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "checks": checks,
    }
    report["ready"] = (
        checks["python_supported"]
        and checks["java_available"]
        and checks["required_packages_installed"]
        and checks["data_valid"]
        and (smoke is None or smoke["ok"])
    )
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

