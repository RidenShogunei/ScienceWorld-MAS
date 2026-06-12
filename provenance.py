"""Portable experiment provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "scienceworld",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "huggingface_hub",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root or Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def portable_reference(value: str, root: Path | None = None) -> dict:
    root = (root or Path(__file__).parent).resolve()
    path = Path(value)
    if not path.exists():
        return {"type": "huggingface_id", "value": value}

    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(root))
        reference_type = "repository_path"
    except ValueError:
        display = resolved.name
        reference_type = "external_local_path"

    config_path = resolved / "config.json"
    adapter_path = resolved / "adapter_config.json"
    identity_file = config_path if config_path.exists() else adapter_path
    reference = {
        "type": reference_type,
        "value": display,
        "identity_file": identity_file.name if identity_file.exists() else None,
        "identity_sha256": file_sha256(identity_file) if identity_file.exists() else None,
    }
    adapter_weights = resolved / "adapter_model.safetensors"
    if adapter_weights.exists():
        reference["weights_file"] = adapter_weights.name
        reference["weights_sha256"] = file_sha256(adapter_weights)
    return reference


def experiment_provenance(references: dict[str, str] | None = None) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "packages": versions,
        "references": {
            name: portable_reference(value) for name, value in (references or {}).items()
        },
    }
