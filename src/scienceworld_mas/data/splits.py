"""Deterministic grouping and split helpers."""

from __future__ import annotations

import hashlib
import re


def strip_embedded_instruction(text: str, marker: str) -> str:
    position = text.rfind(marker)
    if position < 0:
        return text.strip()
    return text[position + len(marker) :].strip()


def normalize_group_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def task_family(task_description: str) -> str:
    task = strip_embedded_instruction(task_description, "Task Description:")
    return normalize_group_text(task)


def assign_split(
    group_key: str,
    *,
    seed: int = 123,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> str:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("ratios must satisfy train > 0, val >= 0, and train + val < 1")
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"
