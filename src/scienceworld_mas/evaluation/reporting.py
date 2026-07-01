"""Evaluation report serialization."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import EvaluationReport


def write_evaluation_report(report: EvaluationReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output
