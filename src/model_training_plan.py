from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path


DEFAULT_TRAINING_TARGETS = ROOT / "config" / "model_training_targets.json"


def load_training_targets(path: Path = DEFAULT_TRAINING_TARGETS) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def target_report(target: dict[str, Any]) -> dict[str, Any]:
    required = [str(path) for path in target.get("required_paths", [])]
    missing = [path for path in required if not project_path(path).exists()]
    return {
        "id": target.get("id", ""),
        "area": target.get("area", ""),
        "model_id": target.get("model_id", ""),
        "status": "ready" if not missing else "needs_data",
        "dataset_root": target.get("dataset_root", ""),
        "required_paths": required,
        "missing_paths": missing,
        "candidate_output_dir": target.get("candidate_output_dir", ""),
        "baseline_model": target.get("baseline_model", ""),
        "candidate_command": target.get("candidate_command", ""),
        "promotion_gate": target.get("promotion_gate", ""),
        "notes": target.get("notes", ""),
    }


def build_model_training_plan(config: dict[str, Any], *, target_ids: list[str] | None = None) -> dict[str, Any]:
    selected = set(target_ids or [])
    targets = [
        target_report(target)
        for target in config.get("targets", [])
        if not selected or target.get("id") in selected
    ]
    missing_ids = sorted(selected - {target["id"] for target in targets})
    status = "ready" if targets and all(target["status"] == "ready" for target in targets) else "needs_data"
    if missing_ids:
        status = "failed"
    return {
        "schema_version": 1,
        "status": status,
        "target_count": len(targets),
        "missing_target_ids": missing_ids,
        "targets": targets,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Training Plan",
        "",
        f"- status: `{report.get('status')}`",
        f"- target_count: {report.get('target_count', 0)}",
        f"- missing_target_ids: {', '.join(report.get('missing_target_ids', [])) or '-'}",
        "",
        "| id | area | status | missing paths |",
        "| --- | --- | --- | ---: |",
    ]
    for target in report.get("targets", []):
        lines.append(
            f"| `{target.get('id', '')}` | {target.get('area', '')} | `{target.get('status', '')}` | "
            f"{len(target.get('missing_paths', []))} |"
        )
    for target in report.get("targets", []):
        lines.extend(
            [
                "",
                f"## {target.get('id', '')}",
                "",
                f"- model_id: `{target.get('model_id', '')}`",
                f"- dataset_root: `{target.get('dataset_root', '')}`",
                f"- candidate_output_dir: `{target.get('candidate_output_dir', '')}`",
                f"- baseline_model: `{target.get('baseline_model', '')}`",
                f"- promotion_gate: `{target.get('promotion_gate', '')}`",
                f"- candidate_command: `{target.get('candidate_command', '')}`",
                f"- notes: {target.get('notes', '')}",
            ]
        )
        if target.get("missing_paths"):
            lines.extend(["", "Missing paths:"])
            lines.extend(f"- `{path}`" for path in target["missing_paths"])
    lines.append("")
    return "\n".join(lines)
