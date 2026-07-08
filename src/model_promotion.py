from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path
from src.model_registry import file_sha256, load_model_registry, save_model_registry


DEFAULT_PROMOTION_BACKUP_DIR = ROOT / "outputs" / "model_promotion_backups"
PASSING_VALIDATION_STATUSES = {"passed", "ready"}


def find_model_entry(registry: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for entry in registry.get("models", []):
        if entry.get("id") == model_id:
            return entry
    return None


def read_validation_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "status": "", "payload_status": ""}
    target = project_path(path)
    if not target.exists():
        return {"path": str(path), "exists": False, "status": "missing", "payload_status": ""}
    with target.open(encoding="utf-8") as f:
        payload = json.load(f)
    payload_status = str(payload.get("status", ""))
    status = "passed" if payload_status in PASSING_VALIDATION_STATUSES else "failed"
    return {
        "path": str(path),
        "exists": True,
        "status": status,
        "payload_status": payload_status,
    }


def build_model_promotion_plan(
    registry: dict[str, Any],
    *,
    model_id: str,
    candidate_path: Path,
    validation_report: Path | None = None,
    backup_dir: Path = DEFAULT_PROMOTION_BACKUP_DIR,
) -> dict[str, Any]:
    entry = find_model_entry(registry, model_id)
    candidate = project_path(candidate_path)
    blockers: list[str] = []
    warnings: list[str] = []
    validation = read_validation_status(validation_report)

    if entry is None:
        blockers.append(f"model id is not registered: {model_id}")
        target_path = Path("")
        expected_suffix = ""
        promotion_gate = ""
    else:
        target_path = project_path(str(entry.get("path", "")))
        expected_suffix = f".{entry.get('file_type', '')}".strip(".")
        expected_suffix = f".{expected_suffix}" if expected_suffix else target_path.suffix
        promotion_gate = str(entry.get("promotion_gate", ""))

    if not candidate.exists() or not candidate.is_file():
        blockers.append(f"candidate model file is missing: {candidate_path}")
    if entry is not None and expected_suffix and candidate.suffix != expected_suffix:
        blockers.append(f"candidate suffix {candidate.suffix or '(none)'} does not match expected {expected_suffix}")
    if entry is not None and candidate.exists() and candidate.resolve() == target_path.resolve():
        blockers.append("candidate path is already the registered model path")

    if validation_report is None and promotion_gate:
        warnings.append(f"validation report is required before apply; run {promotion_gate} first")
    elif validation["status"] == "missing":
        blockers.append(f"validation report is missing: {validation_report}")
    elif validation["status"] == "failed":
        blockers.append(f"validation report status is not passing: {validation['payload_status'] or '(empty)'}")

    candidate_sha = file_sha256(candidate) if candidate.exists() and candidate.is_file() else ""
    target_exists = bool(entry is not None and target_path.exists())
    target_sha = file_sha256(target_path) if target_exists else ""
    status = "ready"
    if blockers:
        status = "failed"
    elif validation_report is None and promotion_gate:
        status = "needs_validation"

    return {
        "schema_version": 1,
        "status": status,
        "model_id": model_id,
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "target_path": str(entry.get("path", "")) if entry else "",
        "target_exists": target_exists,
        "target_sha256": target_sha,
        "backup_dir": str(backup_dir),
        "expected_suffix": expected_suffix if entry else "",
        "promotion_gate": promotion_gate if entry else "",
        "validation": validation,
        "blockers": blockers,
        "warnings": warnings,
        "actions": [
            "copy candidate model to registered path",
            "update registry expected_sha256",
            "write backup of existing registered model when present",
        ],
    }


def apply_model_promotion(
    registry_path: Path,
    plan: dict[str, Any],
    *,
    backup_dir: Path = DEFAULT_PROMOTION_BACKUP_DIR,
) -> dict[str, Any]:
    if plan.get("status") != "ready":
        raise ValueError(f"promotion plan is not ready: {plan.get('status')}")

    registry = load_model_registry(registry_path)
    entry = find_model_entry(registry, str(plan.get("model_id", "")))
    if entry is None:
        raise ValueError(f"model id is not registered: {plan.get('model_id')}")

    candidate = project_path(str(plan.get("candidate_path", "")))
    target = project_path(str(entry.get("path", "")))
    resolved_backup_dir = project_path(backup_dir)
    backup_path = Path("")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        resolved_backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = resolved_backup_dir / f"{target.name}.{plan.get('target_sha256', '')[:12]}.bak"
        shutil.copy2(target, backup_path)

    shutil.copy2(candidate, target)
    entry["expected_sha256"] = str(plan.get("candidate_sha256", ""))
    save_model_registry(registry, registry_path)

    promoted = dict(plan)
    promoted["status"] = "promoted"
    promoted["backup_path"] = str(backup_path) if backup_path else ""
    promoted["applied_registry"] = str(registry_path)
    return promoted


def render_markdown(report: dict[str, Any]) -> str:
    validation = report.get("validation", {})
    lines = [
        "# Model Promotion Plan",
        "",
        f"- status: `{report.get('status')}`",
        f"- model_id: `{report.get('model_id', '')}`",
        f"- candidate_path: `{report.get('candidate_path', '')}`",
        f"- target_path: `{report.get('target_path', '')}`",
        f"- candidate_sha256: `{report.get('candidate_sha256', '')}`",
        f"- target_exists: {report.get('target_exists')}",
        f"- promotion_gate: `{report.get('promotion_gate', '')}`",
        f"- validation_status: `{validation.get('payload_status', '') or validation.get('status', '')}`",
        f"- apply_error: {report.get('apply_error', '-')}",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers", [])
    lines.extend([f"- {item}" for item in blockers] or ["- -"])
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend([f"- {item}" for item in warnings] or ["- -"])
    lines.extend(["", "## Apply Actions", ""])
    lines.extend([f"- {item}" for item in report.get("actions", [])] or ["- -"])
    lines.append("")
    return "\n".join(lines)
