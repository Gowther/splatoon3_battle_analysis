from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path


DEFAULT_MODEL_REGISTRY = ROOT / "config" / "models.json"


def load_model_registry(path: Path = DEFAULT_MODEL_REGISTRY) -> dict[str, Any]:
    target = project_path(path)
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file_report(entry: dict[str, Any], *, verify_hash: bool = False) -> dict[str, Any]:
    path = project_path(entry.get("path", ""))
    exists = path.exists()
    expected_sha = str(entry.get("expected_sha256", ""))
    actual_sha = file_sha256(path) if verify_hash and exists else ""
    hash_status = "not_checked"
    if verify_hash:
        hash_status = "passed" if expected_sha and actual_sha == expected_sha else "mismatch"
        if not expected_sha:
            hash_status = "missing_expected"
        if not exists:
            hash_status = "missing_file"

    return {
        "id": entry.get("id", ""),
        "area": entry.get("area", ""),
        "role": entry.get("role", ""),
        "path": entry.get("path", ""),
        "framework": entry.get("framework", ""),
        "file_type": entry.get("file_type", ""),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "hash_status": hash_status,
        "labels": entry.get("labels", ""),
        "training_status": entry.get("training_status", ""),
        "training_entrypoint": entry.get("training_entrypoint", ""),
        "promotion_gate": entry.get("promotion_gate", ""),
        "notes": entry.get("notes", ""),
    }


def build_model_registry_report(registry: dict[str, Any], *, verify_hash: bool = False) -> dict[str, Any]:
    entries = [model_file_report(entry, verify_hash=verify_hash) for entry in registry.get("models", [])]
    missing = [entry["id"] for entry in entries if not entry["exists"]]
    hash_mismatches = [entry["id"] for entry in entries if entry["hash_status"] == "mismatch"]
    status = "passed"
    if missing or hash_mismatches:
        status = "failed"
    return {
        "schema_version": 1,
        "status": status,
        "verify_hash": verify_hash,
        "registry_schema_version": registry.get("schema_version"),
        "model_count": len(entries),
        "missing_models": missing,
        "hash_mismatches": hash_mismatches,
        "models": entries,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Registry",
        "",
        f"- status: `{report.get('status')}`",
        f"- model_count: {report.get('model_count', 0)}",
        f"- verify_hash: {report.get('verify_hash')}",
        f"- missing_models: {', '.join(report.get('missing_models', [])) or '-'}",
        f"- hash_mismatches: {', '.join(report.get('hash_mismatches', [])) or '-'}",
        "",
        "| id | area | path | exists | size bytes | hash | training |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for entry in report.get("models", []):
        lines.append(
            f"| `{entry.get('id', '')}` | {entry.get('area', '')} | `{entry.get('path', '')}` | "
            f"{entry.get('exists')} | {entry.get('size_bytes', 0)} | {entry.get('hash_status', '')} | "
            f"{entry.get('training_status', '')} |"
        )
    lines.append("")
    return "\n".join(lines)
