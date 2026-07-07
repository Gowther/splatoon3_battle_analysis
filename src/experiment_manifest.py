from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from src.data_registry import display_path, resolve_project_path


DEFAULT_SOURCES = [
    ("data_registry", "config/data_registry.json"),
    ("evaluation_matches", "config/evaluation_matches.json"),
    ("annotation_samples", "config/annotation_samples.json"),
    ("model_experiments", "config/model_experiments.json"),
]


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(label: str, path_value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_value) or Path(path_value).expanduser()
    exists = path.exists()
    record: dict[str, Any] = {
        "label": label,
        "path": display_path(path),
        "exists": exists,
        "kind": "dir" if path.is_dir() else "file",
    }
    if exists and path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    return record


def parse_labeled_path(value: str) -> tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, path
    path = value
    return Path(value).stem, path


def build_experiment_manifest(
    *,
    experiment_id: str,
    sources: list[tuple[str, str | Path]] | None = None,
    artifacts: list[tuple[str, str | Path]] | None = None,
    verification: list[str] | None = None,
    notes: list[str] | None = None,
    git_status: str | None = None,
) -> dict[str, Any]:
    source_items = sources if sources is not None else [(label, path) for label, path in DEFAULT_SOURCES]
    artifact_items = artifacts or []
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "generated_at": now_iso(),
        "status": "ready",
        "sources": [file_record(label, path) for label, path in source_items],
        "artifacts": [file_record(label, path) for label, path in artifact_items],
        "verification": verification or [],
        "notes": notes or [],
        "git_status": git_status or "",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Experiment Manifest",
        "",
        f"- experiment_id: `{manifest.get('experiment_id')}`",
        f"- status: `{manifest.get('status')}`",
        f"- generated_at: `{manifest.get('generated_at')}`",
        "",
        "## Sources",
        "",
        "| label | exists | size | sha256 | path |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in manifest.get("sources", []):
        lines.append(
            f"| {item['label']} | {item['exists']} | {item.get('size_bytes', '')} | "
            f"`{str(item.get('sha256', ''))[:12]}` | `{item['path']}` |"
        )
    lines.extend(["", "## Artifacts", "", "| label | exists | size | sha256 | path |", "| --- | --- | ---: | --- | --- |"])
    for item in manifest.get("artifacts", []):
        lines.append(
            f"| {item['label']} | {item['exists']} | {item.get('size_bytes', '')} | "
            f"`{str(item.get('sha256', ''))[:12]}` | `{item['path']}` |"
        )
    lines.extend(["", "## Verification", ""])
    lines.extend(f"- {item}" for item in manifest.get("verification", []))
    if not manifest.get("verification"):
        lines.append("- pending")
    if manifest.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {item}" for item in manifest.get("notes", []))
    if manifest.get("git_status"):
        lines.extend(["", "## Git Status", "", "```text", manifest["git_status"].rstrip(), "```"])
    lines.append("")
    return "\n".join(lines)
