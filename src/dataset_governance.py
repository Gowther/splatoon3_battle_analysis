from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path
from src.data_registry import DEFAULT_REGISTRY, display_path, load_registry
from src.weapon_training import summarize_dataset


def registry_metadata_issues(registry: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for match in registry.get("matches", []):
        match_id = str(match.get("id", ""))
        if not match.get("purpose"):
            issues.append({"match_id": match_id, "field": "purpose", "detail": "missing purpose list"})
        if not str(match.get("notes", "")).strip():
            issues.append({"match_id": match_id, "field": "notes", "detail": "missing notes"})
        for window in match.get("analysis_windows", []):
            for field in ("id", "start_seconds", "stop_seconds", "sample_fps", "device"):
                if window.get(field) in (None, ""):
                    issues.append({"match_id": match_id, "field": f"analysis_windows.{field}", "detail": "missing analysis window field"})
        heatmap = match.get("heatmap") if isinstance(match.get("heatmap"), dict) else None
        if heatmap:
            for field in ("config", "report", "player_tracks", "teams", "quality_gates"):
                if heatmap.get(field) in (None, "", []):
                    issues.append({"match_id": match_id, "field": f"heatmap.{field}", "detail": "missing heatmap field"})
    return issues


def small_class_counts(class_counts: dict[str, int], min_images_per_class: int) -> dict[str, int]:
    return {label: count for label, count in sorted(class_counts.items()) if count < min_images_per_class}


def build_dataset_governance_report(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    dataset: Path = ROOT / "main_training_dataset",
    labels: Path = ROOT / "main_weapon_list.txt",
    model: Path | None = ROOT / "models" / "main_weapons_classification_weight.pth",
    min_images_per_class: int = 20,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    weapon_summary = summarize_dataset(
        project_path(dataset),
        project_path(labels),
        project_path(model) if model else None,
    )
    metadata_issues = registry_metadata_issues(registry)
    small_classes = small_class_counts(weapon_summary.class_counts, min_images_per_class)
    status = "passed"
    if not weapon_summary.ok or metadata_issues:
        status = "failed"
    elif small_classes:
        status = "needs_review"
    return {
        "status": status,
        "registry": display_path(project_path(registry_path)),
        "dataset": display_path(project_path(dataset)),
        "labels": display_path(project_path(labels)),
        "model": display_path(project_path(model)) if model else "",
        "min_images_per_class": min_images_per_class,
        "weapon_dataset": asdict(weapon_summary),
        "small_classes": small_classes,
        "registry_metadata_issues": metadata_issues,
        "summary": {
            "images": weapon_summary.images,
            "dataset_classes": weapon_summary.dataset_classes,
            "label_classes": weapon_summary.label_classes,
            "small_class_count": len(small_classes),
            "registry_metadata_issue_count": len(metadata_issues),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    weapon = report["weapon_dataset"]
    lines = [
        "# Dataset Governance Report",
        "",
        f"- status: `{report['status']}`",
        f"- registry: `{report['registry']}`",
        f"- dataset: `{report['dataset']}`",
        f"- labels: `{report['labels']}`",
        f"- images: {summary['images']}",
        f"- dataset_classes: {summary['dataset_classes']}",
        f"- label_classes: {summary['label_classes']}",
        f"- small_classes: {summary['small_class_count']}",
        f"- registry_metadata_issues: {summary['registry_metadata_issue_count']}",
        "",
        "## Label Alignment",
        "",
        f"- missing_dataset_classes: {len(weapon['missing_dataset_classes'])}",
        f"- missing_label_classes: {len(weapon['missing_label_classes'])}",
        f"- duplicate_labels: {len(weapon['duplicate_labels'])}",
        f"- model_output_classes: {weapon['model_output_classes']}",
        "",
        "## Small Classes",
        "",
    ]
    small_classes = report["small_classes"]
    if small_classes:
        for label, count in list(small_classes.items())[:50]:
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Registry Metadata Issues", ""])
    issues = report["registry_metadata_issues"]
    if issues:
        for issue in issues[:50]:
            lines.append(f"- `{issue['match_id']}` {issue['field']}: {issue['detail']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
