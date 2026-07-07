from __future__ import annotations

import json
from pathlib import Path
from typing import Any


THEME_RULES = [
    (
        "manual_heatmap_labels",
        (
            "annotation_round",
            "annotation_ui",
            "annotation_samples",
            "prepare_heatmap_annotation_round",
            "build_heatmap_annotation_ui",
            "heatmap_annotation",
        ),
    ),
    (
        "heatmap_tuning_and_quality",
        (
            "heatmap/tuning",
            "parameter_experiment",
            "suggest_heatmap_tuning",
            "heatmap_quality",
            "heatmap_config",
            "config_f_match",
            "config_overhead",
            "render_heatmaps",
            "infer_player_tracks",
            "run_pipeline",
        ),
    ),
    (
        "runtime_and_benchmarking",
        (
            "runtime_report",
            "run_with_runtime_report",
            "model_benchmark",
            "benchmark_model_experiments",
            "experiment_manifest",
            "runtime_benchmark",
        ),
    ),
    (
        "stage_coordinate_normalization",
        (
            "stage_coordinates",
            "stage coordinate",
            "coordinate_normalization",
            "report_stage_coordinates",
        ),
    ),
    (
        "validation_and_governance",
        (
            "validation_suite",
            "validation_sample",
            "dataset_governance",
            "project_hygiene",
            "sample_intake",
            "data_registry",
            "evaluation_matches",
            "check_project",
            "project_check_registry",
        ),
    ),
    (
        "analysis_pipeline",
        (
            "run_analysis",
            "model_error_report",
            "count_smoothing",
            "report_model_errors",
            "evaluate_matches",
        ),
    ),
    (
        "documentation",
        (
            "PROJECT_",
            "DATA_AND_TRAINING",
            "README",
        ),
    ),
]


def parse_status_line(line: str) -> dict[str, str]:
    status = line[:2].strip() or "??"
    path = line[3:].strip()
    return {"status": status, "path": path}


def parse_git_status(text: str) -> list[dict[str, str]]:
    return [parse_status_line(line) for line in text.splitlines() if line.strip()]


def change_group(path: str) -> str:
    if path.startswith("src/"):
        return "runtime_code"
    if path.startswith("scripts/"):
        return "cli_tools"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("config/"):
        return "config"
    if path.endswith(".md"):
        return "docs"
    if path.startswith("outputs/"):
        return "outputs"
    if path.startswith("yolov5/"):
        return "legacy_yolov5_review_separately"
    return "other"


def change_theme(path: str) -> str:
    if path.startswith("yolov5/"):
        return "legacy_yolov5_review_separately"
    lowered = path.lower()
    for theme, needles in THEME_RULES:
        if any(needle.lower() in lowered for needle in needles):
            return theme
    return change_group(path)


def group_changes(changes: list[dict[str, str]], key_name: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for change in changes:
        key = change_theme(change["path"]) if key_name == "theme" else change_group(change["path"])
        groups.setdefault(key, []).append(change)
    return groups


def summary_counts(changes: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_theme: dict[str, int] = {}
    for change in changes:
        by_status[change["status"]] = by_status.get(change["status"], 0) + 1
        theme = change_theme(change["path"])
        by_theme[theme] = by_theme.get(theme, 0) + 1
    return {"by_status": by_status, "by_theme": by_theme}


def risk_flags(changes: list[dict[str, str]]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if any(change["path"].startswith("yolov5/") for change in changes):
        flags.append(
            {
                "severity": "medium",
                "area": "legacy_yolov5_review_separately",
                "message": "Legacy YOLOv5 files are dirty; review or commit separately from project refactors.",
            }
        )
    untracked_count = sum(1 for change in changes if change["status"] == "??")
    if untracked_count >= 20:
        flags.append(
            {
                "severity": "medium",
                "area": "worktree_size",
                "message": f"{untracked_count} untracked paths are present; split review into theme batches.",
            }
        )
    if any(change["path"].startswith("config/") for change in changes):
        flags.append(
            {
                "severity": "low",
                "area": "config",
                "message": "Config and registry files changed; run registry/config validation before committing.",
            }
        )
    return flags


def build_change_package(status_text: str, verification: list[str] | None = None) -> dict[str, Any]:
    changes = parse_git_status(status_text)
    return {
        "schema_version": 1,
        "status": "ready",
        "change_count": len(changes),
        "summary": summary_counts(changes),
        "groups": group_changes(changes, "directory"),
        "themes": group_changes(changes, "theme"),
        "risk_flags": risk_flags(changes),
        "commit_batches": [
            {"name": "validation_governance", "themes": ["validation_and_governance", "analysis_pipeline"]},
            {"name": "heatmap_labeling_and_tuning", "themes": ["manual_heatmap_labels", "heatmap_tuning_and_quality"]},
            {"name": "runtime_benchmarking", "themes": ["runtime_and_benchmarking"]},
            {"name": "stage_coordinates", "themes": ["stage_coordinate_normalization"]},
            {"name": "docs_and_delivery", "themes": ["documentation", "docs"]},
            {"name": "legacy_review_separately", "themes": ["legacy_yolov5_review_separately"]},
        ],
        "verification": verification or [],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Refactor Change Package",
        "",
        f"- status: `{report.get('status')}`",
        f"- changed paths: {report.get('change_count', 0)}",
        "",
        "## Summary",
        "",
        f"- by_status: {json.dumps(report.get('summary', {}).get('by_status', {}), ensure_ascii=False)}",
        f"- by_theme: {json.dumps(report.get('summary', {}).get('by_theme', {}), ensure_ascii=False)}",
        "",
        "## Risk Flags",
        "",
    ]
    for flag in report.get("risk_flags", []):
        lines.append(f"- `{flag['severity']}` {flag['area']}: {flag['message']}")
    if not report.get("risk_flags"):
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Themes",
            "",
        ]
    )
    for theme, changes in sorted(report.get("themes", {}).items()):
        lines.extend([f"### {theme}", ""])
        for change in changes:
            lines.append(f"- `{change['status']}` {change['path']}")
        lines.append("")
    lines.extend(
        [
            "## Directory Groups",
            "",
        ]
    )
    for group, changes in sorted(report.get("groups", {}).items()):
        lines.extend([f"### {group}", ""])
        for change in changes:
            lines.append(f"- `{change['status']}` {change['path']}")
        lines.append("")
    lines.extend(["## Suggested Commit Batches", ""])
    for batch in report.get("commit_batches", []):
        lines.append(f"- `{batch['name']}`: {', '.join(batch['themes'])}")
    lines.extend(["", "## Verification", ""])
    verification = report.get("verification", [])
    lines.extend(f"- {item}" for item in verification)
    if not verification:
        lines.append("- pending")
    lines.append("")
    return "\n".join(lines)
