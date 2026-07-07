from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_registry import display_path
from src.heatmap.annotation_eval import evaluate_annotations, evaluate_gates
from src.heatmap.annotation_samples import export_annotation_package


def build_quality_loop_report(
    registry: dict[str, Any],
    *,
    package_dir: Path | None = None,
    annotation_csv: Path | None = None,
    frames_per_match: int = 5,
    match_ids: list[str] | None = None,
    export_package: bool = False,
    threshold_px: float = 80.0,
    min_recall: float | None = None,
    max_mean_error_px: float | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    if export_package:
        if package_dir is None:
            raise ValueError("package_dir is required when export_package is true")
        manifest = export_annotation_package(
            registry,
            package_dir,
            match_ids=match_ids,
            frames_per_match=frames_per_match,
        )
        annotation_csv = annotation_csv or package_dir / "annotation_template.csv"

    annotation_path = annotation_csv.expanduser() if annotation_csv else None
    metrics: dict[str, Any] | None = None
    checks: dict[str, dict[str, Any]] = {}
    status = "needs_labels"

    if annotation_path and annotation_path.exists():
        metrics = evaluate_annotations(annotation_path, registry, threshold_px=threshold_px)
        checks = evaluate_gates(metrics, min_recall=min_recall, max_mean_error_px=max_mean_error_px)
        failed_checks = any(not check["ok"] for check in checks.values())
        if metrics["status"] == "no_labels":
            status = "needs_labels"
        elif failed_checks:
            status = "needs_review"
        else:
            status = "passed"

    return {
        "status": status,
        "package_dir": display_path(package_dir) if package_dir else "",
        "annotation_csv": display_path(annotation_path) if annotation_path else "",
        "manifest": manifest,
        "metrics": metrics or {},
        "checks": checks,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "# Heatmap Quality Loop",
        "",
        f"- status: `{report.get('status')}`",
        f"- package_dir: `{report.get('package_dir', '')}`",
        f"- annotation_csv: `{report.get('annotation_csv', '')}`",
        "",
        "## Annotation Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "annotation_rows",
        "labeled_rows",
        "complete_frame_team_groups",
        "matched_labels",
        "missed_labels",
        "false_positive_predictions",
        "recall",
        "precision_on_complete_groups",
        "mean_error_px",
        "median_error_px",
        "p90_error_px",
    ):
        lines.append(f"| {key} | {metrics.get(key, '')} |")

    lines.extend(["", "## Quality Gates", "", "| check | expected | actual | status |", "| --- | --- | --- | --- |"])
    checks = report.get("checks", {})
    if checks:
        for key, check in checks.items():
            status = "passed" if check["ok"] else "failed"
            lines.append(f"| {key} | {check['expected']} | {check['actual']} | {status} |")
    else:
        lines.append("| none |  |  | passed |")

    manifest = report.get("manifest") or {}
    matches = manifest.get("matches", [])
    lines.extend(["", "## Annotation Package", "", f"- matches: {len(matches)}"])
    for match in matches:
        lines.append(f"- `{match.get('match_id')}` frames={len(match.get('frames', []))} rows={match.get('annotation_rows')}")

    lines.append("")
    return "\n".join(lines)
