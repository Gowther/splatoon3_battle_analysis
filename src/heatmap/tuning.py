from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_registry import display_path, load_registry
from src.heatmap.annotation_eval import evaluate_annotations, evaluate_gates


def load_optional_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    target = path.expanduser()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def add_recommendation(items: list[dict[str, str]], priority: str, area: str, action: str, reason: str) -> None:
    items.append({"priority": priority, "area": area, "action": action, "reason": reason})


def recommendations_from_metrics(metrics: dict[str, Any], checks: dict[str, Any]) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if metrics.get("status") in (None, "no_labels"):
        add_recommendation(
            recommendations,
            "high",
            "labels",
            "Fill manual x/y labels in the annotation template and mark complete frame/team groups.",
            "Tuning cannot distinguish misses from false positives until at least one labeled batch exists.",
        )
        return recommendations

    recall = metrics.get("recall")
    precision = metrics.get("precision_on_complete_groups")
    mean_error = metrics.get("mean_error_px")
    missed = int(metrics.get("missed_labels") or 0)
    false_positive = int(metrics.get("false_positive_predictions") or 0)

    if recall is not None and recall < 0.9:
        add_recommendation(
            recommendations,
            "high",
            "marker_detection",
            "Probe lower `marker_detection.min_confidence` and wider `marker_detection.label_proximity_px` on the labeled frames.",
            f"Recall is {recall}; {missed} manual labels were missed.",
        )
    if precision is not None and precision < 0.85:
        add_recommendation(
            recommendations,
            "medium",
            "point_cleaning",
            "Probe higher `point_cleaning.min_confidence` or stricter `point_cleaning.max_points_per_team_per_frame`.",
            f"Precision on complete groups is {precision}; {false_positive} unmatched predictions remain.",
        )
    if mean_error is not None and mean_error > 60:
        add_recommendation(
            recommendations,
            "medium",
            "map_geometry",
            "Inspect map mask and label boxes on high-error frames before widening thresholds.",
            f"Mean label-to-prediction error is {mean_error}px.",
        )
    for key, check in checks.items():
        if not check.get("ok"):
            add_recommendation(
                recommendations,
                "high",
                "quality_gate",
                f"Resolve failed gate `{key}` before promoting tracker defaults.",
                f"Expected {check.get('expected')}, actual {check.get('actual')}.",
            )
    if not recommendations:
        add_recommendation(
            recommendations,
            "low",
            "defaults",
            "Keep current heatmap defaults for the labeled batch and expand labels to more colors/maps.",
            "The available labeled metrics did not trigger recall, precision, or error warnings.",
        )
    return recommendations


def candidate_matrix() -> list[dict[str, str]]:
    return [
        {
            "config_key": "marker_detection.min_confidence",
            "direction": "down when recall is low; up when false positives dominate",
            "risk": "Too low admits ink blobs near labels.",
        },
        {
            "config_key": "marker_detection.label_proximity_px",
            "direction": "up when labels are missed near player names",
            "risk": "Too high can attach unrelated colored components to a name label.",
        },
        {
            "config_key": "point_cleaning.merge_distance_px",
            "direction": "up when duplicate detections cluster around one player",
            "risk": "Too high can merge nearby teammates.",
        },
        {
            "config_key": "point_cleaning.max_track_step_px",
            "direction": "up when real movement becomes `jump_reset`; down when identity swaps increase",
            "risk": "Too high can connect different players across crowded frames.",
        },
        {
            "config_key": "teams.*.hsv_ranges",
            "direction": "regenerate via auto color calibration when color drift causes misses",
            "risk": "Bad calibration can swap teams or include UI ink patches.",
        },
    ]


def build_tuning_report(
    *,
    registry_path: Path,
    annotation_csv: Path,
    threshold_px: float = 80.0,
    min_recall: float | None = None,
    max_mean_error_px: float | None = None,
    heatmap_comparison_json: Path | None = None,
) -> dict[str, Any]:
    annotation_path = annotation_csv.expanduser()
    metrics: dict[str, Any]
    checks: dict[str, Any] = {}
    if annotation_path.exists():
        registry = load_registry(registry_path)
        metrics = evaluate_annotations(annotation_path, registry, threshold_px=threshold_px)
        checks = evaluate_gates(metrics, min_recall=min_recall, max_mean_error_px=max_mean_error_px)
    else:
        metrics = {
            "status": "no_labels",
            "annotation_path": display_path(annotation_path),
            "threshold_px": threshold_px,
            "annotation_rows": 0,
            "labeled_rows": 0,
        }
    comparison = load_optional_json(heatmap_comparison_json)
    recommendations = recommendations_from_metrics(metrics, checks)
    if comparison:
        aggregate = comparison.get("aggregate", {})
        anomaly_counts = aggregate.get("anomaly_counts", {})
        if anomaly_counts.get("jump_reset") or anomaly_counts.get("track_gap"):
            add_recommendation(
                recommendations,
                "medium",
                "identity_tracking",
                "Use anomaly exports beside manual labels when tuning `point_cleaning.max_track_step_px`.",
                f"Comparison anomalies: {json.dumps(anomaly_counts, ensure_ascii=False)}.",
            )
    status = "needs_labels" if metrics.get("status") == "no_labels" else ("needs_review" if any(not check.get("ok") for check in checks.values()) else "ready")
    return {
        "schema_version": 1,
        "status": status,
        "annotation_csv": display_path(annotation_path),
        "metrics": metrics,
        "checks": checks,
        "recommendations": recommendations,
        "candidate_matrix": candidate_matrix(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "# Heatmap Tuning Suggestions",
        "",
        f"- status: `{report.get('status')}`",
        f"- annotation_csv: `{report.get('annotation_csv', '')}`",
        f"- labeled_rows: {metrics.get('labeled_rows', '')}",
        f"- recall: {metrics.get('recall', '')}",
        f"- precision_on_complete_groups: {metrics.get('precision_on_complete_groups', '')}",
        f"- mean_error_px: {metrics.get('mean_error_px', '')}",
        "",
        "## Recommendations",
        "",
        "| priority | area | action | reason |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("recommendations", []):
        lines.append(f"| {item['priority']} | {item['area']} | {item['action']} | {item['reason']} |")
    lines.extend(["", "## Candidate Matrix", "", "| config key | direction | risk |", "| --- | --- | --- |"])
    for item in report.get("candidate_matrix", []):
        lines.append(f"| `{item['config_key']}` | {item['direction']} | {item['risk']} |")
    lines.append("")
    return "\n".join(lines)
