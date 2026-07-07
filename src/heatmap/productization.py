from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_registry import display_path, resolve_project_path


def load_optional_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    target = resolve_project_path(path) or path.expanduser()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def labeled_rows(annotation_round: dict[str, Any] | None) -> int:
    return int((annotation_round or {}).get("progress", {}).get("labeled_rows") or 0)


def heatmap_anomalies(heatmap_comparison: dict[str, Any] | None) -> dict[str, Any]:
    return (heatmap_comparison or {}).get("aggregate", {}).get("anomaly_counts", {})


def readiness_items(
    annotation_round: dict[str, Any] | None,
    heatmap_comparison: dict[str, Any] | None,
    runtime_benchmarks: dict[str, Any] | None,
) -> list[dict[str, str]]:
    labels = labeled_rows(annotation_round)
    anomalies = heatmap_anomalies(heatmap_comparison)
    runtime_status = (runtime_benchmarks or {}).get("status", "missing")
    return [
        {
            "area": "manual_accuracy",
            "status": "ready" if labels > 0 else "blocked",
            "detail": f"{labels} labeled rows available.",
        },
        {
            "area": "tracker_stability",
            "status": "needs_review" if anomalies.get("jump_reset") or anomalies.get("track_gap") else "ready",
            "detail": json.dumps(anomalies, ensure_ascii=False) if anomalies else "No anomaly aggregate available.",
        },
        {
            "area": "runtime_baseline",
            "status": "ready" if runtime_status in {"ready", "partial"} else "planned",
            "detail": f"Runtime benchmark status: {runtime_status}.",
        },
        {
            "area": "coordinate_normalization",
            "status": "planned",
            "detail": "Stage-map homography is not implemented; current coordinates remain source-video pixels.",
        },
        {
            "area": "event_join",
            "status": "planned",
            "detail": "Event CSV join exists, but real kill/death event sources are not yet registered.",
        },
    ]


def milestones() -> list[dict[str, str]]:
    return [
        {
            "id": "label_gate",
            "goal": "Fill first heatmap annotation round and pass recall/error gates.",
            "exit_criteria": "Quality loop has labeled rows, recall target, and bounded mean error.",
        },
        {
            "id": "tracker_parameter_baseline",
            "goal": "Run parameter experiments against manual labels.",
            "exit_criteria": "One candidate improves recall or stability without worsening false positives.",
        },
        {
            "id": "stage_map_normalization",
            "goal": "Map video-pixel points to normalized stage-map coordinates.",
            "exit_criteria": "At least one stage has checked homography/control points and rendered normalized heatmaps.",
        },
        {
            "id": "event_and_identity_confidence",
            "goal": "Separate team-slot routes from verified player identity and optional event joins.",
            "exit_criteria": "Reports show confidence/limitations for player routes and event proximity.",
        },
        {
            "id": "product_report_surface",
            "goal": "Produce stable match comparison reports for repeated use.",
            "exit_criteria": "Reports include quality gates, runtime, configuration manifest, and visual artifacts.",
        },
    ]


def build_productization_report(
    *,
    annotation_round: dict[str, Any] | None = None,
    tuning_report: dict[str, Any] | None = None,
    heatmap_comparison: dict[str, Any] | None = None,
    runtime_benchmarks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = readiness_items(annotation_round, heatmap_comparison, runtime_benchmarks)
    blockers = [item for item in readiness if item["status"] == "blocked"]
    needs_review = [item for item in readiness if item["status"] == "needs_review"]
    return {
        "schema_version": 1,
        "status": "blocked" if blockers else ("needs_review" if needs_review else "planned"),
        "readiness": readiness,
        "milestones": milestones(),
        "current_tuning_status": (tuning_report or {}).get("status", ""),
        "recommended_next_action": "Fill manual heatmap labels." if blockers else "Run heatmap parameter experiments.",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Heatmap Productization Plan",
        "",
        f"- status: `{report.get('status')}`",
        f"- current_tuning_status: `{report.get('current_tuning_status', '')}`",
        f"- recommended_next_action: {report.get('recommended_next_action', '')}",
        "",
        "## Readiness",
        "",
        "| area | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in report.get("readiness", []):
        lines.append(f"| {item['area']} | {item['status']} | {item['detail']} |")
    lines.extend(["", "## Milestones", "", "| id | goal | exit criteria |", "| --- | --- | --- |"])
    for item in report.get("milestones", []):
        lines.append(f"| {item['id']} | {item['goal']} | {item['exit_criteria']} |")
    lines.append("")
    return "\n".join(lines)


def source_report(path: Path | None) -> dict[str, Any] | None:
    payload = load_optional_json(path)
    if payload is None:
        return None
    return {**payload, "_source": display_path(resolve_project_path(path) or path.expanduser())}
