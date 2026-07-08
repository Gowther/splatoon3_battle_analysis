from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PRIORITY_QUEUE_FIELDS = [
    "annotation_id",
    "match_id",
    "time",
    "team",
    "track_status",
    "confidence",
    "frame_path",
    "preview_path",
]


def priority_queue_rows(tasks: list[dict[str, Any]], *, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in tasks[:limit]:
        rows.append({field: str(task.get(field, "")) for field in PRIORITY_QUEUE_FIELDS})
    return rows


def write_priority_queue(path: Path, tasks: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    rows = priority_queue_rows(tasks, limit=limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRIORITY_QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "rows": len(rows), "limit": limit}


def workbench_status(round_report: dict[str, Any], priority_rows: int) -> str:
    progress = round_report.get("progress", {})
    if progress.get("status") == "missing_template":
        return "missing_template"
    if round_report.get("label_readiness", {}).get("status") == "ready":
        return "ready_for_tuning"
    if priority_rows:
        return "ready_to_label"
    if int(progress.get("labeled_rows") or 0) > 0:
        return "needs_more_labels"
    return "needs_labels"


def build_labeling_workbench_report(
    round_report: dict[str, Any],
    *,
    priority_queue: dict[str, Any] | None = None,
    annotation_ui: dict[str, Any] | None = None,
) -> dict[str, Any]:
    priority_queue = priority_queue or {"path": "", "rows": 0, "limit": 0}
    status = workbench_status(round_report, int(priority_queue.get("rows") or 0))
    annotation_csv = str(round_report.get("annotation_csv", ""))
    return {
        "schema_version": 1,
        "status": status,
        "round_status": round_report.get("status", ""),
        "blocking_reason": round_report.get("blocking_reason", ""),
        "round": round_report.get("round", {}),
        "annotation_csv": annotation_csv,
        "package_dir": round_report.get("package_dir", ""),
        "progress": round_report.get("progress", {}),
        "label_readiness": round_report.get("label_readiness", {}),
        "priority_queue": priority_queue,
        "annotation_ui": annotation_ui or {},
        "next_commands": {
            "edit_csv": annotation_csv,
            "evaluate": f"python scripts/evaluate_heatmap_annotations.py {annotation_csv}",
            "tune": f"python scripts/run_heatmap_parameter_experiments.py --annotation-csv {annotation_csv} --write-configs",
        },
        "round_report": round_report,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    progress = report.get("progress", {})
    priority_queue = report.get("priority_queue", {})
    annotation_ui = report.get("annotation_ui", {})
    commands = report.get("next_commands", {})
    lines = [
        "# Heatmap Labeling Workbench",
        "",
        f"- status: `{report.get('status')}`",
        f"- round_status: `{report.get('round_status')}`",
        f"- annotation_csv: `{report.get('annotation_csv', '')}`",
        f"- priority_queue: `{priority_queue.get('path', '')}`",
        f"- priority_rows: {priority_queue.get('rows', 0)}",
        f"- annotation_ui: `{annotation_ui.get('output_html', '')}`",
        f"- blocking_reason: `{report.get('blocking_reason', '')}`",
        "",
        "## Progress",
        "",
        f"- labeled_rows: {progress.get('labeled_rows', 0)}",
        f"- task_rows: {progress.get('task_rows', 0)}",
        f"- complete_frame_team_groups: {progress.get('complete_frame_team_groups', 0)}",
        "",
        "## Commands",
        "",
        f"- edit_csv: `{commands.get('edit_csv', '')}`",
        f"- evaluate: `{commands.get('evaluate', '')}`",
        f"- tune: `{commands.get('tune', '')}`",
        "",
    ]
    return "\n".join(lines)
