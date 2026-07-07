from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.data_registry import display_path, load_registry, resolve_project_path
from src.heatmap.quality_loop import build_quality_loop_report


def load_round_config(path: Path) -> dict[str, Any]:
    target = resolve_project_path(path) or path.expanduser()
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_round(config: dict[str, Any], round_id: str) -> dict[str, Any]:
    rounds = config.get("rounds", {})
    if round_id in rounds:
        return dict(rounds[round_id])
    if round_id == "all":
        return {
            "id": "all",
            "description": "All configured heatmap matches.",
            "matches": list(config.get("heatmap_matches", [])),
            "frames_per_match": config.get("defaults", {}).get("frames_per_match", 5),
        }
    raise KeyError(f"Unknown annotation round: {round_id}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def has_manual_position(row: dict[str, str]) -> bool:
    return bool(row.get("x", "").strip() and row.get("y", "").strip())


def is_visible_task(row: dict[str, str]) -> bool:
    visibility = row.get("visibility", "visible").strip().lower()
    return visibility in {"visible", "uncertain", ""}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def annotation_progress(annotation_csv: Path | None) -> dict[str, Any]:
    if annotation_csv is None:
        return {"status": "missing_template", "annotation_csv": "", "total_rows": 0}
    path = annotation_csv.expanduser()
    if not path.exists():
        return {"status": "missing_template", "annotation_csv": display_path(path), "total_rows": 0}

    rows = read_csv_rows(path)
    task_rows = [row for row in rows if is_visible_task(row)]
    labeled_rows = [row for row in task_rows if has_manual_position(row)]
    skipped_rows = [row for row in rows if not is_visible_task(row)]
    complete_groups = {
        (row.get("match_id", ""), row.get("time", ""), row.get("frame_index", ""), row.get("team", ""))
        for row in rows
        if truthy(row.get("frame_complete"))
    }
    matches: dict[str, dict[str, int]] = {}
    for row in rows:
        match_id = row.get("match_id", "")
        item = matches.setdefault(match_id, {"rows": 0, "labeled": 0, "skipped": 0, "complete_groups": 0})
        item["rows"] += 1
        if row in labeled_rows:
            item["labeled"] += 1
        if row in skipped_rows:
            item["skipped"] += 1
    for match_id, _, _, _ in complete_groups:
        matches.setdefault(match_id, {"rows": 0, "labeled": 0, "skipped": 0, "complete_groups": 0})["complete_groups"] += 1

    status = "ready_for_evaluation" if labeled_rows else "needs_labels"
    completion_ratio = round(len(labeled_rows) / len(task_rows), 4) if task_rows else None
    return {
        "status": status,
        "annotation_csv": display_path(path),
        "total_rows": len(rows),
        "task_rows": len(task_rows),
        "labeled_rows": len(labeled_rows),
        "skipped_rows": len(skipped_rows),
        "complete_frame_team_groups": len(complete_groups),
        "completion_ratio": completion_ratio,
        "matches": matches,
    }


def priority_score(row: dict[str, str]) -> tuple[int, float]:
    status = row.get("source_track_status", "").strip().lower()
    confidence_text = row.get("source_confidence", "").strip()
    try:
        confidence = float(confidence_text)
    except ValueError:
        confidence = 1.0
    status_rank = 0 if status == "jump_reset" else (1 if status in {"new", "gap"} else 2)
    return status_rank, confidence


def annotation_priority_tasks(annotation_csv: Path | None, *, limit: int = 24) -> list[dict[str, Any]]:
    if annotation_csv is None:
        return []
    path = annotation_csv.expanduser()
    if not path.exists():
        return []
    rows = [
        row
        for row in read_csv_rows(path)
        if is_visible_task(row) and not has_manual_position(row)
    ]
    rows.sort(key=priority_score)
    tasks: list[dict[str, Any]] = []
    seen_groups: set[tuple[str, str, str]] = set()
    for row in rows:
        group = (row.get("match_id", ""), row.get("time", ""), row.get("team", ""))
        if group in seen_groups:
            continue
        seen_groups.add(group)
        tasks.append(
            {
                "match_id": row.get("match_id", ""),
                "time": row.get("time", ""),
                "team": row.get("team", ""),
                "annotation_id": row.get("annotation_id", ""),
                "track_status": row.get("source_track_status", ""),
                "confidence": row.get("source_confidence", ""),
                "frame_path": row.get("frame_path", ""),
                "preview_path": row.get("preview_path", ""),
            }
        )
        if len(tasks) >= limit:
            break
    return tasks


def evaluate_progress_gates(
    progress: dict[str, Any],
    *,
    min_labeled_rows: int | None = None,
    min_complete_groups: int | None = None,
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    if min_labeled_rows is not None:
        actual = int(progress.get("labeled_rows") or 0)
        checks["min_labeled_rows"] = {
            "expected": f">= {min_labeled_rows}",
            "actual": actual,
            "ok": actual >= min_labeled_rows,
        }
    if min_complete_groups is not None:
        actual = int(progress.get("complete_frame_team_groups") or 0)
        checks["min_complete_groups"] = {
            "expected": f">= {min_complete_groups}",
            "actual": actual,
            "ok": actual >= min_complete_groups,
        }
    return checks


def build_annotation_round_report(
    *,
    registry_path: Path,
    config_path: Path,
    round_id: str,
    package_dir: Path,
    match_ids: list[str] | None = None,
    frames_per_match: int | None = None,
    export_package: bool = True,
    annotation_csv: Path | None = None,
    threshold_px: float | None = None,
    min_labeled_rows: int | None = None,
    min_complete_groups: int | None = None,
) -> dict[str, Any]:
    config = load_round_config(config_path)
    selected_round = resolve_round(config, round_id)
    defaults = config.get("defaults", {})
    selected_matches = match_ids or list(selected_round.get("matches", []))
    selected_frames = frames_per_match or int(selected_round.get("frames_per_match", defaults.get("frames_per_match", 5)))
    threshold = threshold_px or float(defaults.get("annotation_distance_threshold_px", 80.0))
    registry = load_registry(registry_path)
    template = annotation_csv or package_dir / "annotation_template.csv"
    quality_report = build_quality_loop_report(
        registry,
        package_dir=package_dir,
        annotation_csv=annotation_csv,
        frames_per_match=selected_frames,
        match_ids=selected_matches,
        export_package=export_package,
        threshold_px=threshold,
    )
    progress = annotation_progress(template)
    priority_tasks = annotation_priority_tasks(template)
    progress_checks = evaluate_progress_gates(
        progress,
        min_labeled_rows=min_labeled_rows,
        min_complete_groups=min_complete_groups,
    )
    status = "ready_for_evaluation" if progress.get("labeled_rows", 0) else "needs_labels"
    if quality_report.get("status") == "passed":
        status = "passed"
    elif quality_report.get("status") == "needs_review":
        status = "needs_review"
    if any(not check["ok"] for check in progress_checks.values()):
        status = "needs_labels"
    return {
        "schema_version": 1,
        "status": status,
        "round": {
            "id": round_id,
            "description": selected_round.get("description", ""),
            "matches": selected_matches,
            "frames_per_match": selected_frames,
            "threshold_px": threshold,
        },
        "package_dir": display_path(package_dir),
        "annotation_csv": display_path(template),
        "progress": progress,
        "priority_tasks": priority_tasks,
        "progress_checks": progress_checks,
        "quality_loop": quality_report,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    progress = report.get("progress", {})
    round_info = report.get("round", {})
    lines = [
        "# Heatmap Annotation Round",
        "",
        f"- status: `{report.get('status')}`",
        f"- round: `{round_info.get('id', '')}`",
        f"- package_dir: `{report.get('package_dir', '')}`",
        f"- annotation_csv: `{report.get('annotation_csv', '')}`",
        f"- matches: {', '.join(round_info.get('matches', []))}",
        f"- frames_per_match: {round_info.get('frames_per_match', '')}",
        "",
        "## Progress",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "total_rows",
        "task_rows",
        "labeled_rows",
        "skipped_rows",
        "complete_frame_team_groups",
        "completion_ratio",
    ):
        lines.append(f"| {key} | {progress.get(key, '')} |")

    lines.extend(["", "## Progress Gates", "", "| check | expected | actual | status |", "| --- | --- | --- | --- |"])
    checks = report.get("progress_checks", {})
    if checks:
        for key, check in checks.items():
            status = "passed" if check["ok"] else "failed"
            lines.append(f"| {key} | {check['expected']} | {check['actual']} | {status} |")
    else:
        lines.append("| none |  |  | passed |")

    lines.extend(["", "## Matches", "", "| match | rows | labeled | skipped | complete groups |", "| --- | ---: | ---: | ---: | ---: |"])
    for match_id, item in sorted(progress.get("matches", {}).items()):
        lines.append(
            f"| {match_id} | {item.get('rows', 0)} | {item.get('labeled', 0)} | "
            f"{item.get('skipped', 0)} | {item.get('complete_groups', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Priority Tasks",
            "",
            "| match | time | team | status | confidence | annotation |",
            "| --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    priority_tasks = report.get("priority_tasks", [])
    if priority_tasks:
        for task in priority_tasks:
            lines.append(
                f"| {task.get('match_id', '')} | {task.get('time', '')} | {task.get('team', '')} | "
                f"{task.get('track_status', '')} | {task.get('confidence', '')} | `{task.get('annotation_id', '')}` |"
            )
    else:
        lines.append("| none |  |  |  |  |  |")

    quality = report.get("quality_loop", {})
    lines.extend(
        [
            "",
            "## Quality Loop",
            "",
            f"- status: `{quality.get('status', '')}`",
            f"- matched_labels: {quality.get('metrics', {}).get('matched_labels', '')}",
            f"- missed_labels: {quality.get('metrics', {}).get('missed_labels', '')}",
            f"- mean_error_px: {quality.get('metrics', {}).get('mean_error_px', '')}",
            "",
        ]
    )
    return "\n".join(lines)
