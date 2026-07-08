from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path
from src.data_registry import display_path, load_registry, resolve_project_path
from src.heatmap.anomaly_export import export_anomalies
from src.model_error_report import COUNT_FIELDS, PLAYER_STATE_FIELDS, WEAPON_FIELDS, int_or_none, read_csv


CANDIDATE_FIELDS = [
    "candidate_id",
    "target",
    "reason",
    "source_id",
    "match_id",
    "video",
    "elapsed_time",
    "row_index",
    "frame_path",
    "details",
]


def load_json(path: Path) -> Any:
    with project_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = CANDIDATE_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def analysis_windows_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for match in registry.get("matches", []):
        for window in match.get("analysis_windows", []):
            windows[str(window.get("id", ""))] = {
                "match_id": match.get("id", ""),
                "video": match.get("video", ""),
                "window": window,
            }
    return windows


def is_complete_player_state(row: dict[str, str]) -> bool:
    return all(row.get(field) for field in PLAYER_STATE_FIELDS)


def weapon_tuple(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in WEAPON_FIELDS)


def count_jump_events(rows: list[dict[str, str]], threshold: int = 20) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: dict[str, tuple[int, str, int]] = {}
    for index, row in enumerate(rows, start=1):
        elapsed = row.get("elapsed_time", "")
        for field in COUNT_FIELDS:
            value = int_or_none(row.get(field))
            if value is None:
                continue
            if field in previous:
                prev_index, prev_elapsed, prev_value = previous[field]
                if abs(value - prev_value) > threshold:
                    events.append(
                        {
                            "field": field,
                            "row_index": index,
                            "elapsed_time": elapsed,
                            "value": value,
                            "previous_row_index": prev_index,
                            "previous_elapsed_time": prev_elapsed,
                            "previous_value": prev_value,
                        }
                    )
            previous[field] = (index, elapsed, value)
    return events


def safe_seconds(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_name(source_id: str, elapsed_time: str, reason: str, suffix: str = ".jpg") -> str:
    safe_time = str(elapsed_time).replace(".", "_")
    safe_reason = reason.replace(":", "_").replace("/", "_")
    return f"{source_id}_{safe_time}_{safe_reason}{suffix}"


def export_video_frame(video: str, elapsed_time: str, output_dir: Path, source_id: str, reason: str) -> str:
    seconds = safe_seconds(elapsed_time)
    source = resolve_project_path(video)
    if seconds is None or source is None or not source.exists():
        return ""
    try:
        import cv2
    except ImportError:
        return ""

    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return ""
    finally:
        cap.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / frame_name(source_id, elapsed_time, reason)
    if not cv2.imwrite(str(target), frame):
        return ""
    return display_path(target)


def candidate_row(
    *,
    target: str,
    reason: str,
    source_id: str,
    match_id: str,
    video: str,
    elapsed_time: str,
    row_index: int,
    frame_path: str,
    details: str,
    index: int,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{target}:{source_id}:{index:04d}",
        "target": target,
        "reason": reason,
        "source_id": source_id,
        "match_id": match_id,
        "video": video,
        "elapsed_time": elapsed_time,
        "row_index": row_index,
        "frame_path": frame_path,
        "details": details,
    }


def collect_analysis_candidates(
    evaluation_results: list[dict[str, Any]],
    registry: dict[str, Any],
    output_dir: Path,
    *,
    max_rows_per_reason: int = 24,
    export_frames: bool = True,
) -> dict[str, Any]:
    windows = analysis_windows_by_id(registry)
    frame_dir = output_dir / "frames"
    rows_by_target: dict[str, list[dict[str, Any]]] = {
        "ui_detector_yolo": [],
        "count_ocr_yolo": [],
        "message_ocr_yolo": [],
        "weapon_classifier_resnet18": [],
    }
    summaries: list[dict[str, Any]] = []

    for result in evaluation_results:
        if result.get("kind") != "analysis":
            continue
        source_id = str(result.get("id", ""))
        window = windows.get(source_id, {})
        match_id = str(window.get("match_id", ""))
        video = str(window.get("video", ""))
        raw_csv = result.get("raw_csv")
        smoothed_csv = result.get("smoothed_csv") or raw_csv
        if not raw_csv:
            continue

        _, raw_rows = read_csv(Path(str(raw_csv)))
        _, smoothed_rows = read_csv(Path(str(smoothed_csv))) if smoothed_csv else (None, raw_rows)
        source_counts = {target: 0 for target in rows_by_target}

        missing_state = [
            (index, row)
            for index, row in enumerate(smoothed_rows, start=1)
            if not is_complete_player_state(row)
        ][:max_rows_per_reason]
        for index, row in missing_state:
            elapsed = row.get("elapsed_time", "")
            reason = "missing_player_state"
            frame = export_video_frame(video, elapsed, frame_dir, source_id, reason) if export_frames else ""
            rows_by_target["ui_detector_yolo"].append(
                candidate_row(
                    target="ui_detector_yolo",
                    reason=reason,
                    source_id=source_id,
                    match_id=match_id,
                    video=video,
                    elapsed_time=elapsed,
                    row_index=index,
                    frame_path=frame,
                    details="one or more player_state columns are empty",
                    index=len(rows_by_target["ui_detector_yolo"]) + 1,
                )
            )
            source_counts["ui_detector_yolo"] += 1

        for event in count_jump_events(raw_rows)[:max_rows_per_reason]:
            elapsed = str(event.get("elapsed_time", ""))
            reason = f"count_jump:{event['field']}"
            frame = export_video_frame(video, elapsed, frame_dir, source_id, reason) if export_frames else ""
            rows_by_target["count_ocr_yolo"].append(
                candidate_row(
                    target="count_ocr_yolo",
                    reason=reason,
                    source_id=source_id,
                    match_id=match_id,
                    video=video,
                    elapsed_time=elapsed,
                    row_index=int(event["row_index"]),
                    frame_path=frame,
                    details=(
                        f"{event['field']} {event['previous_value']}->{event['value']} "
                        f"from {event['previous_elapsed_time']}s to {elapsed}s"
                    ),
                    index=len(rows_by_target["count_ocr_yolo"]) + 1,
                )
            )
            source_counts["count_ocr_yolo"] += 1

        first_weapon = next((i for i, row in enumerate(smoothed_rows, start=1) if row.get("weapon_1")), None)
        if first_weapon is not None:
            first_weapons = weapon_tuple(smoothed_rows[first_weapon - 1])
            weapon_candidates = [
                (index, row, "missing_weapon_after_warmup")
                for index, row in enumerate(smoothed_rows[first_weapon - 1 :], start=first_weapon)
                if not row.get("weapon_1")
            ]
            weapon_candidates.extend(
                (index, row, "weapon_set_changed")
                for index, row in enumerate(smoothed_rows[first_weapon - 1 :], start=first_weapon)
                if row.get("weapon_1") and weapon_tuple(row) != first_weapons
            )
            for index, row, reason in weapon_candidates[:max_rows_per_reason]:
                elapsed = row.get("elapsed_time", "")
                frame = export_video_frame(video, elapsed, frame_dir, source_id, reason) if export_frames else ""
                rows_by_target["weapon_classifier_resnet18"].append(
                    candidate_row(
                        target="weapon_classifier_resnet18",
                        reason=reason,
                        source_id=source_id,
                        match_id=match_id,
                        video=video,
                        elapsed_time=elapsed,
                        row_index=index,
                        frame_path=frame,
                        details="review warmup/final weapon classification crop quality",
                        index=len(rows_by_target["weapon_classifier_resnet18"]) + 1,
                    )
                )
                source_counts["weapon_classifier_resnet18"] += 1

        message_rows = [
            (index, row)
            for index, row in enumerate(smoothed_rows, start=1)
            if row.get("message")
        ][:max_rows_per_reason]
        for index, row in message_rows:
            elapsed = row.get("elapsed_time", "")
            reason = "message_ocr_review"
            frame = export_video_frame(video, elapsed, frame_dir, source_id, reason) if export_frames else ""
            rows_by_target["message_ocr_yolo"].append(
                candidate_row(
                    target="message_ocr_yolo",
                    reason=reason,
                    source_id=source_id,
                    match_id=match_id,
                    video=video,
                    elapsed_time=elapsed,
                    row_index=index,
                    frame_path=frame,
                    details=f"message={row.get('message', '')}",
                    index=len(rows_by_target["message_ocr_yolo"]) + 1,
                )
            )
            source_counts["message_ocr_yolo"] += 1

        summaries.append(
            {
                "source_id": source_id,
                "match_id": match_id,
                "video": video,
                "raw_csv": raw_csv,
                "smoothed_csv": smoothed_csv,
                "candidates": source_counts,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_reports: dict[str, dict[str, Any]] = {}
    for target, rows in rows_by_target.items():
        csv_path = output_dir / f"{target}_candidates.csv"
        write_csv(csv_path, rows)
        target_reports[target] = {"csv": display_path(csv_path), "rows": len(rows)}

    return {
        "status": "ready" if any(report["rows"] for report in target_reports.values()) else "empty",
        "output_dir": display_path(output_dir),
        "targets": target_reports,
        "sources": summaries,
        "frame_dir": display_path(frame_dir),
    }


def target_training_status(model_training_plan: dict[str, Any] | None) -> dict[str, Any]:
    if not model_training_plan:
        return {}
    return {
        target.get("id", ""): {
            "status": target.get("status", ""),
            "dataset_status": target.get("dataset_status", ""),
            "candidate_output_dir": target.get("candidate_output_dir", ""),
            "candidate_command": target.get("candidate_command", ""),
            "missing_paths": target.get("missing_paths", []),
        }
        for target in model_training_plan.get("targets", [])
    }


def build_training_sample_package(
    *,
    registry_path: Path,
    evaluation_results_path: Path,
    output_dir: Path,
    model_training_plan_path: Path | None = None,
    include_heatmap: bool = True,
    heatmap_match_ids: list[str] | None = None,
    max_rows_per_reason: int = 24,
    max_heatmap_items_per_match: int = 24,
    export_frames: bool = True,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    evaluation_results = load_json(evaluation_results_path)
    if not isinstance(evaluation_results, list):
        raise ValueError("evaluation_results must be a list")
    model_training_plan = (
        load_json(model_training_plan_path)
        if model_training_plan_path and project_path(model_training_plan_path).exists()
        else None
    )

    output_dir = project_path(output_dir)
    analysis_report = collect_analysis_candidates(
        evaluation_results,
        registry,
        output_dir / "analysis",
        max_rows_per_reason=max_rows_per_reason,
        export_frames=export_frames,
    )
    heatmap_report = {}
    if include_heatmap:
        heatmap_report = export_anomalies(
            registry,
            output_dir / "heatmap",
            match_ids=heatmap_match_ids,
            max_items_per_match=max_heatmap_items_per_match,
        )

    training_status = target_training_status(model_training_plan)
    target_rows = {
        target: report.get("rows", 0)
        for target, report in analysis_report.get("targets", {}).items()
    }
    if heatmap_report:
        target_rows["heatmap_tracker_labels"] = heatmap_report.get("total_exported", 0)

    report = {
        "schema_version": 1,
        "status": "ready" if any(target_rows.values()) else "empty",
        "output_dir": display_path(output_dir),
        "evaluation_results": display_path(project_path(evaluation_results_path)),
        "analysis": analysis_report,
        "heatmap": heatmap_report,
        "target_rows": target_rows,
        "training_status": training_status,
        "next_steps": next_steps(target_rows, training_status),
    }
    write_json(output_dir / "manifest.json", report)
    return report


def next_steps(target_rows: dict[str, int], training_status: dict[str, Any]) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    if target_rows.get("ui_detector_yolo", 0):
        steps.append(
            {
                "id": "label_ui_detector_candidates",
                "reason": "frames with missing player state can become YOLO bbox labels for UI/player detection",
                "command": "review outputs/training_sample_candidates/analysis/ui_detector_yolo_candidates.csv",
            }
        )
        steps.append(
            {
                "id": "train_ui_detector_after_labels",
                "reason": "ui_detector_yolo dataset is the active detector target",
                "command": "python scripts/run_model_training_target.py --target ui_detector_yolo",
            }
        )
    for target in ("count_ocr_yolo", "message_ocr_yolo"):
        if target_rows.get(target, 0):
            status = training_status.get(target, {})
            steps.append(
                {
                    "id": f"build_{target}_dataset",
                    "reason": f"{target} currently needs labeled crops before training (status={status.get('status', 'unknown')})",
                    "command": f"review outputs/training_sample_candidates/analysis/{target}_candidates.csv",
                }
            )
    if target_rows.get("weapon_classifier_resnet18", 0):
        steps.append(
            {
                "id": "review_weapon_candidates",
                "reason": "weapon candidates need icon crops or corrected class labels before classifier retraining",
                "command": "review outputs/training_sample_candidates/analysis/weapon_classifier_resnet18_candidates.csv",
            }
        )
    if target_rows.get("heatmap_tracker_labels", 0):
        steps.append(
            {
                "id": "label_heatmap_anomalies",
                "reason": "jump resets and track gaps should become manual x/y labels before parameter tuning",
                "command": "python scripts/start_heatmap_labeling_round.py",
            }
        )
    return steps


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Training Sample Candidates",
        "",
        f"- status: `{report.get('status')}`",
        f"- output_dir: `{report.get('output_dir', '')}`",
        f"- evaluation_results: `{report.get('evaluation_results', '')}`",
        "",
        "## Candidate Counts",
        "",
        "| target | rows | dataset status |",
        "| --- | ---: | --- |",
    ]
    training_status = report.get("training_status", {})
    for target, rows in sorted(report.get("target_rows", {}).items()):
        status = training_status.get(target, {}).get("status", "manual_labeling")
        lines.append(f"| `{target}` | {rows} | `{status}` |")

    lines.extend(["", "## Analysis Queues", "", "| target | csv | rows |", "| --- | --- | ---: |"])
    for target, item in sorted(report.get("analysis", {}).get("targets", {}).items()):
        lines.append(f"| `{target}` | `{item.get('csv', '')}` | {item.get('rows', 0)} |")

    heatmap = report.get("heatmap", {})
    if heatmap:
        lines.extend(
            [
                "",
                "## Heatmap Anomalies",
                "",
                f"- total_exported: {heatmap.get('total_exported', 0)}",
                f"- anomalies_csv: `{heatmap.get('anomalies_csv', '')}`",
            ]
        )
        for match in heatmap.get("matches", []):
            lines.append(f"- `{match.get('match_id')}`: {match.get('exported', 0)} {match.get('by_type', {})}")

    lines.extend(["", "## Next Steps", ""])
    for step in report.get("next_steps", []):
        lines.append(f"- `{step.get('id', '')}`: {step.get('reason', '')}")
        lines.append(f"  - `{step.get('command', '')}`")
    if not report.get("next_steps"):
        lines.append("- No candidates were exported.")
    lines.append("")
    return "\n".join(lines)
