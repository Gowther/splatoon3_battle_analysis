from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.data_registry import display_path, iter_heatmap_matches, resolve_project_path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def frame_team_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("match_id", ""),
        row.get("frame_index", ""),
        row.get("time", ""),
        row.get("team", ""),
    )


def visible_annotation(row: dict[str, str]) -> bool:
    x = float_or_none(row.get("x"))
    y = float_or_none(row.get("y"))
    visibility = row.get("visibility", "visible").strip().lower()
    return x is not None and y is not None and visibility in {"visible", "uncertain"}


def prediction_indexes_by_key(registry: dict[str, Any]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    indexed: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for match, heatmap in iter_heatmap_matches(registry):
        tracks_path = resolve_project_path(heatmap.get("player_tracks"))
        if tracks_path is None or not tracks_path.exists():
            continue
        for row in read_csv_rows(tracks_path):
            row = dict(row)
            row["match_id"] = match["id"]
            indexed.setdefault(frame_team_key(row), []).append(row)
    return indexed


def distance(label: dict[str, str], prediction: dict[str, str]) -> float:
    label_x = float_or_none(label.get("x")) or 0.0
    label_y = float_or_none(label.get("y")) or 0.0
    pred_x = float_or_none(prediction.get("x")) or 0.0
    pred_y = float_or_none(prediction.get("y")) or 0.0
    return math.hypot(label_x - pred_x, label_y - pred_y)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((percent / 100.0) * (len(values) - 1))))
    return round(values[index], 4)


def evaluate_annotations(
    annotation_path: Path,
    registry: dict[str, Any],
    threshold_px: float = 80.0,
) -> dict[str, Any]:
    annotations = read_csv_rows(annotation_path)
    labels = [row for row in annotations if visible_annotation(row)]
    predictions_by_key = prediction_indexes_by_key(registry)
    used_predictions: dict[tuple[str, str, str, str], set[int]] = {}
    matches: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    complete_keys = {frame_team_key(row) for row in annotations if bool_text(row.get("frame_complete"))}

    for label in labels:
        key = frame_team_key(label)
        predictions = predictions_by_key.get(key, [])
        used = used_predictions.setdefault(key, set())
        candidates = [(index, distance(label, prediction)) for index, prediction in enumerate(predictions) if index not in used]
        if not candidates:
            misses.append({"annotation_id": label.get("annotation_id", ""), "reason": "no_prediction", "key": key})
            continue
        best_index, best_distance = min(candidates, key=lambda item: item[1])
        prediction = predictions[best_index]
        if best_distance <= threshold_px:
            used.add(best_index)
            matches.append(
                {
                    "annotation_id": label.get("annotation_id", ""),
                    "match_id": label.get("match_id", ""),
                    "time": label.get("time", ""),
                    "frame_index": label.get("frame_index", ""),
                    "team": label.get("team", ""),
                    "distance_px": round(best_distance, 4),
                    "prediction_player_id": prediction.get("player_id", ""),
                    "prediction_confidence": prediction.get("confidence", ""),
                    "prediction_track_status": prediction.get("track_status", ""),
                }
            )
        else:
            misses.append(
                {
                    "annotation_id": label.get("annotation_id", ""),
                    "reason": "nearest_prediction_too_far",
                    "distance_px": round(best_distance, 4),
                    "key": key,
                }
            )

    false_positives: list[dict[str, Any]] = []
    for key in complete_keys:
        predictions = predictions_by_key.get(key, [])
        used = used_predictions.get(key, set())
        for index, prediction in enumerate(predictions):
            if index in used:
                continue
            false_positives.append(
                {
                    "match_id": key[0],
                    "frame_index": key[1],
                    "time": key[2],
                    "team": key[3],
                    "prediction_player_id": prediction.get("player_id", ""),
                    "prediction_confidence": prediction.get("confidence", ""),
                    "prediction_track_status": prediction.get("track_status", ""),
                }
            )

    distances = [match["distance_px"] for match in matches]
    label_count = len(labels)
    matched_count = len(matches)
    false_positive_count = len(false_positives)
    complete_prediction_denominator = matched_count + false_positive_count
    return {
        "annotation_path": display_path(annotation_path),
        "threshold_px": threshold_px,
        "annotation_rows": len(annotations),
        "labeled_rows": label_count,
        "complete_frame_team_groups": len(complete_keys),
        "matched_labels": matched_count,
        "missed_labels": len(misses),
        "false_positive_predictions": false_positive_count,
        "recall": round(matched_count / label_count, 4) if label_count else None,
        "precision_on_complete_groups": round(matched_count / complete_prediction_denominator, 4)
        if complete_prediction_denominator
        else None,
        "mean_error_px": round(mean(distances), 4) if distances else None,
        "median_error_px": round(median(distances), 4) if distances else None,
        "p90_error_px": percentile(distances, 90),
        "matches": matches,
        "misses": misses,
        "false_positives": false_positives,
        "status": "no_labels" if label_count == 0 else "evaluated",
    }


def evaluate_gates(metrics: dict[str, Any], min_recall: float | None = None, max_mean_error_px: float | None = None) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    if min_recall is not None:
        actual = metrics.get("recall")
        checks["min_recall"] = {
            "expected": f">= {min_recall}",
            "actual": actual,
            "ok": actual is not None and actual >= min_recall,
        }
    if max_mean_error_px is not None:
        actual = metrics.get("mean_error_px")
        checks["max_mean_error_px"] = {
            "expected": f"<= {max_mean_error_px}",
            "actual": actual,
            "ok": actual is not None and actual <= max_mean_error_px,
        }
    return checks


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, metrics: dict[str, Any], checks: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Heatmap Annotation Evaluation",
        "",
        f"- annotation: `{metrics['annotation_path']}`",
        f"- status: {metrics['status']}",
        f"- threshold px: {metrics['threshold_px']}",
        "",
        "## Metrics",
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
        lines.append(f"| {key} | {metrics.get(key)} |")

    lines.extend(["", "## Quality Gates", "", "| check | expected | actual | status |", "| --- | --- | --- | --- |"])
    if checks:
        for key, check in checks.items():
            status = "passed" if check["ok"] else "failed"
            lines.append(f"| {key} | {check['expected']} | {check['actual']} | {status} |")
    else:
        lines.append("| none |  |  | passed |")

    lines.extend(["", "## Misses", ""])
    if metrics["misses"]:
        lines.extend(f"- {miss}" for miss in metrics["misses"][:50])
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
