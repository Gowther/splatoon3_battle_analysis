from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.data_registry import display_path, iter_heatmap_matches, resolve_project_path


ANOMALY_FIELDS = [
    "match_id",
    "heatmap_id",
    "anomaly_type",
    "severity",
    "time",
    "frame_index",
    "team",
    "track_slot",
    "player_id",
    "x",
    "y",
    "confidence",
    "track_status",
    "step_distance",
    "frame_path",
    "exported_frame",
    "preview_path",
    "note",
]


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANOMALY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def float_or_zero(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def candidate_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        row.get("anomaly_type", ""),
        row.get("time", ""),
        row.get("team", ""),
        row.get("track_slot", ""),
        row.get("player_id", ""),
    )


def collect_candidates(
    match_id: str,
    heatmap_id: str,
    tracks: list[dict[str, str]],
    gaps: list[dict[str, str]],
    low_confidence: float,
    large_step_px: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in tracks:
        confidence = float_or_zero(row.get("confidence"))
        step_distance = float_or_zero(row.get("step_distance"))
        if row.get("track_status") == "jump_reset":
            candidates.append(build_candidate(match_id, heatmap_id, "jump_reset", step_distance, row, "track_status=jump_reset"))
        if step_distance >= large_step_px:
            candidates.append(build_candidate(match_id, heatmap_id, "large_step", step_distance, row, f"step_distance>={large_step_px}"))
        if confidence and confidence <= low_confidence:
            candidates.append(
                build_candidate(
                    match_id,
                    heatmap_id,
                    "low_confidence",
                    low_confidence - confidence,
                    row,
                    f"confidence<={low_confidence}",
                )
            )

    for row in gaps:
        candidates.append(build_candidate(match_id, heatmap_id, "track_gap", float_or_zero(row.get("step_distance")), row, row.get("note", "")))

    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        if key not in unique or candidate["severity"] > unique[key]["severity"]:
            unique[key] = candidate
    return sorted(unique.values(), key=lambda row: (-float_or_zero(row["severity"]), row.get("time", "")))


def build_candidate(
    match_id: str,
    heatmap_id: str,
    anomaly_type: str,
    severity: float,
    row: dict[str, str],
    note: str,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "heatmap_id": heatmap_id,
        "anomaly_type": anomaly_type,
        "severity": round(severity, 4),
        "time": row.get("time", ""),
        "frame_index": row.get("frame_index", ""),
        "team": row.get("team", ""),
        "track_slot": row.get("track_slot", ""),
        "player_id": row.get("player_id", ""),
        "x": row.get("x", ""),
        "y": row.get("y", ""),
        "confidence": row.get("confidence", ""),
        "track_status": row.get("track_status", ""),
        "step_distance": row.get("step_distance", ""),
        "frame_path": row.get("frame_path", ""),
        "exported_frame": "",
        "preview_path": "",
        "note": note,
    }


def draw_preview(source: Path, preview: Path, candidate: dict[str, Any]) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    image = cv2.imread(str(source))
    if image is None:
        return False
    x = int(round(float_or_zero(candidate.get("x"))))
    y = int(round(float_or_zero(candidate.get("y"))))
    if x or y:
        cv2.circle(image, (x, y), 16, (0, 0, 255), 3)
        cv2.putText(
            image,
            str(candidate.get("anomaly_type", "")),
            (x + 18, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    preview.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(preview), image))


def export_candidate_assets(output_dir: Path, candidate: dict[str, Any], index: int) -> dict[str, Any]:
    source = resolve_project_path(candidate.get("frame_path"))
    if source is None or not source.exists():
        return candidate

    safe_time = str(candidate.get("time", "")).replace(".", "_")
    stem = f"{index:04d}_{candidate['match_id']}_{safe_time}_{candidate['anomaly_type']}_{candidate.get('team', '')}_{candidate.get('track_slot', '')}"
    frame_dest = output_dir / "frames" / f"{stem}{source.suffix}"
    frame_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, frame_dest)
    candidate["exported_frame"] = display_path(frame_dest)

    preview_dest = output_dir / "previews" / f"{stem}.jpg"
    if draw_preview(source, preview_dest, candidate):
        candidate["preview_path"] = display_path(preview_dest)
    return candidate


def export_anomalies(
    registry: dict[str, Any],
    output_dir: Path,
    match_ids: list[str] | None = None,
    low_confidence: float = 0.56,
    large_step_px: float = 420.0,
    max_items_per_match: int = 24,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(match_ids or [])
    all_rows: list[dict[str, Any]] = []
    match_summaries: list[dict[str, Any]] = []

    for match, heatmap in iter_heatmap_matches(registry):
        if selected and match["id"] not in selected:
            continue
        tracks = read_csv_rows(resolve_project_path(heatmap.get("player_tracks")))
        gaps = read_csv_rows(resolve_project_path(heatmap.get("player_track_gaps")))
        candidates = collect_candidates(
            match["id"],
            heatmap.get("id", ""),
            tracks,
            gaps,
            low_confidence=low_confidence,
            large_step_px=large_step_px,
        )[:max_items_per_match]
        exported = [export_candidate_assets(output_dir, candidate, len(all_rows) + index + 1) for index, candidate in enumerate(candidates)]
        all_rows.extend(exported)
        by_type: dict[str, int] = {}
        for row in exported:
            by_type[row["anomaly_type"]] = by_type.get(row["anomaly_type"], 0) + 1
        match_summaries.append(
            {
                "match_id": match["id"],
                "heatmap_id": heatmap.get("id", ""),
                "exported": len(exported),
                "by_type": by_type,
            }
        )

    write_csv(output_dir / "anomalies.csv", all_rows)
    summary = {
        "output_dir": display_path(output_dir),
        "low_confidence": low_confidence,
        "large_step_px": large_step_px,
        "max_items_per_match": max_items_per_match,
        "total_exported": len(all_rows),
        "matches": match_summaries,
        "anomalies_csv": display_path(output_dir / "anomalies.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
