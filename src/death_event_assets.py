from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2

from src.data_registry import display_path


DEFAULT_FRAME_OFFSETS = (-4.0, -2.0, 0.0, 1.5, 3.0)

DEATH_ASSET_FIELDS = [
    "event_id",
    "match_id",
    "time",
    "victim",
    "victim_slot",
    "source_video",
    "asset_dir",
    "frame_times",
    "frame_paths",
    "clip_start",
    "clip_end",
    "clip_path",
    "status",
    "error",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def safe_name(value: Any) -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return safe.strip("_") or "event"


def event_time(row: Mapping[str, Any]) -> float | None:
    return float_or_none(row.get("time")) or float_or_none(row.get("elapsed_time"))


def event_window(row: Mapping[str, Any], default_before: float, default_after: float) -> tuple[float, float] | None:
    time_value = event_time(row)
    if time_value is None:
        return None
    start = float_or_none(row.get("clip_start"))
    stop = float_or_none(row.get("clip_end"))
    if start is None:
        start = max(0.0, time_value - default_before)
    if stop is None:
        stop = time_value + default_after
    if stop < start:
        start, stop = stop, start
    return max(0.0, start), max(0.0, stop)


def sample_times_for_event(
    row: Mapping[str, Any],
    offsets: Sequence[float] = DEFAULT_FRAME_OFFSETS,
    default_before: float = 8.0,
    default_after: float = 4.0,
) -> list[float]:
    time_value = event_time(row)
    window = event_window(row, default_before, default_after)
    if time_value is None or window is None:
        return []
    start, stop = window
    samples = [min(max(time_value + float(offset), start), stop) for offset in offsets]
    unique: list[float] = []
    seen: set[str] = set()
    for sample in samples:
        key = format_seconds(sample)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_frame_at(video_path: Path, time_seconds: float):
    if is_image(video_path):
        frame = cv2.imread(str(video_path))
        return frame, 0 if frame is not None else None

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None, None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return None, None
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return frame, frame_index
    finally:
        cap.release()


def write_frame(video_path: Path, time_seconds: float, output_path: Path) -> bool:
    frame, _ = read_frame_at(video_path, time_seconds)
    if frame is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def write_clip(video_path: Path, start_seconds: float, stop_seconds: float, output_path: Path) -> bool:
    if is_image(video_path):
        return False

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_seconds) * 1000.0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            elapsed = frame_index / fps
            if elapsed > stop_seconds:
                break
            writer.write(frame)
        return output_path.exists() and output_path.stat().st_size > 0
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def asset_stem(row: Mapping[str, Any], index: int) -> str:
    event_id = row.get("event_id") or f"event_{index:04d}"
    time_value = event_time(row)
    time_part = format_seconds(time_value).replace(".", "p") if time_value is not None else "unknown"
    return f"{index:04d}_{safe_name(event_id)}_{time_part}"


def export_event_assets(
    row: Mapping[str, Any],
    index: int,
    video_path: Path,
    output_dir: Path,
    frame_offsets: Sequence[float] = DEFAULT_FRAME_OFFSETS,
    default_before: float = 8.0,
    default_after: float = 4.0,
    write_clips: bool = False,
) -> dict[str, Any]:
    time_value = event_time(row)
    window = event_window(row, default_before, default_after)
    if time_value is None or window is None:
        return {
            "event_id": row.get("event_id", ""),
            "match_id": row.get("match_id", ""),
            "time": row.get("time", ""),
            "victim": row.get("victim", ""),
            "victim_slot": row.get("victim_slot", ""),
            "source_video": display_path(video_path),
            "status": "failed",
            "error": "missing event time",
        }

    stem = asset_stem(row, index)
    asset_dir = output_dir / stem
    frames_dir = asset_dir / "frames"
    frame_paths: list[str] = []
    frame_times: list[str] = []
    for sample in sample_times_for_event(row, frame_offsets, default_before, default_after):
        frame_path = frames_dir / f"frame_{format_seconds(sample).replace('.', 'p')}s.jpg"
        if write_frame(video_path, sample, frame_path):
            frame_paths.append(display_path(frame_path))
            frame_times.append(format_seconds(sample))

    clip_start, clip_end = window
    clip_path = ""
    clip_error = ""
    if write_clips:
        candidate_clip = asset_dir / "clip.mp4"
        if write_clip(video_path, clip_start, clip_end, candidate_clip):
            clip_path = display_path(candidate_clip)
        else:
            clip_error = "clip export failed"

    status = "ready" if frame_paths and (clip_path or not write_clips) else "partial" if frame_paths else "failed"
    error = clip_error if status != "ready" else ""
    if not frame_paths:
        error = "no frames exported"

    return {
        "event_id": row.get("event_id", ""),
        "match_id": row.get("match_id", ""),
        "time": format_seconds(time_value),
        "victim": row.get("victim", ""),
        "victim_slot": row.get("victim_slot", ""),
        "source_video": display_path(video_path),
        "asset_dir": display_path(asset_dir),
        "frame_times": ";".join(frame_times),
        "frame_paths": ";".join(frame_paths),
        "clip_start": format_seconds(clip_start),
        "clip_end": format_seconds(clip_end),
        "clip_path": clip_path,
        "status": status,
        "error": error,
    }


def export_death_event_assets(
    events: Sequence[Mapping[str, Any]],
    video_path: Path,
    output_dir: Path,
    frame_offsets: Sequence[float] = DEFAULT_FRAME_OFFSETS,
    default_before: float = 8.0,
    default_after: float = 4.0,
    write_clips: bool = False,
) -> dict[str, Any]:
    rows = [
        export_event_assets(
            row,
            index=index,
            video_path=video_path,
            output_dir=output_dir,
            frame_offsets=frame_offsets,
            default_before=default_before,
            default_after=default_after,
            write_clips=write_clips,
        )
        for index, row in enumerate(events, start=1)
    ]
    ready = sum(1 for row in rows if row["status"] == "ready")
    partial = sum(1 for row in rows if row["status"] == "partial")
    failed = sum(1 for row in rows if row["status"] == "failed")
    return {
        "status": "ready" if rows and failed == 0 else "empty" if not rows else "needs_review",
        "source_video": display_path(video_path),
        "output_dir": display_path(output_dir),
        "event_count": len(events),
        "asset_count": len(rows),
        "ready_count": ready,
        "partial_count": partial,
        "failed_count": failed,
        "write_clips": write_clips,
        "frame_offsets": [float(offset) for offset in frame_offsets],
        "assets": rows,
    }


def enrich_events_with_assets(
    events: Sequence[Mapping[str, Any]],
    assets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_event_id = {str(row.get("event_id", "")): row for row in assets if row.get("event_id")}
    enriched: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        asset = by_event_id.get(str(row.get("event_id", "")))
        if asset:
            if asset.get("clip_path"):
                row["clip_path"] = asset["clip_path"]
            note = str(row.get("notes", "")).strip()
            asset_note = f"asset_dir={asset.get('asset_dir', '')}"
            row["notes"] = "; ".join(part for part in (note, asset_note) if part)
        enriched.append(row)
    return enriched
