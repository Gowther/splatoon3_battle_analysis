from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2

from src.core.paths import project_path
from src.data_registry import display_path


TARGET = "death_event_ocr"

DEATH_OCR_CANDIDATE_FIELDS = [
    "candidate_id",
    "target",
    "reason",
    "source_id",
    "match_id",
    "video",
    "event_id",
    "elapsed_time",
    "row_index",
    "victim",
    "victim_slot",
    "frame_time",
    "frame_path",
    "source_frame_path",
    "crop_path",
    "region",
    "roi",
    "x1",
    "y1",
    "x2",
    "y2",
    "ocr_text",
    "ocr_confidence",
    "details",
]


@dataclass(frozen=True)
class OcrRegion:
    name: str
    roi: tuple[float, float, float, float]
    reason: str


DEFAULT_OCR_REGIONS = (
    OcrRegion("kill_log_right", (0.62, 0.04, 0.98, 0.34), "review_kill_log"),
    OcrRegion("death_message_center", (0.18, 0.40, 0.82, 0.72), "review_death_message"),
    OcrRegion("full_death_screen", (0.0, 0.0, 1.0, 1.0), "review_death_screen"),
)


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
    return safe.strip("_") or "item"


def split_semicolon(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def parse_frame_entries(asset: Mapping[str, Any]) -> list[dict[str, Any]]:
    times = split_semicolon(asset.get("frame_times"))
    paths = split_semicolon(asset.get("frame_paths"))
    entries: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        time_value = float_or_none(times[index]) if index < len(times) else None
        entries.append({"frame_time": time_value, "frame_path": path})
    return entries


def selected_frame_entries(asset: Mapping[str, Any], max_frames_per_event: int) -> list[dict[str, Any]]:
    entries = parse_frame_entries(asset)
    if max_frames_per_event <= 0:
        return entries
    event_time = float_or_none(asset.get("time"))
    if event_time is None:
        return entries[:max_frames_per_event]

    def rank(entry: Mapping[str, Any]) -> tuple[int, float, str]:
        frame_time = entry.get("frame_time")
        if frame_time is None:
            return (2, 999999.0, str(entry.get("frame_path", "")))
        after_penalty = 0 if float(frame_time) >= event_time else 1
        preferred_delta = abs(float(frame_time) - (event_time + 1.5))
        return (after_penalty, preferred_delta, str(entry.get("frame_path", "")))

    return sorted(entries, key=rank)[:max_frames_per_event]


def pixel_box(width: int, height: int, roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x1 = int(round(max(0.0, min(1.0, roi[0])) * width))
    y1 = int(round(max(0.0, min(1.0, roi[1])) * height))
    x2 = int(round(max(0.0, min(1.0, roi[2])) * width))
    y2 = int(round(max(0.0, min(1.0, roi[3])) * height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_region(source_frame: Path, output_path: Path, region: OcrRegion) -> tuple[bool, tuple[int, int, int, int] | None]:
    image = cv2.imread(str(source_frame))
    if image is None:
        return False, None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = pixel_box(width, height, region.roi)
    crop = image[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), crop)), (x1, y1, x2, y2)


def roi_text(roi: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.4f}" for value in roi)


def build_candidate_id(asset: Mapping[str, Any], frame_index: int, region: OcrRegion, index: int) -> str:
    event_id = safe_name(asset.get("event_id") or f"event_{index}")
    return f"{TARGET}:{event_id}:f{frame_index:02d}:{safe_name(region.name)}"


def build_death_ocr_candidates(
    assets: Sequence[Mapping[str, Any]],
    output_dir: Path,
    regions: Sequence[OcrRegion] = DEFAULT_OCR_REGIONS,
    max_frames_per_event: int = 2,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for event_index, asset in enumerate(assets, start=1):
        entries = selected_frame_entries(asset, max_frames_per_event)
        for frame_index, entry in enumerate(entries, start=1):
            source_frame = project_path(str(entry.get("frame_path", "")))
            frame_time = entry.get("frame_time")
            for region in regions:
                candidate_id = build_candidate_id(asset, frame_index, region, len(rows) + 1)
                crop_path = output_dir / "crops" / region.name / f"{safe_name(candidate_id)}.jpg"
                ok, box = crop_region(source_frame, crop_path, region)
                if not ok or box is None:
                    failures.append(
                        {
                            "candidate_id": candidate_id,
                            "event_id": asset.get("event_id", ""),
                            "frame_path": display_path(source_frame),
                            "region": region.name,
                            "error": "crop failed",
                        }
                    )
                    continue

                x1, y1, x2, y2 = box
                crop_display = display_path(crop_path)
                source_display = display_path(source_frame)
                details = (
                    f"event_id={asset.get('event_id', '')}; victim={asset.get('victim', '')}; "
                    f"region={region.name}; source_frame={source_display}"
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target": TARGET,
                        "reason": region.reason,
                        "source_id": asset.get("event_id", ""),
                        "match_id": asset.get("match_id", ""),
                        "video": asset.get("source_video", ""),
                        "event_id": asset.get("event_id", ""),
                        "elapsed_time": asset.get("time", ""),
                        "row_index": len(rows) + 1,
                        "victim": asset.get("victim", ""),
                        "victim_slot": asset.get("victim_slot", ""),
                        "frame_time": format_seconds(float(frame_time)) if frame_time is not None else "",
                        "frame_path": crop_display,
                        "source_frame_path": source_display,
                        "crop_path": crop_display,
                        "region": region.name,
                        "roi": roi_text(region.roi),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "ocr_text": "",
                        "ocr_confidence": "",
                        "details": details,
                    }
                )

    by_region: dict[str, int] = {}
    for row in rows:
        by_region[str(row["region"])] = by_region.get(str(row["region"]), 0) + 1
    return {
        "status": "ready" if rows else "empty",
        "asset_count": len(assets),
        "candidate_count": len(rows),
        "failure_count": len(failures),
        "by_region": by_region,
        "output_dir": display_path(output_dir),
        "candidates": rows,
        "failures": failures,
    }
