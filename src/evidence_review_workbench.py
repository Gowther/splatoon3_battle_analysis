from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.active_learning_workbench import display_path, safe_project_file, utc_now
from src.core.paths import ROOT, project_path
from src.data_review_workbench import (
    describe_video,
    discover_data_sources,
    discover_video_files,
    is_video_path,
    matching_sources,
    numeric_value,
    read_csv_rows,
)
from src.death_events import build_death_event_report


DEFAULT_EVIDENCE_ROOT = ROOT / "outputs" / "evidence_review"
DEFAULT_REVIEW_LOG_PATH = DEFAULT_EVIDENCE_ROOT / "reviews.jsonl"
DEFAULT_WEAPON_CORRECTION_ROOT = ROOT / "outputs" / "weapon_correction_dataset"
DEFAULT_WEAPON_CORRECTION_LOG_PATH = DEFAULT_WEAPON_CORRECTION_ROOT / "corrections.jsonl"
DEFAULT_WEAPON_LABELS_PATH = ROOT / "main_weapon_list.txt"
WEAPON_CROP_SIZE = 64
WEAPON_FIELDS = [f"weapon_{index}" for index in range(1, 9)]
PLAYER_STATE_FIELDS = [f"player_state_{index}" for index in range(1, 9)]
WEAPON_SLOT_CENTERS_X = (0.337, 0.379, 0.421, 0.463, 0.537, 0.579, 0.621, 0.663)
WEAPON_SLOT_CENTER_Y = 0.073
WEAPON_SLOT_WIDTH_RATIO = 0.045
WEAPON_SLOT_HEIGHT_RATIO = 0.085


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "item"


def load_weapon_labels(labels_path: str | Path = DEFAULT_WEAPON_LABELS_PATH) -> list[str]:
    resolved = project_path(labels_path)
    if resolved.exists():
        return [line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset_root = ROOT / "main_training_dataset"
    if not dataset_root.exists():
        return []
    return sorted(item.name for item in dataset_root.iterdir() if item.is_dir())


def row_time(row: dict[str, Any]) -> float | None:
    return numeric_value(row.get("elapsed_time") or row.get("time"))


def row_weapons(row: dict[str, Any]) -> list[str]:
    return [str(row.get(field, "")).strip() for field in WEAPON_FIELDS]


def has_weapon_data(row: dict[str, Any]) -> bool:
    return any(row_weapons(row))


def has_player_state_fields(fieldnames: list[str]) -> bool:
    return any(field in fieldnames for field in PLAYER_STATE_FIELDS)


def weapon_slot_boxes(width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    box_width = max(1, int(round(width * WEAPON_SLOT_WIDTH_RATIO)))
    box_height = max(1, int(round(height * WEAPON_SLOT_HEIGHT_RATIO)))
    for index, center_x_ratio in enumerate(WEAPON_SLOT_CENTERS_X, start=1):
        left = int(round(width * center_x_ratio - box_width / 2))
        top = int(round(height * WEAPON_SLOT_CENTER_Y - box_height / 2))
        left = max(0, min(left, max(0, width - box_width)))
        top = max(0, min(top, max(0, height - box_height)))
        boxes.append(
            {
                "slot": index,
                "left": left,
                "top": top,
                "width": box_width,
                "height": box_height,
                "left_pct": round(left / max(1, width) * 100, 3),
                "top_pct": round(top / max(1, height) * 100, 3),
                "width_pct": round(box_width / max(1, width) * 100, 3),
                "height_pct": round(box_height / max(1, height) * 100, 3),
            }
        )
    return boxes


def image_slot_boxes(image_path: str) -> list[dict[str, Any]]:
    if not image_path:
        return []
    try:
        from PIL import Image
    except ImportError:
        return []
    resolved = safe_project_file(image_path)
    if not resolved.exists():
        return []
    with Image.open(resolved) as image:
        return weapon_slot_boxes(image.width, image.height)


def analysis_source_score(source: dict[str, Any]) -> tuple[int, str]:
    path = str(source.get("path", ""))
    score = 100
    if source.get("kind") == "analysis_csv":
        score -= 40
    if "/heatmap_" in path and path.endswith("/ui_state.csv"):
        score -= 35
    if "/evaluation/" in path and path.endswith("/smoothed.csv"):
        score -= 25
    if "/analysis_window_scan/" in path:
        score += 15
    if "/validation_suite/" in path:
        score += 20
    if path.endswith("/raw.csv"):
        score += 5
    return score, path


def candidate_analysis_sources(video_path: str) -> list[dict[str, Any]]:
    sources = discover_data_sources()
    matched_paths = set(matching_sources(video_path, sources, limit=100))
    candidates = [
        source
        for source in sources
        if source.get("path") in matched_paths
        and source.get("kind") == "analysis_csv"
        and has_player_state_fields(list(source.get("fieldnames", [])))
    ]
    return sorted(candidates, key=analysis_source_score)


def load_rows_for_source(source: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    return read_csv_rows(str(source.get("path", "")))


def select_weapon_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if has_weapon_data(row):
            return row
    for row in rows:
        if row_time(row) is not None:
            return row
    return rows[0] if rows else None


def export_frame(video_path: Path, time_value: float, output_path: Path) -> str:
    if output_path.exists():
        return display_path(output_path)
    try:
        import cv2
    except ImportError:
        return ""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return ""
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_value)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return ""
    finally:
        cap.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        return ""
    return display_path(output_path)


def build_weapon_evidence(video_path: Path, source: dict[str, Any], rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    row = select_weapon_row(rows)
    time_value = row_time(row or {}) or 0.0
    image_path = export_frame(video_path, time_value, output_dir / f"weapon_{time_value:.3f}s.jpg")
    weapons = row_weapons(row or {})
    return {
        "id": f"weapon:{video_path.stem}",
        "status": "ready" if any(weapons) else "needs_data",
        "source_path": source.get("path", ""),
        "time": round(time_value, 3),
        "image_path": image_path,
        "weapons": [
            {"slot": index + 1, "weapon": weapon or "", "box": {}}
            for index, weapon in enumerate(weapons)
        ],
        "slot_boxes": [],
        "crop_mode": "manual",
        "blocking_reason": "" if any(weapons) else "当前分析 CSV 没有 weapon_1..8 结果",
    }


def build_death_evidence(video_path: Path, source: dict[str, Any], rows: list[dict[str, str]], output_dir: Path) -> dict[str, Any]:
    report = build_death_event_report(rows, match_id=video_path.stem)
    events: list[dict[str, Any]] = []
    for index, event in enumerate(report.get("events", []), start=1):
        time_value = numeric_value(event.get("time")) or 0.0
        event_id = str(event.get("event_id") or f"death:{video_path.stem}:{index}")
        image_path = export_frame(video_path, time_value, output_dir / "deaths" / f"{safe_id(event_id)}_{time_value:.3f}s.jpg")
        events.append(
            {
                **event,
                "review_id": event_id,
                "time": round(time_value, 3),
                "image_path": image_path,
            }
        )
    return {
        "id": f"death:{video_path.stem}",
        "status": "ready" if events else "empty",
        "source_path": source.get("path", ""),
        "event_count": len(events),
        "blocking_reason": "" if events else str(report.get("blocking_reason", "没有检测到死亡状态转移")),
        "events": events,
    }


def build_evidence_review_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ready",
        "videos": discover_video_files(),
        "review_log_path": display_path(DEFAULT_REVIEW_LOG_PATH),
        "weapon_labels": load_weapon_labels(),
        "weapon_correction_dataset": display_path(DEFAULT_WEAPON_CORRECTION_ROOT),
        "weapon_correction_log_path": display_path(DEFAULT_WEAPON_CORRECTION_LOG_PATH),
    }


def build_video_evidence(video_path: str, output_root: str | Path = DEFAULT_EVIDENCE_ROOT) -> dict[str, Any]:
    video = safe_project_file(video_path)
    if not is_video_path(video):
        raise ValueError("unsupported video type")
    if not video.exists():
        raise FileNotFoundError(video)
    sources = candidate_analysis_sources(display_path(video))
    if not sources:
        return {
            "schema_version": 1,
            "status": "needs_data",
            "video": describe_video(video),
            "source": {},
            "weapon": {"status": "missing", "weapons": [], "image_path": "", "blocking_reason": "没有找到匹配的分析 CSV"},
            "death": {"status": "missing", "events": [], "event_count": 0, "blocking_reason": "没有找到匹配的分析 CSV"},
        }

    source = sources[0]
    _fieldnames, rows = load_rows_for_source(source)
    output_dir = project_path(output_root) / safe_id(video.stem)
    weapon = build_weapon_evidence(video, source, rows, output_dir)
    death = build_death_evidence(video, source, rows, output_dir)
    return {
        "schema_version": 1,
        "status": "ready",
        "video": describe_video(video),
        "source": source,
        "candidate_sources": sources[:8],
        "weapon": weapon,
        "death": death,
    }


def record_evidence_review(payload: dict[str, Any], review_path: str | Path = DEFAULT_REVIEW_LOG_PATH) -> dict[str, Any]:
    item_type = str(payload.get("item_type", "")).strip()
    decision = str(payload.get("decision", "")).strip()
    if item_type not in {"weapon", "death"}:
        raise ValueError("item_type must be weapon or death")
    if decision not in {"accurate", "incorrect", "needs_review", "not_death", "skipped"}:
        raise ValueError("unsupported decision")
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "item_type": item_type,
        "item_id": str(payload.get("item_id", "")).strip(),
        "video_path": str(payload.get("video_path", "")).strip(),
        "source_path": str(payload.get("source_path", "")).strip(),
        "decision": decision,
        "note": str(payload.get("note", "")).strip(),
        "payload": payload.get("payload", {}),
    }
    resolved = project_path(review_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "saved", "review": record, "review_log_path": display_path(resolved)}


def _box_number(raw: dict[str, Any], key: str, fallback: float) -> int:
    try:
        return int(round(float(raw.get(key, fallback))))
    except (TypeError, ValueError):
        return int(round(fallback))


def coerce_crop_box(raw_box: Any, slot: int, width: int, height: int) -> dict[str, int]:
    if not isinstance(raw_box, dict):
        raise ValueError("crop_box is required")
    raw = raw_box
    if {"left", "top", "width", "height"}.issubset(raw):
        left = _box_number(raw, "left", 0)
        top = _box_number(raw, "top", 0)
        box_width = _box_number(raw, "width", width)
        box_height = _box_number(raw, "height", height)
    elif {"left_pct", "top_pct", "width_pct", "height_pct"}.issubset(raw):
        left = int(round(float(raw["left_pct"]) * width / 100))
        top = int(round(float(raw["top_pct"]) * height / 100))
        box_width = max(1, int(round(float(raw["width_pct"]) * width / 100)))
        box_height = max(1, int(round(float(raw["height_pct"]) * height / 100)))
    else:
        raise ValueError("crop_box must include left/top/width/height")
    box_width = max(1, min(int(box_width), width))
    box_height = max(1, min(int(box_height), height))
    left = max(0, min(int(left), max(0, width - box_width)))
    top = max(0, min(int(top), max(0, height - box_height)))
    return {"slot": slot, "left": left, "top": top, "width": box_width, "height": box_height}


def crop_weapon_slot_image(image: Any, box: dict[str, int], output_size: int = WEAPON_CROP_SIZE) -> Any:
    from PIL import Image

    side = max(int(box["width"]), int(box["height"]), 1)
    side = int(round(side * 1.12))
    center_x = int(box["left"] + box["width"] / 2)
    center_y = int(box["top"] + box["height"] / 2)
    left = center_x - side // 2
    top = center_y - side // 2
    right = left + side
    bottom = top + side

    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    source_box = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        paste_at = (source_box[0] - left, source_box[1] - top)
        canvas.paste(image.crop(source_box), paste_at)

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return canvas.resize((output_size, output_size), resampling)


def shifted_image(image: Any, offset_x: int, offset_y: int) -> Any:
    from PIL import Image

    shifted = Image.new("RGB", image.size, (0, 0, 0))
    shifted.paste(image, (offset_x, offset_y))
    return shifted


def weapon_crop_variants(image: Any) -> list[tuple[str, Any]]:
    from PIL import ImageEnhance, ImageFilter

    return [
        ("aug_brightness", ImageEnhance.Brightness(image).enhance(1.16)),
        ("aug_contrast", ImageEnhance.Contrast(image).enhance(1.18)),
        ("aug_soft", image.filter(ImageFilter.GaussianBlur(radius=0.45))),
        ("aug_shift", shifted_image(image, 2, -2)),
    ]


def record_weapon_correction(
    payload: dict[str, Any],
    correction_root: str | Path = DEFAULT_WEAPON_CORRECTION_ROOT,
    correction_log_path: str | Path = DEFAULT_WEAPON_CORRECTION_LOG_PATH,
    labels_path: str | Path = DEFAULT_WEAPON_LABELS_PATH,
) -> dict[str, Any]:
    try:
        slot = int(payload.get("slot", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("slot must be 1..8") from exc
    if slot < 1 or slot > 8:
        raise ValueError("slot must be 1..8")

    actual_weapon = str(payload.get("actual_weapon", "")).strip()
    labels = load_weapon_labels(labels_path)
    if actual_weapon not in labels:
        raise ValueError("actual_weapon must be one of main_weapon_list.txt")

    evidence_image = safe_project_file(str(payload.get("evidence_image_path") or payload.get("image_path") or ""))
    if not evidence_image.exists():
        raise FileNotFoundError(evidence_image)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to crop weapon corrections") from exc

    with Image.open(evidence_image) as source_image:
        rgb_image = source_image.convert("RGB")
        crop_box = coerce_crop_box(payload.get("crop_box"), slot, rgb_image.width, rgb_image.height)
        crop = crop_weapon_slot_image(rgb_image, crop_box)

    root = project_path(correction_root)
    target_dir = root / actual_weapon
    target_dir.mkdir(parents=True, exist_ok=True)

    time_value = numeric_value(payload.get("time"))
    time_part = f"{time_value:.3f}s" if time_value is not None else "unknown"
    video_stem = safe_id(Path(str(payload.get("video_path") or evidence_image.stem)).stem)
    suffix = safe_id(utc_now())
    stem = f"{video_stem}_{safe_id(time_part)}_slot{slot}_{suffix}"
    original_path = target_dir / f"{stem}_orig.jpg"
    crop.save(original_path, quality=95)

    augmented_paths: list[str] = []
    for variant_name, variant in weapon_crop_variants(crop):
        variant_path = target_dir / f"{stem}_{variant_name}.jpg"
        variant.save(variant_path, quality=92)
        augmented_paths.append(display_path(variant_path))

    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "video_path": str(payload.get("video_path", "")).strip(),
        "source_path": str(payload.get("source_path", "")).strip(),
        "evidence_image_path": display_path(evidence_image),
        "slot": slot,
        "predicted_weapon": str(payload.get("predicted_weapon", "")).strip(),
        "actual_weapon": actual_weapon,
        "time": round(time_value, 3) if time_value is not None else None,
        "crop_box": crop_box,
        "original_path": display_path(original_path),
        "augmented_paths": augmented_paths,
        "dataset_root": display_path(root),
        "note": str(payload.get("note", "")).strip(),
    }

    log_path = project_path(correction_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "saved",
        "dataset_root": display_path(root),
        "correction_log_path": display_path(log_path),
        "record": record,
    }
