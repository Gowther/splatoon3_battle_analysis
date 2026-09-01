from __future__ import annotations

import csv
import json
import mimetypes
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.core.paths import ROOT, display_path, project_path
from src.data_registry import load_registry
from src.death_annotation_store import (
    DEATH_REVIEW_FIELDS,
    merge_review_rows,
    read_csv_rows as read_death_review_rows,
    staging_item_to_death_review_row,
)


DEFAULT_STATE_DIR = ROOT / "outputs" / "active_learning_workbench"
DEFAULT_STAGING_PATH = DEFAULT_STATE_DIR / "staging_annotations.json"
DEFAULT_LLM_REVIEWS_PATH = DEFAULT_STATE_DIR / "llm_reviews.json"
DEFAULT_ACTION_RUNS_PATH = DEFAULT_STATE_DIR / "action_runs.json"
DEFAULT_AUTOMATION_RUNS_PATH = DEFAULT_STATE_DIR / "automation_runs.json"
DEFAULT_JOBS_PATH = DEFAULT_STATE_DIR / "jobs.json"
DEFAULT_CANDIDATE_MANIFEST = ROOT / "outputs" / "training_sample_candidates" / "manifest.json"
DEFAULT_DEATH_OCR_CANDIDATES = ROOT / "outputs" / "death_events" / "ocr_candidates" / "death_ocr_candidates.csv"
DEFAULT_MODEL_TRAINING_TARGETS = ROOT / "config" / "model_training_targets.json"
DEFAULT_MODEL_REGISTRY = ROOT / "config" / "models.json"


REPORT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "validation_suite",
        "title": "Validation Suite",
        "title_zh": "验证套件",
        "paths": ["outputs/validation_suite.json"],
        "action": "run_validation_suite",
    },
    {
        "id": "validation_samples",
        "title": "Validation Samples",
        "title_zh": "验证样本",
        "paths": ["outputs/validation_suite/validation_samples.json", "outputs/goal2_validation_samples.json"],
    },
    {
        "id": "model_errors",
        "title": "Model Errors",
        "title_zh": "模型错误",
        "paths": ["outputs/validation_suite/model_error_report_smoothed.json", "outputs/model_error_report_smoothed.json"],
    },
    {
        "id": "training_candidates",
        "title": "Training Candidates",
        "title_zh": "训练候选样本",
        "paths": ["outputs/training_sample_candidates/manifest.json", "outputs/training_sample_candidates.json"],
        "action": "refresh_training_candidates",
    },
    {
        "id": "heatmap_labels",
        "title": "Heatmap Labels",
        "title_zh": "热力图标注",
        "paths": ["outputs/heatmap_annotation_round1.json", "outputs/heatmap_annotation_round_goal4.json"],
    },
    {
        "id": "heatmap_comparison",
        "title": "Heatmap Comparison",
        "title_zh": "热力图对比",
        "paths": ["outputs/validation_suite/heatmap_comparison.json", "outputs/heatmap_comparison.json"],
    },
    {
        "id": "death_ocr_candidates",
        "title": "Death OCR Candidates",
        "title_zh": "死亡 OCR 候选",
        "paths": ["outputs/death_events/ocr_candidates/death_ocr_candidates.json"],
    },
    {
        "id": "death_attribution",
        "title": "Death Attribution",
        "title_zh": "死亡归因",
        "paths": ["outputs/death_events/death_attribution_report.json"],
    },
    {
        "id": "training_datasets",
        "title": "Training Datasets",
        "title_zh": "训练数据集",
        "paths": [
            "outputs/validation_suite/model_training_datasets.json",
            "outputs/model_training_datasets.json",
            "outputs/model_training_datasets_goal6_final.json",
        ],
        "action": "validate_training_datasets",
    },
    {
        "id": "model_data_readiness",
        "title": "Model/Data Readiness",
        "title_zh": "模型/数据就绪",
        "paths": ["outputs/model_data_readiness.json", "outputs/model_experiment_baseline_goal7/model_data_readiness.json"],
        "action": "refresh_model_data_readiness",
    },
    {
        "id": "model_registry",
        "title": "Model Registry",
        "title_zh": "模型登记表",
        "paths": ["outputs/validation_suite/model_registry.json", "outputs/model_registry.json"],
    },
    {
        "id": "runtime_benchmarks",
        "title": "Runtime Benchmarks",
        "title_zh": "运行时基准",
        "paths": ["outputs/runtime/runtime_benchmarks.json", "outputs/goal2_runtime_benchmarks.json"],
    },
    {
        "id": "model_baseline",
        "title": "Model Baseline",
        "title_zh": "模型基线",
        "paths": [
            "outputs/model_benchmarks/baseline_snapshot.json",
            "outputs/model_experiment_baseline_goal7/baseline_snapshot.json",
        ],
        "action": "run_model_baseline",
    },
    {
        "id": "promotion_plan",
        "title": "Promotion Plan",
        "title_zh": "模型提升计划",
        "paths": ["outputs/model_promotion_plan.json", "outputs/goal3_model_promotion_plan.json"],
        "action": "promotion_plan",
    },
)


TARGET_DATASET_PATHS: dict[str, dict[str, str]] = {
    "ui_detector_yolo": {
        "train_images": "yolov5/train/images",
        "train_labels": "yolov5/train/labels",
        "val_images": "yolov5/valid/images",
        "val_labels": "yolov5/valid/labels",
    },
    "count_ocr_yolo": {
        "train_images": "outputs/model_training/count_ocr_dataset/images/train",
        "train_labels": "outputs/model_training/count_ocr_dataset/labels/train",
        "val_images": "outputs/model_training/count_ocr_dataset/images/val",
        "val_labels": "outputs/model_training/count_ocr_dataset/labels/val",
    },
    "message_ocr_yolo": {
        "train_images": "outputs/model_training/message_ocr_dataset/images/train",
        "train_labels": "outputs/model_training/message_ocr_dataset/labels/train",
        "val_images": "outputs/model_training/message_ocr_dataset/images/val",
        "val_labels": "outputs/model_training/message_ocr_dataset/labels/val",
    },
}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ActionDefinition:
    id: str
    label: str
    description: str
    needs_confirmation: bool = False
    long_running: bool = False
    label_zh: str = ""
    description_zh: str = ""
    automation_safe: bool = True
    human_gate: bool = False


ACTION_DEFINITIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(
        "refresh_training_candidates",
        "Refresh Candidates",
        "Rebuild failed-sample candidate queues.",
        label_zh="刷新候选样本",
        description_zh="重新生成失败样本候选队列。",
    ),
    ActionDefinition(
        "run_validation_suite",
        "Run Validation",
        "Run validation suite; optionally rerun analysis.",
        long_running=True,
        label_zh="运行验证",
        description_zh="运行验证套件，可选择重新分析。",
    ),
    ActionDefinition(
        "intake_video",
        "Intake Video",
        "Register one new video and optionally scan analysis windows.",
        label_zh="接入视频",
        description_zh="登记一个新视频，并可选扫描分析窗口。",
    ),
    ActionDefinition(
        "validate_training_datasets",
        "Validate Datasets",
        "Check configured model training datasets.",
        label_zh="验证训练集",
        description_zh="检查已配置的模型训练数据集。",
    ),
    ActionDefinition(
        "refresh_model_data_readiness",
        "Refresh Readiness",
        "Refresh model/data readiness report.",
        label_zh="刷新就绪状态",
        description_zh="刷新模型/数据就绪报告。",
    ),
    ActionDefinition(
        "training_dry_run",
        "Training Dry Run",
        "Build launch plan for a configured training target.",
        label_zh="训练预演",
        description_zh="为指定训练目标生成启动计划。",
    ),
    ActionDefinition(
        "training_execute",
        "Execute Training",
        "Run the configured training command for a target.",
        needs_confirmation=True,
        long_running=True,
        label_zh="执行训练",
        description_zh="运行指定训练目标的配置命令。",
        automation_safe=False,
        human_gate=True,
    ),
    ActionDefinition(
        "run_model_baseline",
        "Run Baseline",
        "Generate a model experiment baseline package.",
        long_running=True,
        label_zh="运行基线",
        description_zh="生成模型实验基线包。",
    ),
    ActionDefinition(
        "promotion_plan",
        "Promotion Plan",
        "Build a candidate model promotion plan.",
        label_zh="提升计划",
        description_zh="生成候选模型提升计划。",
    ),
    ActionDefinition(
        "promotion_apply",
        "Apply Promotion",
        "Copy a validated candidate into the registered model path.",
        needs_confirmation=True,
        label_zh="应用提升",
        description_zh="把已验证候选模型复制到登记的正式模型路径。",
        automation_safe=False,
        human_gate=True,
    ),
)

ACTION_BY_ID = {definition.id: definition for definition in ACTION_DEFINITIONS}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: str | Path, default: Any) -> Any:
    if not str(path):
        return default
    resolved = project_path(path)
    if not resolved.exists() or resolved.is_dir():
        return default
    with resolved.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    if not str(path):
        return []
    resolved = project_path(path)
    if not resolved.exists() or resolved.is_dir():
        return []
    with resolved.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return resolved


def first_existing_path(paths: list[str]) -> Path | None:
    for item in paths:
        candidate = project_path(item)
        if candidate.exists():
            return candidate
    return None


def payload_status(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("status") or payload.get("overall") or "ready")
    if isinstance(payload, list):
        return "ready"
    return "ready" if payload else "missing"


def payload_counts(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"rows": len(payload)}
    if not isinstance(payload, dict):
        return {}
    counts: dict[str, Any] = {}
    for key in ("target_count", "file_count", "model_error_status", "heatmap_comparison_status"):
        if key in payload:
            counts[key] = payload[key]
    if isinstance(payload.get("summary"), dict):
        counts.update({f"summary_{key}": value for key, value in payload["summary"].items() if isinstance(value, (int, float, str))})
    if isinstance(payload.get("target_rows"), dict):
        counts["target_rows"] = payload["target_rows"]
    for key in ("event_count", "candidate_count", "attributed_count", "failure_count"):
        if key in payload:
            counts[key] = payload[key]
    if isinstance(payload.get("progress"), dict):
        progress = payload["progress"]
        counts["labeled_rows"] = progress.get("labeled_rows", 0)
        counts["unlabeled_rows"] = progress.get("unlabeled_rows", 0)
    return counts


def load_report_summaries(root: Path = ROOT) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for spec in REPORT_SPECS:
        path = first_existing_path(list(spec["paths"]))
        payload = read_json(path, {}) if path else {}
        summaries.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "title_zh": spec.get("title_zh", spec["title"]),
                "status": payload_status(payload) if path else "missing",
                "path": display_path(path) if path else "",
                "action": spec.get("action", ""),
                "counts": payload_counts(payload),
                "missing_candidates": [display_path(root / item) for item in spec["paths"] if not project_path(item).exists()],
            }
        )
    return summaries


def scan_asset_inbox(root: Path = ROOT, registry_path: Path = ROOT / "config" / "data_registry.json") -> dict[str, Any]:
    registry = load_registry(registry_path)
    registered = {
        display_path(match.get("video", ""))
        for match in registry.get("matches", [])
        if isinstance(match, dict) and match.get("video")
    }
    footages = root / "footages"
    videos: list[dict[str, Any]] = []
    if footages.exists():
        for path in sorted(item for item in footages.iterdir() if item.suffix.lower() in VIDEO_EXTENSIONS):
            relative = display_path(path)
            registered_match = next(
                (
                    str(match.get("id"))
                    for match in registry.get("matches", [])
                    if display_path(match.get("video", "")) == relative
                ),
                "",
            )
            videos.append(
                {
                    "path": relative,
                    "suggested_match_id": path.stem,
                    "status": "registered" if relative in registered else "new",
                    "registered_match_id": registered_match,
                }
            )
    return {
        "status": "needs_intake" if any(item["status"] == "new" for item in videos) else "ready",
        "video_count": len(videos),
        "new_count": sum(1 for item in videos if item["status"] == "new"),
        "videos": videos,
    }


def load_staging(path: Path = DEFAULT_STAGING_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {"schema_version": 1, "updated_at": "", "items": []}
    payload.setdefault("items", [])
    return payload


def staging_by_id(staging: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): item for item in staging.get("items", []) if isinstance(item, dict)}


def load_llm_reviews(path: Path = DEFAULT_LLM_REVIEWS_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {"schema_version": 1, "updated_at": "", "reviews": {}}
    payload.setdefault("reviews", {})
    return payload


def numeric_value(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def text_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return ""


def candidate_annotation_type(target: str) -> str:
    if target == "heatmap_tracker_labels":
        return "heatmap_point"
    if target == "weapon_classifier_resnet18":
        return "classification"
    if target in {"count_ocr_yolo", "message_ocr_yolo", "death_event_ocr"}:
        return "ocr_box_text"
    return "yolo_box"


def build_box_preannotation(raw: dict[str, Any]) -> dict[str, Any] | None:
    direct = {
        "x_center": numeric_value(raw, "x_center", "xc", "cx"),
        "y_center": numeric_value(raw, "y_center", "yc", "cy"),
        "width": numeric_value(raw, "width", "w"),
        "height": numeric_value(raw, "height", "h"),
    }
    if all(value is not None for value in direct.values()):
        box = {key: float(value) for key, value in direct.items() if value is not None}
        box["class_id"] = int(numeric_value(raw, "class_id", "class", "label_id") or 0)
        class_name = text_value(raw, "class_name", "label", "name")
        if class_name:
            box["class_name"] = class_name
        return box

    x1 = numeric_value(raw, "x1", "xmin", "x_min", "left")
    y1 = numeric_value(raw, "y1", "ymin", "y_min", "top")
    x2 = numeric_value(raw, "x2", "xmax", "x_max", "right")
    y2 = numeric_value(raw, "y2", "ymax", "y_max", "bottom")
    if None in {x1, y1, x2, y2}:
        return None
    image_width = numeric_value(raw, "image_width", "img_width", "width_px")
    image_height = numeric_value(raw, "image_height", "img_height", "height_px")
    if image_width and image_height and max(float(x2), float(y2)) > 1:
        x1, x2 = float(x1) / image_width, float(x2) / image_width
        y1, y2 = float(y1) / image_height, float(y2) / image_height
    box = {
        "class_id": int(numeric_value(raw, "class_id", "class", "label_id") or 0),
        "x_center": (float(x1) + float(x2)) / 2,
        "y_center": (float(y1) + float(y2)) / 2,
        "width": abs(float(x2) - float(x1)),
        "height": abs(float(y2) - float(y1)),
    }
    class_name = text_value(raw, "class_name", "label", "name")
    if class_name:
        box["class_name"] = class_name
    return box


def build_candidate_preannotation(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = candidate.get("raw", {}) if isinstance(candidate.get("raw"), dict) else {}
    target = str(candidate.get("target", ""))
    annotation: dict[str, Any] = {}
    confidence = numeric_value(raw, "confidence", "score", "model_confidence")

    if target == "heatmap_tracker_labels":
        x = numeric_value(raw, "x", "player_x", "track_x")
        y = numeric_value(raw, "y", "player_y", "track_y")
        if x is not None and y is not None:
            annotation["point"] = {
                "x": f"{x:.1f}",
                "y": f"{y:.1f}",
                "visibility": "visible",
            }
    elif candidate.get("annotation_type") in {"yolo_box", "ocr_box_text"}:
        box = build_box_preannotation(raw)
        if box:
            annotation["boxes"] = [box]

    text = text_value(raw, "text", "ocr_text", "value", "recognized_text")
    if text:
        annotation["text"] = text
    note = text_value(raw, "note", "details")
    if note:
        annotation["notes"] = note

    label_ready = "point" in annotation or "boxes" in annotation
    if not annotation:
        return {
            "status": "empty",
            "source": "",
            "annotation": {},
            "confidence": confidence,
            "needs_human": True,
        }
    return {
        "status": "ready" if label_ready else "metadata",
        "source": "candidate_raw",
        "annotation": annotation,
        "confidence": confidence,
        "needs_human": confidence is None or confidence < 0.85,
    }


def candidate_group_key(candidate: dict[str, Any]) -> str:
    target = str(candidate.get("target", ""))
    status = str(candidate.get("status", "todo"))
    if status not in {"todo", "draft"}:
        return "|".join([target, status, str(candidate.get("id", ""))])
    match_id = str(candidate.get("match_id", ""))
    source_id = str(candidate.get("source_id", ""))
    reason = str(candidate.get("reason", ""))
    raw = candidate.get("raw", {}) if isinstance(candidate.get("raw"), dict) else {}
    elapsed = numeric_value(candidate, "elapsed_time")
    time_bucket = "unknown"
    if elapsed is not None:
        time_bucket = str(int(elapsed // 2))
    if target == "heatmap_tracker_labels":
        slot = text_value(raw, "track_slot", "player_id", "team")
        return "|".join([target, match_id, reason, time_bucket, slot])
    if target == "death_event_ocr":
        region = text_value(raw, "region", "reason")
        event_id = text_value(raw, "event_id", "source_id")
        return "|".join([target, match_id, event_id, region, time_bucket])
    return "|".join([target, match_id, source_id, reason, time_bucket])


def candidate_priority(candidate: dict[str, Any]) -> float:
    score = 0.0
    status = str(candidate.get("status", "todo"))
    if status == "todo":
        score += 100
    elif status == "draft":
        score += 80
    reason = str(candidate.get("reason", ""))
    if "missing" in reason:
        score += 25
    if "jump" in reason or "large_step" in reason:
        score += 20
    if candidate.get("preannotation", {}).get("status") == "ready":
        score += 15
    raw = candidate.get("raw", {}) if isinstance(candidate.get("raw"), dict) else {}
    severity = numeric_value(raw, "severity", "step_distance")
    if severity is not None:
        score += min(severity / 100, 25)
    confidence = numeric_value(raw, "confidence")
    if confidence is not None:
        score += max(0, (1 - confidence) * 20)
    if not (candidate.get("frame_path") or candidate.get("preview_path")):
        score -= 50
    return round(score, 3)


def rank_candidate_queue(candidates: list[dict[str, Any]], *, dedupe: bool = True) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["group_key"] = candidate_group_key(candidate)
        candidate["priority_score"] = candidate_priority(candidate)
        groups.setdefault(candidate["group_key"], []).append(candidate)

    if not dedupe:
        for items in groups.values():
            ranked.extend(items)
    else:
        for items in groups.values():
            items = sorted(items, key=lambda item: (-float(item.get("priority_score", 0)), str(item.get("id", ""))))
            representative = dict(items[0])
            representative["duplicate_count"] = len(items)
            representative["group_member_ids"] = [str(item.get("id", "")) for item in items]
            ranked.append(representative)
    return sorted(ranked, key=lambda item: (-float(item.get("priority_score", 0)), str(item.get("id", ""))))


def normalize_candidate(row: dict[str, str], *, target: str, row_index: int) -> dict[str, Any]:
    candidate_id = row.get("candidate_id") or f"{target}:{row.get('match_id', 'unknown')}:{row_index:04d}"
    frame_path = row.get("frame_path") or row.get("exported_frame") or row.get("preview_path") or ""
    candidate = {
        "id": candidate_id,
        "target": target,
        "annotation_type": candidate_annotation_type(target),
        "reason": row.get("reason") or row.get("anomaly_type") or "",
        "source_id": row.get("source_id") or row.get("heatmap_id") or "",
        "match_id": row.get("match_id", ""),
        "video": row.get("video", ""),
        "elapsed_time": row.get("elapsed_time") or row.get("time") or "",
        "row_index": row.get("row_index") or row_index,
        "frame_path": frame_path,
        "preview_path": row.get("preview_path", ""),
        "details": row.get("details") or row.get("note") or "",
        "raw": row,
    }
    candidate["preannotation"] = build_candidate_preannotation(candidate)
    return candidate


def load_candidate_queue(
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
    *,
    dedupe: bool = True,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path, {})
    staging = staging_by_id(load_staging(staging_path))
    reviews = load_llm_reviews(reviews_path).get("reviews", {})
    candidates: list[dict[str, Any]] = []
    analysis_targets = (manifest or {}).get("analysis", {}).get("targets", {}) if isinstance(manifest, dict) else {}
    for target, report in analysis_targets.items():
        for index, row in enumerate(read_csv_rows(report.get("csv", "")), start=1):
            candidate = normalize_candidate(row, target=str(target), row_index=index)
            candidate["status"] = str(staging.get(candidate["id"], {}).get("status", "todo"))
            candidate["staging"] = staging.get(candidate["id"], {})
            candidate["llm_review"] = reviews.get(candidate["id"], {})
            candidates.append(candidate)

    heatmap_csv = (manifest or {}).get("heatmap", {}).get("anomalies_csv", "") if isinstance(manifest, dict) else ""
    for index, row in enumerate(read_csv_rows(heatmap_csv), start=1):
        target = "heatmap_tracker_labels"
        candidate = normalize_candidate(row, target=target, row_index=index)
        candidate["id"] = (
            f"{target}:{row.get('match_id', 'unknown')}:"
            f"{row.get('time', row.get('elapsed_time', '0'))}:{row.get('track_slot', index)}:{index:04d}"
        )
        candidate["status"] = str(staging.get(candidate["id"], {}).get("status", "todo"))
        candidate["staging"] = staging.get(candidate["id"], {})
        candidate["llm_review"] = reviews.get(candidate["id"], {})
        candidates.append(candidate)

    death_csv = ""
    if isinstance(manifest, dict):
        death_csv = str((manifest.get("death_events", {}) or {}).get("ocr_candidates_csv", ""))
    if not death_csv and DEFAULT_DEATH_OCR_CANDIDATES.exists():
        death_csv = str(DEFAULT_DEATH_OCR_CANDIDATES)
    for index, row in enumerate(read_csv_rows(death_csv), start=1):
        target = row.get("target") or "death_event_ocr"
        candidate = normalize_candidate(row, target=target, row_index=index)
        candidate["status"] = str(staging.get(candidate["id"], {}).get("status", "todo"))
        candidate["staging"] = staging.get(candidate["id"], {})
        candidate["llm_review"] = reviews.get(candidate["id"], {})
        candidates.append(candidate)
    return rank_candidate_queue(candidates, dedupe=dedupe)


def summarize_queue(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_target: dict[str, int] = {}
    hidden_duplicates = 0
    for item in candidates:
        by_status[str(item.get("status", "todo"))] = by_status.get(str(item.get("status", "todo")), 0) + 1
        by_target[str(item.get("target", ""))] = by_target.get(str(item.get("target", "")), 0) + 1
        hidden_duplicates += max(0, int(item.get("duplicate_count", 1) or 1) - 1)
    return {
        "status": "needs_human" if by_status.get("todo", 0) else "ready",
        "total": len(candidates),
        "hidden_duplicates": hidden_duplicates,
        "by_status": by_status,
        "by_target": by_target,
    }


def summarize_staging(staging: dict[str, Any]) -> dict[str, Any]:
    items = [item for item in staging.get("items", []) if isinstance(item, dict)]
    by_status: dict[str, int] = {}
    by_target: dict[str, int] = {}
    for item in items:
        by_status[str(item.get("status", "draft"))] = by_status.get(str(item.get("status", "draft")), 0) + 1
        by_target[str(item.get("target", ""))] = by_target.get(str(item.get("target", "")), 0) + 1
    return {
        "status": "has_ready_items" if by_status.get("done", 0) else ("has_drafts" if items else "empty"),
        "total": len(items),
        "by_status": by_status,
        "by_target": by_target,
    }


def action_catalog() -> list[dict[str, Any]]:
    return [definition.__dict__ for definition in ACTION_DEFINITIONS]


def action_definition(action_id: str) -> ActionDefinition:
    try:
        return ACTION_BY_ID[action_id]
    except KeyError as exc:
        raise ValueError(f"unknown action: {action_id}") from exc


def build_workbench_state(
    *,
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
) -> dict[str, Any]:
    candidates = load_candidate_queue(manifest_path, staging_path, reviews_path)
    staging = load_staging(staging_path)
    inbox = scan_asset_inbox()
    reports = load_report_summaries()
    blockers = [
        report
        for report in reports
        if report["status"] in {"failed", "blocked", "needs_data", "needs_labels", "missing"}
    ]
    return {
        "schema_version": 1,
        "status": "needs_attention" if blockers or inbox["new_count"] or candidates else "ready",
        "updated_at": utc_now(),
        "reports": reports,
        "asset_inbox": inbox,
        "queue_summary": summarize_queue(candidates),
        "staging_summary": summarize_staging(staging),
        "recent_actions": read_json(DEFAULT_ACTION_RUNS_PATH, {"runs": []}).get("runs", [])[-10:],
        "recent_jobs": load_jobs().get("jobs", [])[-10:],
        "automation_plan": build_automation_plan_from_state(
            {
                "reports": reports,
                "asset_inbox": inbox,
                "queue_summary": summarize_queue(candidates),
                "staging_summary": summarize_staging(staging),
            }
        ),
        "actions": action_catalog(),
    }


def build_automation_plan_from_state(state: dict[str, Any]) -> dict[str, Any]:
    reports = {str(report.get("id")): report for report in state.get("reports", []) if isinstance(report, dict)}
    inbox = state.get("asset_inbox", {}) if isinstance(state.get("asset_inbox"), dict) else {}
    queue_summary = state.get("queue_summary", {}) if isinstance(state.get("queue_summary"), dict) else {}
    staging_summary = state.get("staging_summary", {}) if isinstance(state.get("staging_summary"), dict) else {}
    steps: list[dict[str, Any]] = []

    for video in inbox.get("videos", []):
        if not isinstance(video, dict) or video.get("status") != "new":
            continue
        steps.append(
            {
                "id": f"intake:{video.get('suggested_match_id', '')}",
                "kind": "action",
                "action_id": "intake_video",
                "payload": {
                    "video": video.get("path", ""),
                    "match_id": video.get("suggested_match_id", ""),
                    "scan_analysis_windows": True,
                },
                "status": "runnable",
                "reason": "new footage can be registered automatically",
            }
        )

    validation_status = str(reports.get("validation_suite", {}).get("status", "missing"))
    if validation_status in {"missing", "failed", "blocked"}:
        steps.append(
            {
                "id": "run_validation_suite",
                "kind": "action",
                "action_id": "run_validation_suite",
                "payload": {"run_analysis": False},
                "status": "runnable",
                "reason": f"validation_suite is {validation_status}",
            }
        )

    candidates_status = str(reports.get("training_candidates", {}).get("status", "missing"))
    if candidates_status in {"missing", "failed", "blocked"}:
        steps.append(
            {
                "id": "refresh_training_candidates",
                "kind": "action",
                "action_id": "refresh_training_candidates",
                "payload": {},
                "status": "runnable",
                "reason": f"training_candidates is {candidates_status}",
            }
        )

    if int(staging_summary.get("by_status", {}).get("done", 0) or 0):
        steps.append(
            {
                "id": "apply_staging_dry_run",
                "kind": "apply_staging_dry_run",
                "payload": {"dry_run": True},
                "status": "runnable",
                "reason": "done staging annotations should be validated before apply",
            }
        )

    training_datasets_status = str(reports.get("training_datasets", {}).get("status", "missing"))
    if training_datasets_status in {"missing", "needs_data", "needs_review", "failed", "blocked"}:
        steps.append(
            {
                "id": "validate_training_datasets",
                "kind": "action",
                "action_id": "validate_training_datasets",
                "payload": {},
                "status": "runnable",
                "reason": f"training_datasets is {training_datasets_status}",
            }
        )

    readiness_status = str(reports.get("model_data_readiness", {}).get("status", "missing"))
    if readiness_status in {"missing", "needs_data", "needs_review", "failed", "blocked"}:
        steps.append(
            {
                "id": "refresh_model_data_readiness",
                "kind": "action",
                "action_id": "refresh_model_data_readiness",
                "payload": {},
                "status": "runnable",
                "reason": f"model_data_readiness is {readiness_status}",
            }
        )

    if int(queue_summary.get("by_status", {}).get("todo", 0) or 0):
        steps.append(
            {
                "id": "annotate_candidates",
                "kind": "human_gate",
                "status": "needs_human",
                "reason": f"{queue_summary.get('by_status', {}).get('todo', 0)} candidate groups still need review",
            }
        )
    if str(reports.get("heatmap_labels", {}).get("status", "")) == "needs_labels":
        steps.append(
            {
                "id": "heatmap_labels",
                "kind": "human_gate",
                "status": "needs_human",
                "reason": "heatmap labels still need human confirmation",
            }
        )
    return {
        "schema_version": 1,
        "status": "ready" if not steps else "has_steps",
        "runnable_count": sum(1 for step in steps if step.get("status") == "runnable"),
        "human_gate_count": sum(1 for step in steps if step.get("kind") == "human_gate"),
        "steps": steps,
    }


def build_automation_plan(
    *,
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
) -> dict[str, Any]:
    return build_automation_plan_from_state(
        build_workbench_state(manifest_path=manifest_path, staging_path=staging_path, reviews_path=reviews_path)
    )


def upsert_staging_annotation(
    payload: dict[str, Any],
    *,
    staging_path: Path = DEFAULT_STAGING_PATH,
) -> dict[str, Any]:
    item_id = str(payload.get("id", "")).strip()
    if not item_id:
        raise ValueError("annotation id is required")
    staging = load_staging(staging_path)
    existing = staging_by_id(staging)
    candidate = payload.get("candidate", {}) if isinstance(payload.get("candidate"), dict) else {}
    item = dict(existing.get(item_id, {}))
    item.update(
        {
            "id": item_id,
            "target": str(payload.get("target") or item.get("target") or candidate.get("target", "")),
            "annotation_type": str(
                payload.get("annotation_type") or item.get("annotation_type") or candidate.get("annotation_type", "")
            ),
            "status": str(payload.get("status") or item.get("status") or "draft"),
            "split": str(payload.get("split") or item.get("split") or "train"),
            "candidate": candidate or item.get("candidate", {}),
            "annotation": payload.get("annotation", item.get("annotation", {})),
            "source": str(payload.get("source") or item.get("source") or "human"),
            "updated_at": utc_now(),
        }
    )
    validation_errors = validate_staging_item(item) if item.get("status") == "done" else []
    item["validation_errors"] = validation_errors
    item["validation_status"] = "needs_fix" if validation_errors else ("ready" if item.get("status") == "done" else "draft")
    items = [entry for entry in staging.get("items", []) if isinstance(entry, dict) and str(entry.get("id")) != item_id]
    items.append(item)
    staging["items"] = sorted(items, key=lambda entry: str(entry.get("id", "")))
    staging["updated_at"] = utc_now()
    write_json(staging_path, staging)
    return item


def validate_yolo_boxes(boxes: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not boxes:
        return ["at least one box is required"]
    for index, box in enumerate(boxes, start=1):
        try:
            class_id = int(box.get("class_id"))
            x_center = float(box.get("x_center"))
            y_center = float(box.get("y_center"))
            width = float(box.get("width"))
            height = float(box.get("height"))
        except (TypeError, ValueError):
            errors.append(f"box {index} has non-numeric YOLO values")
            continue
        if class_id < 0:
            errors.append(f"box {index} class_id must be >= 0")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            errors.append(f"box {index} center must be normalized 0..1")
        if not (0 < width <= 1 and 0 < height <= 1):
            errors.append(f"box {index} size must be normalized 0..1")
    return errors


def validate_staging_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    target = str(item.get("target", ""))
    candidate = item.get("candidate", {}) if isinstance(item.get("candidate"), dict) else {}
    annotation = item.get("annotation", {}) if isinstance(item.get("annotation"), dict) else {}
    if item.get("status") != "done":
        errors.append("item is not marked done")
    image_path = candidate.get("frame_path") or annotation.get("image_path")
    if target in TARGET_DATASET_PATHS and not image_path:
        errors.append("candidate frame_path is required")
    elif image_path and not project_path(str(image_path)).exists():
        errors.append(f"image does not exist: {image_path}")
    if target in TARGET_DATASET_PATHS:
        boxes = annotation.get("boxes", [])
        errors.extend(validate_yolo_boxes(boxes if isinstance(boxes, list) else []))
    elif target == "heatmap_tracker_labels":
        point = annotation.get("point", {})
        if not isinstance(point, dict) or point.get("x") in (None, "") or point.get("y") in (None, ""):
            errors.append("heatmap point x/y is required")
    elif target == "death_event_ocr":
        if not (annotation.get("text") or annotation.get("notes")):
            errors.append("death event OCR text or notes are required")
    return errors


def safe_dataset_stem(item_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in item_id)[:140]


def label_lines(annotation: dict[str, Any]) -> list[str]:
    lines = []
    for box in annotation.get("boxes", []):
        lines.append(
            " ".join(
                [
                    str(int(box["class_id"])),
                    f"{float(box['x_center']):.6f}",
                    f"{float(box['y_center']):.6f}",
                    f"{float(box['width']):.6f}",
                    f"{float(box['height']):.6f}",
                ]
            )
        )
    return lines


def apply_staging_annotations(
    *,
    staging_path: Path = DEFAULT_STAGING_PATH,
    dry_run: bool = True,
    report_path: Path = DEFAULT_STATE_DIR / "apply_report.json",
    death_labels_path: Path = DEFAULT_STATE_DIR / "death_event_ocr_labels.csv",
) -> dict[str, Any]:
    staging = load_staging(staging_path)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    death_rows: list[dict[str, Any]] = []
    for item in staging.get("items", []):
        if not isinstance(item, dict) or item.get("status") != "done":
            continue
        target = str(item.get("target", ""))
        errors = validate_staging_item(item)
        if errors:
            skipped.append({"id": item.get("id", ""), "target": target, "errors": errors})
            continue
        annotation = item.get("annotation", {}) if isinstance(item.get("annotation"), dict) else {}
        candidate = item.get("candidate", {}) if isinstance(item.get("candidate"), dict) else {}
        if target == "heatmap_tracker_labels":
            point = annotation.get("point", {})
            heatmap_rows.append(
                {
                    "candidate_id": item.get("id", ""),
                    "match_id": candidate.get("match_id", ""),
                    "time": candidate.get("elapsed_time", ""),
                    "x": point.get("x", ""),
                    "y": point.get("y", ""),
                    "visibility": point.get("visibility", "visible"),
                    "notes": annotation.get("notes", ""),
                }
            )
            applied.append({"id": item.get("id", ""), "target": target, "destination": "heatmap_staging_labels.csv"})
            continue
        if target == "death_event_ocr":
            death_rows.append(staging_item_to_death_review_row(item))
            applied.append({"id": item.get("id", ""), "target": target, "destination": "death_event_ocr_labels.csv"})
            continue
        dataset = TARGET_DATASET_PATHS.get(target)
        if not dataset:
            skipped.append({"id": item.get("id", ""), "target": target, "errors": ["target is not supported for apply"]})
            continue
        split = "val" if str(item.get("split", "train")) in {"val", "valid"} else "train"
        image_dir = project_path(dataset[f"{split}_images"])
        label_dir = project_path(dataset[f"{split}_labels"])
        source_image = project_path(str(candidate.get("frame_path") or annotation.get("image_path")))
        stem = safe_dataset_stem(str(item.get("id", source_image.stem)))
        image_dest = image_dir / f"{stem}{source_image.suffix.lower() if source_image.suffix else '.jpg'}"
        label_dest = label_dir / f"{stem}.txt"
        meta_dest = label_dir / f"{stem}.json"
        if not dry_run:
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, image_dest)
            label_dest.write_text("\n".join(label_lines(annotation)) + "\n", encoding="utf-8")
            meta_dest.write_text(
                json.dumps(
                    {
                        "source_candidate": candidate,
                        "annotation": annotation,
                        "applied_from": display_path(staging_path),
                        "applied_at": utc_now(),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        applied.append(
            {
                "id": item.get("id", ""),
                "target": target,
                "split": split,
                "image": display_path(image_dest),
                "label": display_path(label_dest),
            }
        )
    heatmap_csv = DEFAULT_STATE_DIR / "heatmap_staging_labels.csv"
    if heatmap_rows and not dry_run:
        write_csv_rows(heatmap_csv, heatmap_rows, ["candidate_id", "match_id", "time", "x", "y", "visibility", "notes"])
    if death_rows and not dry_run:
        existing_death_rows = read_death_review_rows(death_labels_path)
        merged_death_rows = merge_review_rows(existing_death_rows, death_rows)
        write_csv_rows(death_labels_path, merged_death_rows, DEATH_REVIEW_FIELDS)
    report = {
        "schema_version": 1,
        "status": "ready" if not skipped else "needs_review",
        "dry_run": dry_run,
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied": applied,
        "skipped": skipped,
        "heatmap_labels_csv": display_path(heatmap_csv) if heatmap_rows else "",
        "death_event_labels_csv": display_path(death_labels_path) if death_rows else "",
        "updated_at": utc_now(),
    }
    write_json(report_path, report)
    return report


def build_llm_review_pack(
    *,
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
    limit: int = 30,
    output_path: Path = DEFAULT_STATE_DIR / "llm_review_pack.json",
) -> dict[str, Any]:
    candidates = [
        item
        for item in load_candidate_queue(manifest_path, staging_path, reviews_path)
        if item.get("status") in {"todo", "draft"}
    ][:limit]
    pack = {
        "schema_version": 1,
        "status": "ready" if candidates else "empty",
        "created_at": utc_now(),
        "instructions": [
            "Review each candidate image and return JSON suggestions keyed by id.",
            "Do not mark final approval; provide suggested labels, confidence, and rationale.",
            "Use status needs_human when uncertain or when the image is not enough evidence.",
        ],
        "expected_review_schema": {
            "id": "candidate id",
            "suggestion": "short suggested annotation",
            "confidence": "0..1",
            "needs_human": "boolean",
            "rationale": "short reason",
        },
        "tasks": [
            {
                "id": item["id"],
                "target": item.get("target", ""),
                "annotation_type": item.get("annotation_type", ""),
                "reason": item.get("reason", ""),
                "frame_path": item.get("frame_path", ""),
                "details": item.get("details", ""),
            }
            for item in candidates
        ],
    }
    write_json(output_path, pack)
    return pack


def record_llm_review(
    item_id: str,
    review: dict[str, Any],
    *,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
) -> dict[str, Any]:
    if not item_id:
        raise ValueError("item_id is required")
    payload = load_llm_reviews(reviews_path)
    review = dict(review)
    review["updated_at"] = utc_now()
    payload["reviews"][item_id] = review
    payload["updated_at"] = utc_now()
    write_json(reviews_path, payload)
    return review


def heuristic_llm_review(candidate: dict[str, Any]) -> dict[str, Any]:
    preannotation = candidate.get("preannotation", {}) if isinstance(candidate.get("preannotation"), dict) else {}
    has_image = bool(candidate.get("frame_path") or candidate.get("preview_path"))
    if not has_image:
        return {
            "suggestion": "skip_missing_image",
            "confidence": 0.95,
            "needs_human": False,
            "rationale": "Candidate has no frame or preview image, so it cannot be labeled from the workbench.",
            "source": "heuristic",
        }
    if preannotation.get("status") == "ready":
        return {
            "suggestion": "review_preannotation",
            "confidence": preannotation.get("confidence") if preannotation.get("confidence") is not None else 0.65,
            "needs_human": bool(preannotation.get("needs_human", True)),
            "rationale": "Candidate includes raw coordinates that can be loaded as a draft preannotation.",
            "source": "heuristic",
        }
    return {
        "suggestion": "needs_visual_label",
        "confidence": 0.4,
        "needs_human": True,
        "rationale": "No machine-readable label coordinates are available; visual confirmation is required.",
        "source": "heuristic",
    }


def auto_record_llm_reviews(
    *,
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
    limit: int = 30,
) -> dict[str, Any]:
    candidates = [
        item
        for item in load_candidate_queue(manifest_path, staging_path, reviews_path, dedupe=True)
        if item.get("status") in {"todo", "draft"}
    ][:limit]
    payload = load_llm_reviews(reviews_path)
    recorded: list[dict[str, Any]] = []
    for candidate in candidates:
        review = heuristic_llm_review(candidate)
        review["updated_at"] = utc_now()
        payload["reviews"][str(candidate["id"])] = review
        recorded.append({"id": candidate["id"], **review})
    payload["updated_at"] = utc_now()
    write_json(reviews_path, payload)
    return {
        "schema_version": 1,
        "status": "ready" if recorded else "empty",
        "recorded_count": len(recorded),
        "reviews": recorded,
        "updated_at": utc_now(),
    }


def prefill_candidate_staging(
    *,
    target: str = "",
    status: str = "draft",
    limit: int = 30,
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
) -> dict[str, Any]:
    candidates = [
        item
        for item in load_candidate_queue(manifest_path, staging_path, reviews_path, dedupe=True)
        if item.get("status") in {"todo", "draft"}
        and (not target or item.get("target") == target)
        and item.get("preannotation", {}).get("status") == "ready"
    ][:limit]
    prefills: list[dict[str, Any]] = []
    for candidate in candidates:
        preannotation = candidate.get("preannotation", {})
        item = upsert_staging_annotation(
            {
                "id": candidate["id"],
                "target": candidate.get("target", ""),
                "annotation_type": candidate.get("annotation_type", ""),
                "status": status,
                "split": "train",
                "candidate": candidate,
                "annotation": preannotation.get("annotation", {}),
                "source": "auto_preannotation",
            },
            staging_path=staging_path,
        )
        prefills.append({"id": item["id"], "target": item.get("target", ""), "status": item.get("status", "")})
    return {
        "schema_version": 1,
        "status": "ready" if prefills else "empty",
        "prefilled_count": len(prefills),
        "prefills": prefills,
        "updated_at": utc_now(),
    }


def prefill_heatmap_staging(**kwargs: Any) -> dict[str, Any]:
    return prefill_candidate_staging(target="heatmap_tracker_labels", **kwargs)


def load_jobs(path: Path = DEFAULT_JOBS_PATH) -> dict[str, Any]:
    payload = read_json(path, {})
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {"schema_version": 1, "updated_at": "", "jobs": []}
    payload.setdefault("jobs", [])
    return payload


def upsert_job_record(job: dict[str, Any], path: Path = DEFAULT_JOBS_PATH) -> dict[str, Any]:
    if not job.get("id"):
        raise ValueError("job id is required")
    payload = load_jobs(path)
    jobs = [item for item in payload.get("jobs", []) if isinstance(item, dict) and item.get("id") != job["id"]]
    jobs.append(job)
    payload["jobs"] = sorted(jobs, key=lambda item: str(item.get("created_at", "")))[-100:]
    payload["updated_at"] = utc_now()
    write_json(path, payload)
    return job


def start_job_record(action_id: str, payload: dict[str, Any] | None = None, path: Path = DEFAULT_JOBS_PATH) -> dict[str, Any]:
    definition = action_definition(action_id)
    created_at = utc_now()
    job = {
        "id": safe_dataset_stem(f"job:{action_id}:{created_at}"),
        "action_id": action_id,
        "label": definition.label,
        "label_zh": definition.label_zh,
        "status": "running",
        "payload": payload or {},
        "created_at": created_at,
        "started_at": created_at,
        "completed_at": "",
        "result": {},
    }
    return upsert_job_record(job, path)


def finish_job_record(job_id: str, result: dict[str, Any], path: Path = DEFAULT_JOBS_PATH) -> dict[str, Any]:
    payload = load_jobs(path)
    jobs = [item for item in payload.get("jobs", []) if isinstance(item, dict)]
    job = next((item for item in jobs if item.get("id") == job_id), {"id": job_id, "created_at": utc_now()})
    job["status"] = str(result.get("status", "failed"))
    job["result"] = result
    job["completed_at"] = utc_now()
    return upsert_job_record(job, path)


def reconcile_running_jobs(path: Path = DEFAULT_JOBS_PATH) -> int:
    """Mark jobs still flagged ``running`` as interrupted.

    Jobs run in daemon threads, so a server restart leaves any in-flight job
    stranded at ``running`` forever. Call this once at startup: the threads are
    gone, so anything still ``running`` in the file died with the old process.
    Returns the number of records reconciled.
    """
    payload = load_jobs(path)
    jobs = [item for item in payload.get("jobs", []) if isinstance(item, dict)]
    reconciled = 0
    for job in jobs:
        if job.get("status") == "running":
            job["status"] = "interrupted"
            job["completed_at"] = utc_now()
            result = job.get("result")
            if not isinstance(result, dict):
                result = {}
            result["status"] = "interrupted"
            result.setdefault("error", "server restarted while this job was running")
            job["result"] = result
            reconciled += 1
    if reconciled:
        payload["jobs"] = jobs
        payload["updated_at"] = utc_now()
        write_json(path, payload)
    return reconciled


ActionCommandBuilder = Callable[[dict[str, Any], str], list[str]]


def command_refresh_training_candidates(payload: dict[str, Any], python: str) -> list[str]:
    return [python, "scripts/export_training_sample_candidates.py"]


def command_run_validation_suite(payload: dict[str, Any], python: str) -> list[str]:
    command = [python, "scripts/run_validation_suite.py"]
    if payload.get("run_analysis"):
        command.append("--run-analysis")
    return command


def command_intake_video(payload: dict[str, Any], python: str) -> list[str]:
    video = str(payload.get("video", "")).strip()
    if not video:
        raise ValueError("video is required")
    match_id = str(payload.get("match_id") or Path(video).stem)
    command = [
        python,
        "scripts/intake_samples.py",
        "--video",
        video,
        "--match-id",
        match_id,
        "--purpose",
        "validation",
        "--purpose",
        "analysis_candidate",
        "--write",
    ]
    if payload.get("scan_analysis_windows", True):
        command.append("--scan-analysis-windows")
    return command


def command_validate_training_datasets(payload: dict[str, Any], python: str) -> list[str]:
    return [python, "scripts/validate_model_training_datasets.py"]


def command_refresh_model_data_readiness(payload: dict[str, Any], python: str) -> list[str]:
    return [python, "scripts/report_model_data_readiness.py"]


def command_training_dry_run(payload: dict[str, Any], python: str) -> list[str]:
    target = str(payload.get("target", "")).strip()
    if not target:
        raise ValueError("target is required")
    return [python, "scripts/run_model_training_target.py", "--target", target]


def command_training_execute(payload: dict[str, Any], python: str) -> list[str]:
    if payload.get("confirm") != "execute_training":
        raise ValueError("confirm must be execute_training")
    target = str(payload.get("target", "")).strip()
    if not target:
        raise ValueError("target is required")
    return [python, "scripts/run_model_training_target.py", "--target", target, "--execute"]


def command_run_model_baseline(payload: dict[str, Any], python: str) -> list[str]:
    command = [python, "scripts/run_model_experiment_baseline.py"]
    if payload.get("run_validation_suite"):
        command.append("--run-validation-suite")
    return command


def command_promotion(payload: dict[str, Any], python: str, *, apply: bool = False) -> list[str]:
    model_id = str(payload.get("model_id", "")).strip()
    candidate = str(payload.get("candidate", "")).strip()
    if not model_id or not candidate:
        raise ValueError("model_id and candidate are required")
    command = [python, "scripts/promote_model_candidate.py", "--model-id", model_id, "--candidate", candidate]
    validation_report = str(payload.get("validation_report", "")).strip()
    if validation_report:
        command.extend(["--validation-report", validation_report])
    if apply:
        if payload.get("confirm") != "apply_promotion":
            raise ValueError("confirm must be apply_promotion")
        command.append("--apply")
    return command


ACTION_COMMAND_BUILDERS: dict[str, ActionCommandBuilder] = {
    "refresh_training_candidates": command_refresh_training_candidates,
    "run_validation_suite": command_run_validation_suite,
    "intake_video": command_intake_video,
    "validate_training_datasets": command_validate_training_datasets,
    "refresh_model_data_readiness": command_refresh_model_data_readiness,
    "training_dry_run": command_training_dry_run,
    "training_execute": command_training_execute,
    "run_model_baseline": command_run_model_baseline,
    "promotion_plan": lambda payload, python: command_promotion(payload, python, apply=False),
    "promotion_apply": lambda payload, python: command_promotion(payload, python, apply=True),
}


def command_for_action(action_id: str, payload: dict[str, Any] | None = None) -> list[str]:
    payload = payload or {}
    python = str(payload.get("python") or sys.executable)
    builder = ACTION_COMMAND_BUILDERS.get(action_id)
    if not builder:
        raise ValueError(f"unknown action: {action_id}")
    return builder(payload, python)


def append_action_run(record: dict[str, Any], path: Path = DEFAULT_ACTION_RUNS_PATH) -> dict[str, Any]:
    payload = read_json(path, {"schema_version": 1, "runs": []})
    if not isinstance(payload, dict):
        payload = {"schema_version": 1, "runs": []}
    payload.setdefault("runs", [])
    payload["runs"].append(record)
    payload["updated_at"] = utc_now()
    write_json(path, payload)
    return record


def run_workbench_action(
    action_id: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    command = command_for_action(action_id, payload)
    started = utc_now()
    completed = ""
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=int((payload or {}).get("timeout_seconds", timeout_seconds)),
            check=False,
        )
        completed = utc_now()
        record = {
            "id": f"{action_id}:{started}",
            "action_id": action_id,
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "command": command,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "started_at": started,
            "completed_at": completed,
        }
    except subprocess.TimeoutExpired as exc:
        completed = utc_now()
        record = {
            "id": f"{action_id}:{started}",
            "action_id": action_id,
            "status": "timeout",
            "returncode": None,
            "command": command,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "started_at": started,
            "completed_at": completed,
        }
    return append_action_run(record)


def append_automation_run(record: dict[str, Any], path: Path = DEFAULT_AUTOMATION_RUNS_PATH) -> dict[str, Any]:
    payload = read_json(path, {"schema_version": 1, "runs": []})
    if not isinstance(payload, dict):
        payload = {"schema_version": 1, "runs": []}
    payload.setdefault("runs", [])
    payload["runs"].append(record)
    payload["updated_at"] = utc_now()
    write_json(path, payload)
    return record


def run_automation_pipeline(
    *,
    include_long: bool = False,
    max_steps: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = build_automation_plan()
    executed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    started = utc_now()
    for step in plan.get("steps", []):
        if len(executed) >= max_steps:
            skipped.append({**step, "skip_reason": "max_steps reached"})
            continue
        if step.get("kind") == "human_gate":
            skipped.append({**step, "skip_reason": "human gate"})
            continue
        if step.get("kind") == "action":
            action_id = str(step.get("action_id", ""))
            definition = action_definition(action_id)
            if not definition.automation_safe:
                skipped.append({**step, "skip_reason": "not automation safe"})
                continue
            if definition.long_running and not include_long:
                skipped.append({**step, "skip_reason": "long-running step requires include_long"})
                continue
            if dry_run:
                executed.append({**step, "status": "planned"})
                continue
            result = run_workbench_action(action_id, step.get("payload", {}))
            executed.append({**step, "result": result, "status": result.get("status", "failed")})
            if result.get("status") not in {"passed", "ready"}:
                break
        elif step.get("kind") == "apply_staging_dry_run":
            if dry_run:
                executed.append({**step, "status": "planned"})
                continue
            result = apply_staging_annotations(dry_run=True)
            executed.append({**step, "result": result, "status": result.get("status", "failed")})
        else:
            skipped.append({**step, "skip_reason": "unknown step kind"})
    record = {
        "schema_version": 1,
        "status": "ready" if not executed and not skipped else "completed",
        "dry_run": dry_run,
        "include_long": include_long,
        "started_at": started,
        "completed_at": utc_now(),
        "executed_count": len(executed),
        "skipped_count": len(skipped),
        "executed": executed,
        "skipped": skipped,
    }
    return append_automation_run(record)


def safe_project_file(path: str | Path) -> Path:
    resolved = project_path(path).resolve()
    root = ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside project: {path}") from exc
    return resolved


def media_type_for_path(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
