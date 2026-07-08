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
from typing import Any

from src.core.paths import ROOT, project_path
from src.data_registry import load_registry


DEFAULT_STATE_DIR = ROOT / "outputs" / "active_learning_workbench"
DEFAULT_STAGING_PATH = DEFAULT_STATE_DIR / "staging_annotations.json"
DEFAULT_LLM_REVIEWS_PATH = DEFAULT_STATE_DIR / "llm_reviews.json"
DEFAULT_ACTION_RUNS_PATH = DEFAULT_STATE_DIR / "action_runs.json"
DEFAULT_CANDIDATE_MANIFEST = ROOT / "outputs" / "training_sample_candidates" / "manifest.json"
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
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def display_path(path: str | Path) -> str:
    resolved = project_path(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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


def candidate_annotation_type(target: str) -> str:
    if target == "heatmap_tracker_labels":
        return "heatmap_point"
    if target == "weapon_classifier_resnet18":
        return "classification"
    if target in {"count_ocr_yolo", "message_ocr_yolo"}:
        return "ocr_box_text"
    return "yolo_box"


def normalize_candidate(row: dict[str, str], *, target: str, row_index: int) -> dict[str, Any]:
    candidate_id = row.get("candidate_id") or f"{target}:{row.get('match_id', 'unknown')}:{row_index:04d}"
    frame_path = row.get("frame_path") or row.get("exported_frame") or row.get("preview_path") or ""
    return {
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


def load_candidate_queue(
    manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    staging_path: Path = DEFAULT_STAGING_PATH,
    reviews_path: Path = DEFAULT_LLM_REVIEWS_PATH,
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
    return candidates


def summarize_queue(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_target: dict[str, int] = {}
    for item in candidates:
        by_status[str(item.get("status", "todo"))] = by_status.get(str(item.get("status", "todo")), 0) + 1
        by_target[str(item.get("target", ""))] = by_target.get(str(item.get("target", "")), 0) + 1
    return {
        "status": "needs_human" if by_status.get("todo", 0) else "ready",
        "total": len(candidates),
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
        "actions": action_catalog(),
    }


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
) -> dict[str, Any]:
    staging = load_staging(staging_path)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
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
    report = {
        "schema_version": 1,
        "status": "ready" if not skipped else "needs_review",
        "dry_run": dry_run,
        "applied_count": len(applied),
        "skipped_count": len(skipped),
        "applied": applied,
        "skipped": skipped,
        "heatmap_labels_csv": display_path(heatmap_csv) if heatmap_rows else "",
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


def command_for_action(action_id: str, payload: dict[str, Any] | None = None) -> list[str]:
    payload = payload or {}
    python = str(payload.get("python") or sys.executable)
    if action_id == "refresh_training_candidates":
        return [python, "scripts/export_training_sample_candidates.py"]
    if action_id == "run_validation_suite":
        command = [python, "scripts/run_validation_suite.py"]
        if payload.get("run_analysis"):
            command.append("--run-analysis")
        return command
    if action_id == "intake_video":
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
    if action_id == "validate_training_datasets":
        return [python, "scripts/validate_model_training_datasets.py"]
    if action_id == "refresh_model_data_readiness":
        return [python, "scripts/report_model_data_readiness.py"]
    if action_id == "training_dry_run":
        target = str(payload.get("target", "")).strip()
        if not target:
            raise ValueError("target is required")
        return [python, "scripts/run_model_training_target.py", "--target", target]
    if action_id == "training_execute":
        if payload.get("confirm") != "execute_training":
            raise ValueError("confirm must be execute_training")
        target = str(payload.get("target", "")).strip()
        if not target:
            raise ValueError("target is required")
        return [python, "scripts/run_model_training_target.py", "--target", target, "--execute"]
    if action_id == "run_model_baseline":
        command = [python, "scripts/run_model_experiment_baseline.py"]
        if payload.get("run_validation_suite"):
            command.append("--run-validation-suite")
        return command
    if action_id in {"promotion_plan", "promotion_apply"}:
        model_id = str(payload.get("model_id", "")).strip()
        candidate = str(payload.get("candidate", "")).strip()
        if not model_id or not candidate:
            raise ValueError("model_id and candidate are required")
        command = [python, "scripts/promote_model_candidate.py", "--model-id", model_id, "--candidate", candidate]
        validation_report = str(payload.get("validation_report", "")).strip()
        if validation_report:
            command.extend(["--validation-report", validation_report])
        if action_id == "promotion_apply":
            if payload.get("confirm") != "apply_promotion":
                raise ValueError("confirm must be apply_promotion")
            command.append("--apply")
        return command
    raise ValueError(f"unknown action: {action_id}")


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
