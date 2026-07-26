from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.active_learning_workbench import safe_project_file, utc_now
from src.core.paths import ROOT, project_path
from src.data_registry import display_path
from src.heatmap.stage_coordinates import (
    build_control_point_asset,
    validate_control_point_asset,
)


DEFAULT_REFERENCE_ROOT = ROOT / "outputs" / "stage_reference"
DEFAULT_ASSET_DIR = ROOT / "config" / "stage_control_points"
MIN_CONTROL_POINTS = 4


def load_manifest(package_dir: Path) -> dict[str, Any] | None:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_draft(package_dir: Path) -> dict[str, Any]:
    draft_path = package_dir / "control_points_draft.json"
    if not draft_path.is_file():
        return {}
    try:
        payload = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def promoted_asset_path(stage_id: str) -> Path:
    return DEFAULT_ASSET_DIR / f"{stage_id}.json"


def describe_package(package_dir: Path) -> dict[str, Any] | None:
    manifest = load_manifest(package_dir)
    if manifest is None:
        return None
    draft = load_draft(package_dir)
    stage_id = str(manifest.get("stage_id", package_dir.name))
    promoted = promoted_asset_path(stage_id)
    control_points = draft.get("control_points", []) if isinstance(draft.get("control_points"), list) else []
    return {
        "stage_id": stage_id,
        "package_dir": display_path(package_dir),
        "config": manifest.get("config", ""),
        "source_roi": manifest.get("source_roi", {}),
        "grid_divisions": manifest.get("grid_divisions", 10),
        "frames": [
            frame
            for frame in manifest.get("frames", [])
            if isinstance(frame, Mapping) and frame.get("status") == "exported"
        ],
        "draft_path": display_path(package_dir / "control_points_draft.json"),
        "draft_template": bool(draft.get("template", True)),
        "control_point_count": len(control_points),
        "control_points": control_points,
        "promoted": promoted.is_file(),
        "promoted_path": display_path(promoted) if promoted.is_file() else "",
    }


def build_stage_labeling_state(reference_root: Path | str = DEFAULT_REFERENCE_ROOT) -> dict[str, Any]:
    root = project_path(reference_root)
    packages: list[dict[str, Any]] = []
    if root.is_dir():
        for package_dir in sorted(root.iterdir()):
            if not package_dir.is_dir():
                continue
            described = describe_package(package_dir)
            if described is not None:
                packages.append(described)
    return {
        "generated_at": utc_now(),
        "reference_root": display_path(root),
        "packages": packages,
        "package_count": len(packages),
        "min_control_points": MIN_CONTROL_POINTS,
    }


def normalize_labeled_points(raw_points: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_points, list):
        raise ValueError("points must be a list")
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise ValueError(f"point {index + 1} must be an object")
        try:
            source_x = float(raw["source_x"])
            source_y = float(raw["source_y"])
            stage_x = float(raw["stage_x"])
            stage_y = float(raw["stage_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"point {index + 1} needs numeric source_x/source_y/stage_x/stage_y") from exc
        name = str(raw.get("name", "")).strip() or f"point_{index + 1}"
        points.append(
            {
                "name": name,
                "source": [source_x, source_y],
                "target": [stage_x, stage_y],
            }
        )
    return points


def save_stage_labels(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Save labeled control points into the package draft and validate them."""
    package_value = str(payload.get("package_dir", "")).strip()
    if not package_value:
        raise ValueError("package_dir is required")
    package_dir = safe_project_file(package_value)
    manifest = load_manifest(package_dir)
    if manifest is None:
        raise ValueError(f"not a stage reference package: {package_value}")

    stage_id = str(payload.get("stage_id", "") or manifest.get("stage_id", package_dir.name)).strip()
    points = normalize_labeled_points(payload.get("points"))
    keep_template = len(points) < MIN_CONTROL_POINTS

    asset = build_control_point_asset(
        stage_id,
        points,
        template=keep_template,
        notes=[
            f"Labeled in the stage labeling workbench at {utc_now()}.",
            "Sources are video pixels read from the grid reference frame; targets are stage-normalized 0..1.",
        ],
    )
    report = validate_control_point_asset(asset)

    draft_path = package_dir / "control_points_draft.json"
    draft_path.write_text(json.dumps(asset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "saved": True,
        "draft_path": display_path(draft_path),
        "stage_id": stage_id,
        "template": keep_template,
        "control_point_count": len(points),
        "validation": report,
    }


def promote_stage_labels(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a validated draft into config/stage_control_points/<stage_id>.json."""
    package_value = str(payload.get("package_dir", "")).strip()
    if not package_value:
        raise ValueError("package_dir is required")
    package_dir = safe_project_file(package_value)
    draft = load_draft(package_dir)
    if not draft:
        raise ValueError("draft asset not found or unreadable")

    stage_id = str(draft.get("stage_id", "")).strip()
    if not stage_id:
        raise ValueError("draft has no stage_id")

    report = validate_control_point_asset(draft)
    if report["status"] != "ready":
        return {
            "promoted": False,
            "stage_id": stage_id,
            "validation": report,
        }

    target = promoted_asset_path(stage_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = load_manifest(package_dir) or {}
    config_path = manifest.get("config", "src/heatmap/config_match9.yaml")
    return {
        "promoted": True,
        "stage_id": stage_id,
        "asset_path": display_path(target),
        "validation": report,
        "next_step": (
            f"python scripts/report_stage_coordinates.py --config {config_path} "
            f"--control-points {display_path(target)}"
        ),
    }
