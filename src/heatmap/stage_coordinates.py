from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.data_registry import display_path, resolve_project_path
from src.heatmap.stage_artifacts import write_stage_metadata


STAGE_OUTPUT_COLUMNS = (
    "stage_x",
    "stage_y",
    "stage_inside_roi",
)


@dataclass(frozen=True)
class StageBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageBox":
        return cls(float(value["x1"]), float(value["y1"]), float(value["x2"]), float(value["y2"]))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.width <= 0:
            errors.append("roi width must be positive")
        if self.height <= 0:
            errors.append("roi height must be positive")
        return errors

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def stage_box_from_config(config: Mapping[str, Any]) -> StageBox:
    map_view = config.get("map_view", {})
    if not isinstance(map_view, Mapping):
        raise ValueError("map_view must be a mapping")
    roi = map_view.get("roi")
    if not isinstance(roi, Mapping):
        raise ValueError("map_view.roi is required")
    box = StageBox.from_mapping(roi)
    errors = box.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return box


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalize_point(x: float, y: float, source_box: StageBox, *, clamp: bool = False) -> dict[str, Any]:
    errors = source_box.validate()
    if errors:
        raise ValueError("; ".join(errors))
    stage_x = (float(x) - source_box.x1) / source_box.width
    stage_y = (float(y) - source_box.y1) / source_box.height
    inside = 0.0 <= stage_x <= 1.0 and 0.0 <= stage_y <= 1.0
    if clamp:
        stage_x = clamp01(stage_x)
        stage_y = clamp01(stage_y)
    return {"stage_x": stage_x, "stage_y": stage_y, "inside_roi": inside}


def parse_control_point(value: Mapping[str, Any]) -> dict[str, Any]:
    if "source" in value and "target" in value:
        source = value["source"]
        target = value["target"]
        point: dict[str, Any] = {
            "source_x": float(source[0]),
            "source_y": float(source[1]),
            "stage_x": float(target[0]),
            "stage_y": float(target[1]),
        }
    else:
        point = {
            "source_x": float(value.get("source_x", value.get("video_x", value.get("x")))),
            "source_y": float(value.get("source_y", value.get("video_y", value.get("y")))),
            "stage_x": float(value.get("stage_x", value.get("target_x"))),
            "stage_y": float(value.get("stage_y", value.get("target_y"))),
        }
    name = value.get("name")
    if name:
        point["name"] = str(name)
    return point


def control_points_from_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    stage_coordinates = config.get("stage_coordinates", {}) if isinstance(config, Mapping) else {}
    candidates: Any = []
    if isinstance(map_view, Mapping):
        candidates = map_view.get("control_points") or candidates
        homography = map_view.get("homography", {})
        if isinstance(homography, Mapping):
            candidates = homography.get("control_points") or candidates
        stage_map = map_view.get("stage_map", {})
        if isinstance(stage_map, Mapping):
            candidates = stage_map.get("control_points") or candidates
    if not candidates and isinstance(stage_coordinates, Mapping):
        candidates = stage_coordinates.get("control_points") or []
    if not isinstance(candidates, list):
        return []
    points: list[dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, Mapping):
            points.append(parse_control_point(item))
    return points


def load_control_point_asset(path: Path | str) -> dict[str, Any]:
    target = resolve_project_path(path) or Path(path).expanduser()
    with target.open(encoding="utf-8") as f:
        payload = json.load(f)
    points = payload.get("control_points", [])
    if not isinstance(points, list):
        raise ValueError("control_points must be a list")
    return {
        "path": display_path(target),
        "stage_id": payload.get("stage_id", ""),
        "coordinate_space": payload.get("coordinate_space", ""),
        "target_coordinate_space": payload.get("target_coordinate_space", "stage_normalized_0_1"),
        "template": bool(payload.get("template", False)),
        "control_points": [parse_control_point(point) for point in points if isinstance(point, Mapping)],
        "notes": payload.get("notes", []),
    }


def merge_control_point_asset(config: Mapping[str, Any], asset: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(config)
    stage_coordinates = dict(output.get("stage_coordinates", {})) if isinstance(output.get("stage_coordinates"), Mapping) else {}
    stage_coordinates["control_points"] = list(asset.get("control_points", []))
    stage_coordinates["control_point_asset"] = asset.get("path", "")
    if asset.get("stage_id"):
        stage_coordinates["stage_id"] = asset.get("stage_id")
    output["stage_coordinates"] = stage_coordinates
    return output


DEFAULT_CONTROL_POINT_ASSET_DIR = "config/stage_control_points"


def discover_control_point_asset(
    config: Mapping[str, Any],
    *,
    asset_dir: Path | str = DEFAULT_CONTROL_POINT_ASSET_DIR,
) -> dict[str, Any] | None:
    """Find a promoted control-point asset for this config.

    Checked in order: an explicit stage_coordinates.control_point_asset path,
    then <match_id>.json and <stage_id>.json under the asset directory.
    Template assets are ignored.
    """
    stage_coordinates = config.get("stage_coordinates", {}) if isinstance(config, Mapping) else {}
    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    match = config.get("match", {}) if isinstance(config, Mapping) else {}
    resolved_dir = resolve_project_path(asset_dir) or Path(asset_dir)

    candidates: list[Path] = []
    explicit = stage_coordinates.get("control_point_asset", "") if isinstance(stage_coordinates, Mapping) else ""
    if explicit:
        resolved = resolve_project_path(explicit)
        if resolved is not None:
            candidates.append(resolved)
    for key_source in (match, stage_coordinates, map_view):
        if not isinstance(key_source, Mapping):
            continue
        identifier = str(key_source.get("id", "") or key_source.get("stage_id", "")).strip()
        if identifier:
            candidates.append(resolved_dir / f"{identifier}.json")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            asset = load_control_point_asset(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if asset.get("template"):
            continue
        return asset
    return None


def control_point_summary(control_points: list[dict[str, Any]], source_box: StageBox | None = None) -> dict[str, Any]:
    target_out_of_bounds: list[int] = []
    source_out_of_roi: list[int] = []
    duplicate_sources: list[str] = []
    seen_sources: set[tuple[float, float]] = set()
    for index, point in enumerate(control_points):
        stage_x = point["stage_x"]
        stage_y = point["stage_y"]
        if not (0.0 <= stage_x <= 1.0 and 0.0 <= stage_y <= 1.0):
            target_out_of_bounds.append(index)
        source = (point["source_x"], point["source_y"])
        if source in seen_sources:
            duplicate_sources.append(f"{point['source_x']},{point['source_y']}")
        seen_sources.add(source)
        if source_box is not None:
            normalized = normalize_point(point["source_x"], point["source_y"], source_box)
            if not normalized["inside_roi"]:
                source_out_of_roi.append(index)
    count = len(control_points)
    ready = count >= 4 and not target_out_of_bounds and not duplicate_sources
    return {
        "status": "ready" if ready else "needs_control_points",
        "count": count,
        "required_min": 4,
        "missing_count": max(0, 4 - count),
        "target_out_of_bounds_indices": target_out_of_bounds,
        "source_out_of_roi_indices": source_out_of_roi,
        "duplicate_sources": duplicate_sources,
    }


def coordinate_schema(method: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "coordinate_space": "stage_normalized_0_1",
        "method": method,
        "columns": [
            {"name": "stage_x", "type": "float", "range": [0.0, 1.0], "description": "normalized stage-map x coordinate"},
            {"name": "stage_y", "type": "float", "range": [0.0, 1.0], "description": "normalized stage-map y coordinate"},
            {"name": "stage_inside_roi", "type": "boolean", "description": "false when the source point maps outside the configured stage area"},
        ],
    }


def homography_from_control_points(control_points: list[dict[str, Any]]) -> list[list[float]]:
    if len(control_points) < 4:
        raise ValueError("at least four stage control points are required")
    rows: list[list[float]] = []
    for point in control_points:
        x = point["source_x"]
        y = point["source_y"]
        u = point["stage_x"]
        v = point["stage_y"]
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    matrix = vh[-1].reshape(3, 3)
    if abs(float(matrix[2, 2])) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return [[float(value) for value in row] for row in matrix]


def normalize_point_homography(x: float, y: float, matrix: list[list[float]], *, clamp: bool = False) -> dict[str, Any]:
    homography = np.asarray(matrix, dtype=float)
    target = homography @ np.asarray([float(x), float(y), 1.0], dtype=float)
    if abs(float(target[2])) <= 1e-12:
        raise ValueError("homography produced an invalid point")
    stage_x = float(target[0] / target[2])
    stage_y = float(target[1] / target[2])
    inside = 0.0 <= stage_x <= 1.0 and 0.0 <= stage_y <= 1.0
    if clamp:
        stage_x = clamp01(stage_x)
        stage_y = clamp01(stage_y)
    return {"stage_x": stage_x, "stage_y": stage_y, "inside_roi": inside}


def format_coordinate(value: float) -> str:
    return f"{value:.6f}"


DEFAULT_REPROJECTION_TOLERANCE = 0.02


def reprojection_report(
    control_points: list[dict[str, Any]],
    matrix: list[list[float]],
    *,
    tolerance: float = DEFAULT_REPROJECTION_TOLERANCE,
) -> dict[str, Any]:
    """Map each control point through the solved matrix and compare it to its declared target."""
    errors: list[dict[str, Any]] = []
    for index, point in enumerate(control_points):
        mapped = normalize_point_homography(point["source_x"], point["source_y"], matrix)
        distance = float(
            np.hypot(mapped["stage_x"] - point["stage_x"], mapped["stage_y"] - point["stage_y"])
        )
        errors.append(
            {
                "index": index,
                "name": point.get("name", f"point_{index + 1}"),
                "error": distance,
            }
        )
    if not errors:
        return {"status": "no_control_points", "tolerance": tolerance, "point_errors": []}
    worst = max(errors, key=lambda item: item["error"])
    max_error = float(worst["error"])
    mean_error = float(sum(item["error"] for item in errors) / len(errors))
    return {
        "status": "high_error" if max_error > tolerance else "ready",
        "tolerance": tolerance,
        "max_error": max_error,
        "mean_error": mean_error,
        "worst_point": worst["name"],
        "point_errors": errors,
    }


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    source_box: StageBox,
    *,
    x_field: str = "x",
    y_field: str = "y",
    homography_matrix: list[list[float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    summary = {"input_rows": 0, "normalized_rows": 0, "outside_roi_rows": 0, "invalid_rows": 0}
    for row in rows:
        summary["input_rows"] += 1
        item = dict(row)
        try:
            x = float(str(row.get(x_field, "")).strip())
            y = float(str(row.get(y_field, "")).strip())
        except ValueError:
            item.update({"stage_x": "", "stage_y": "", "stage_inside_roi": "False"})
            summary["invalid_rows"] += 1
            output.append(item)
            continue
        normalized = (
            normalize_point_homography(x, y, homography_matrix)
            if homography_matrix is not None
            else normalize_point(x, y, source_box)
        )
        item.update(
            {
                "stage_x": format_coordinate(float(normalized["stage_x"])),
                "stage_y": format_coordinate(float(normalized["stage_y"])),
                "stage_inside_roi": str(bool(normalized["inside_roi"])),
            }
        )
        summary["normalized_rows"] += 1
        if not normalized["inside_roi"]:
            summary["outside_roi_rows"] += 1
        output.append(item)
    return output, summary


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_normalized_csv(
    input_csv: Path,
    output_csv: Path,
    source_box: StageBox,
    *,
    homography_matrix: list[list[float]] | None = None,
) -> dict[str, int]:
    rows, fieldnames = read_csv_rows(input_csv)
    normalized_rows, summary = normalize_rows(rows, source_box, homography_matrix=homography_matrix)
    output_fields = list(fieldnames)
    for field in STAGE_OUTPUT_COLUMNS:
        if field not in output_fields:
            output_fields.append(field)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return summary


def summarize_points_csv(
    input_csv: Path,
    source_box: StageBox,
    *,
    homography_matrix: list[list[float]] | None = None,
) -> dict[str, int]:
    rows, _ = read_csv_rows(input_csv)
    _, summary = normalize_rows(rows, source_box, homography_matrix=homography_matrix)
    return summary


def build_stage_coordinate_report(
    config: Mapping[str, Any],
    *,
    points_csv: Path | str | None = None,
    normalized_csv: Path | str | None = None,
    control_point_asset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if control_point_asset is not None:
        config = merge_control_point_asset(config, control_point_asset)
    try:
        source_box = stage_box_from_config(config)
        transform_errors: list[str] = []
    except (KeyError, TypeError, ValueError) as exc:
        source_box = None
        transform_errors = [str(exc)]

    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    control_points = control_points_from_config(config)
    asset_is_template = bool(control_point_asset and control_point_asset.get("template"))
    control_summary = control_point_summary(control_points, source_box)
    homography_matrix: list[list[float]] | None = None
    homography_status = "template_only" if asset_is_template else "needs_control_points"
    reprojection: dict[str, Any] = {"status": "not_available"}
    if len(control_points) >= 4 and not asset_is_template:
        try:
            homography_matrix = homography_from_control_points(control_points)
            reprojection = reprojection_report(control_points, homography_matrix)
            homography_status = "ready" if reprojection["status"] == "ready" else reprojection["status"]
        except ValueError as exc:
            transform_errors.append(str(exc))
            homography_status = "invalid"
    method = "homography" if homography_matrix is not None else "roi_linear_normalization"
    quality_gate: dict[str, Any] = {"status": "not_available"}
    if source_box is not None and control_points and not asset_is_template:
        from src.heatmap.stage_quality import build_control_point_quality_report

        quality_gate = build_control_point_quality_report(
            config,
            {"control_points": control_points, "template": False},
            labeled_frames=(control_point_asset or {}).get("labeled_frames"),
        )
    rejected = bool(transform_errors) or quality_gate["status"] not in {"ready", "not_available"}
    quality = "rejected" if rejected else ("calibrated" if method == "homography" else "provisional")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "needs_roi" if source_box is None else ("needs_review" if rejected else "ready"),
        "quality": quality,
        "quality_gate": quality_gate,
        "coordinate_space": map_view.get("coordinate_space", "") if isinstance(map_view, Mapping) else "",
        "target_coordinate_space": "stage_normalized_0_1",
        "transform": {
            "method": method,
            "source_roi": source_box.as_dict() if source_box else {},
            "homography_status": homography_status,
            "control_point_count": len(control_points),
            "control_points": control_summary,
            "control_point_asset": dict(control_point_asset or {}),
            "matrix": homography_matrix or [],
            "reprojection": reprojection,
            "notes": "Uses stage-map homography when four or more control points are available; otherwise maps current video-pixel map ROI to 0..1 coordinates.",
        },
        "output_schema": coordinate_schema(method),
        "points": {"status": "not_requested"},
        "errors": transform_errors,
    }
    metadata = {
        "match_id": config.get("match", {}).get("id", ""),
        "stage_id": config.get("stage_coordinates", {}).get("stage_id", ""),
        "status": report["status"] if rejected else "pending",
        "method": method,
        "quality": quality,
        "quality_gate": quality_gate,
    }
    if normalized_csv is not None:
        write_stage_metadata(normalized_csv, metadata)
    if source_box is None or points_csv is None:
        return report

    input_path = resolve_project_path(points_csv) or Path(points_csv).expanduser()
    points_info: dict[str, Any] = {"input": display_path(input_path), "status": "missing"}
    if input_path.exists():
        points_info["status"] = "ready"
        if normalized_csv is not None:
            output_path = resolve_project_path(normalized_csv) or Path(normalized_csv).expanduser()
            if report["status"] == "ready":
                summary = write_normalized_csv(input_path, output_path, source_box, homography_matrix=homography_matrix)
                points_info["normalized_output"] = display_path(output_path)
            else:
                summary = {}
                points_info["status"] = "blocked"
            metadata_path = write_stage_metadata(
                output_path,
                {**metadata, "status": report["status"]},
            )
            points_info["metadata"] = display_path(metadata_path)
        else:
            summary = summarize_points_csv(input_path, source_box, homography_matrix=homography_matrix)
        points_info["summary"] = summary
    report["points"] = points_info
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def roi_corner_control_points(source_box: StageBox) -> list[dict[str, Any]]:
    """Seed points that reproduce the current ROI linear mapping, for a reviewer to correct."""
    corners = (
        ("roi_top_left", source_box.x1, source_box.y1, 0.0, 0.0),
        ("roi_top_right", source_box.x2, source_box.y1, 1.0, 0.0),
        ("roi_bottom_right", source_box.x2, source_box.y2, 1.0, 1.0),
        ("roi_bottom_left", source_box.x1, source_box.y2, 0.0, 1.0),
    )
    return [
        {"name": name, "source": [source_x, source_y], "target": [stage_x, stage_y]}
        for name, source_x, source_y, stage_x, stage_y in corners
    ]


def build_control_point_asset(
    stage_id: str,
    control_points: list[Mapping[str, Any]],
    *,
    template: bool = False,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(control_points):
        parsed = parse_control_point(raw)
        points.append(
            {
                "name": parsed.get("name", f"point_{index + 1}"),
                "source": [parsed["source_x"], parsed["source_y"]],
                "target": [parsed["stage_x"], parsed["stage_y"]],
            }
        )
    return {
        "schema_version": 1,
        "template": template,
        "stage_id": stage_id,
        "coordinate_space": "video_pixels",
        "target_coordinate_space": "stage_normalized_0_1",
        "control_points": points,
        "notes": notes or [],
    }


def validate_control_point_asset(
    asset: Mapping[str, Any],
    *,
    source_box: StageBox | None = None,
    tolerance: float = DEFAULT_REPROJECTION_TOLERANCE,
) -> dict[str, Any]:
    """Check an asset can drive a homography, and report its reprojection error."""
    points = [parse_control_point(point) for point in asset.get("control_points", []) if isinstance(point, Mapping)]
    summary = control_point_summary(points, source_box)
    errors: list[str] = []
    if asset.get("template"):
        errors.append("template assets are rejected; set template to false after replacing the example points")
    if summary["count"] < 4:
        errors.append(f"at least four control points are required, found {summary['count']}")
    if summary["target_out_of_bounds_indices"]:
        errors.append(f"target coordinates outside 0..1 at indices {summary['target_out_of_bounds_indices']}")
    if summary["duplicate_sources"]:
        errors.append(f"duplicate source points: {summary['duplicate_sources']}")

    reprojection: dict[str, Any] = {"status": "not_available"}
    if not errors:
        try:
            matrix = homography_from_control_points(points)
            reprojection = reprojection_report(points, matrix, tolerance=tolerance)
            if reprojection["status"] == "high_error":
                errors.append(
                    f"reprojection error {reprojection['max_error']:.6f} exceeds tolerance {tolerance} "
                    f"at {reprojection['worst_point']}"
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"homography could not be solved: {exc}")

    return {
        "status": "ready" if not errors else "needs_control_points",
        "stage_id": asset.get("stage_id", ""),
        "control_points": summary,
        "reprojection": reprojection,
        "errors": errors,
    }


def render_control_point_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("control_points", {})
    reprojection = report.get("reprojection", {})
    lines = [
        "# Stage Control Points",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- stage_id: `{report.get('stage_id', '')}`",
        f"- control_point_count: {summary.get('count', 0) if isinstance(summary, Mapping) else 0}",
        f"- reprojection_status: `{reprojection.get('status', '') if isinstance(reprojection, Mapping) else ''}`",
    ]
    if isinstance(reprojection, Mapping) and reprojection.get("point_errors"):
        lines.extend(
            [
                f"- max_error: {reprojection.get('max_error', 0):.6f}",
                f"- mean_error: {reprojection.get('mean_error', 0):.6f}",
                f"- worst_point: `{reprojection.get('worst_point', '')}`",
                "",
                "## Point Errors",
                "",
                "| name | error |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| {item.get('name', '')} | {item.get('error', 0):.6f} |"
            for item in reprojection.get("point_errors", [])
        )
    errors = report.get("errors", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    transform = report.get("transform", {})
    source_roi = transform.get("source_roi", {}) if isinstance(transform, Mapping) else {}
    control_points = transform.get("control_points", {}) if isinstance(transform, Mapping) else {}
    output_schema = report.get("output_schema", {}) if isinstance(report.get("output_schema", {}), Mapping) else {}
    points = report.get("points", {})
    summary = points.get("summary", {}) if isinstance(points, Mapping) else {}
    lines = [
        "# Stage Coordinate Normalization",
        "",
        f"- status: `{report.get('status')}`",
        f"- source: `{report.get('coordinate_space', '')}`",
        f"- target: `{report.get('target_coordinate_space', '')}`",
        f"- method: `{transform.get('method', '') if isinstance(transform, Mapping) else ''}`",
        f"- homography_status: `{transform.get('homography_status', '') if isinstance(transform, Mapping) else ''}`",
        f"- control_point_count: {transform.get('control_point_count', 0) if isinstance(transform, Mapping) else 0}",
        f"- control_point_status: `{control_points.get('status', '') if isinstance(control_points, Mapping) else ''}`",
        f"- source_roi: `{json.dumps(source_roi, ensure_ascii=False)}`",
        f"- output_columns: {', '.join(column.get('name', '') for column in output_schema.get('columns', []))}",
        "",
        "## Point Summary",
        "",
        f"- status: `{points.get('status', '') if isinstance(points, Mapping) else ''}`",
        f"- input_rows: {summary.get('input_rows', 0)}",
        f"- normalized_rows: {summary.get('normalized_rows', 0)}",
        f"- outside_roi_rows: {summary.get('outside_roi_rows', 0)}",
        f"- invalid_rows: {summary.get('invalid_rows', 0)}",
    ]
    normalized_output = points.get("normalized_output") if isinstance(points, Mapping) else ""
    if normalized_output:
        lines.append(f"- normalized_output: `{normalized_output}`")
    reprojection = transform.get("reprojection", {}) if isinstance(transform, Mapping) else {}
    if isinstance(reprojection, Mapping) and reprojection.get("point_errors"):
        lines.extend(
            [
                "",
                "## Reprojection",
                "",
                f"- status: `{reprojection.get('status', '')}`",
                f"- tolerance: {reprojection.get('tolerance', 0)}",
                f"- max_error: {reprojection.get('max_error', 0):.6f}",
                f"- mean_error: {reprojection.get('mean_error', 0):.6f}",
                f"- worst_point: `{reprojection.get('worst_point', '')}`",
            ]
        )
    errors = report.get("errors", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)
