from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data_registry import display_path, resolve_project_path


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


def format_coordinate(value: float) -> str:
    return f"{value:.6f}"


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    source_box: StageBox,
    *,
    x_field: str = "x",
    y_field: str = "y",
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
        normalized = normalize_point(x, y, source_box)
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


def write_normalized_csv(input_csv: Path, output_csv: Path, source_box: StageBox) -> dict[str, int]:
    rows, fieldnames = read_csv_rows(input_csv)
    normalized_rows, summary = normalize_rows(rows, source_box)
    output_fields = list(fieldnames)
    for field in ("stage_x", "stage_y", "stage_inside_roi"):
        if field not in output_fields:
            output_fields.append(field)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return summary


def summarize_points_csv(input_csv: Path, source_box: StageBox) -> dict[str, int]:
    rows, _ = read_csv_rows(input_csv)
    _, summary = normalize_rows(rows, source_box)
    return summary


def build_stage_coordinate_report(
    config: Mapping[str, Any],
    *,
    points_csv: Path | str | None = None,
    normalized_csv: Path | str | None = None,
) -> dict[str, Any]:
    try:
        source_box = stage_box_from_config(config)
        transform_errors: list[str] = []
    except (KeyError, TypeError, ValueError) as exc:
        source_box = None
        transform_errors = [str(exc)]

    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready" if not transform_errors else "needs_roi",
        "coordinate_space": map_view.get("coordinate_space", "") if isinstance(map_view, Mapping) else "",
        "target_coordinate_space": "stage_normalized_0_1",
        "transform": {
            "method": "roi_linear_normalization",
            "source_roi": source_box.as_dict() if source_box else {},
            "notes": "Maps current video-pixel map ROI to 0..1 coordinates. Homography can replace this when stage-map control points are available.",
        },
        "points": {"status": "not_requested"},
        "errors": transform_errors,
    }
    if source_box is None or points_csv is None:
        return report

    input_path = resolve_project_path(points_csv) or Path(points_csv).expanduser()
    points_info: dict[str, Any] = {"input": display_path(input_path), "status": "missing"}
    if input_path.exists():
        points_info["status"] = "ready"
        if normalized_csv is not None:
            output_path = Path(normalized_csv).expanduser()
            summary = write_normalized_csv(input_path, output_path, source_box)
            points_info["normalized_output"] = display_path(output_path)
        else:
            summary = summarize_points_csv(input_path, source_box)
        points_info["summary"] = summary
    report["points"] = points_info
    return report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    transform = report.get("transform", {})
    source_roi = transform.get("source_roi", {}) if isinstance(transform, Mapping) else {}
    points = report.get("points", {})
    summary = points.get("summary", {}) if isinstance(points, Mapping) else {}
    lines = [
        "# Stage Coordinate Normalization",
        "",
        f"- status: `{report.get('status')}`",
        f"- source: `{report.get('coordinate_space', '')}`",
        f"- target: `{report.get('target_coordinate_space', '')}`",
        f"- method: `{transform.get('method', '') if isinstance(transform, Mapping) else ''}`",
        f"- source_roi: `{json.dumps(source_roi, ensure_ascii=False)}`",
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
    errors = report.get("errors", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)
