from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from src.data_registry import display_path
from src.heatmap.render_heatmaps import adjust_heat_for_display, team_display_color


# A fixed canvas is the point of this renderer: stage coordinates are already
# normalized, so every match renders onto the same pixel grid and the outputs
# can be compared or stacked directly.
DEFAULT_CANVAS_SIZE = 900
DEFAULT_MARGIN = 40
BACKGROUND_COLOR = (24, 26, 30)
GRID_COLOR = (58, 62, 70)
BORDER_COLOR = (120, 126, 136)
LABEL_COLOR = (170, 176, 186)


def read_stage_rows(path: Path | str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_stage_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep rows that carry usable stage coordinates inside the 0..1 box."""
    points: list[dict[str, Any]] = []
    for row in rows:
        try:
            stage_x = float(row["stage_x"])
            stage_y = float(row["stage_y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0.0 <= stage_x <= 1.0 and 0.0 <= stage_y <= 1.0):
            continue
        try:
            confidence = float(row.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        points.append(
            {
                "stage_x": stage_x,
                "stage_y": stage_y,
                "team": str(row.get("team", "")),
                "track_slot": str(row.get("track_slot", "")),
                "player_id": str(row.get("player_id", "")),
                "time": str(row.get("time", "")),
                "track_status": str(row.get("track_status", "")),
                "step_distance": str(row.get("step_distance", "")),
                "confidence": confidence,
            }
        )
    return points


def stage_to_pixel(
    stage_x: float,
    stage_y: float,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> tuple[int, int]:
    span = canvas_size - 2 * margin
    return (
        int(round(margin + stage_x * span)),
        int(round(margin + stage_y * span)),
    )


def build_stage_canvas(
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
    grid_divisions: int = 10,
    label: str = "",
) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    span = canvas_size - 2 * margin
    for index in range(grid_divisions + 1):
        offset = int(round(margin + span * index / grid_divisions))
        cv2.line(canvas, (offset, margin), (offset, margin + span), GRID_COLOR, 1)
        cv2.line(canvas, (margin, offset), (margin + span, offset), GRID_COLOR, 1)
    cv2.rectangle(canvas, (margin, margin), (margin + span, margin + span), BORDER_COLOR, 2)
    for index in range(grid_divisions + 1):
        fraction = index / grid_divisions
        offset = int(round(margin + span * fraction))
        cv2.putText(canvas, f"{fraction:.1f}", (offset - 10, margin - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, LABEL_COLOR, 1)
        cv2.putText(canvas, f"{fraction:.1f}", (6, offset + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, LABEL_COLOR, 1)
    if label:
        cv2.putText(canvas, label, (margin, canvas_size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1)
    return canvas


def build_stage_heat(
    points: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> np.ndarray:
    """Accumulate a heat field on the fixed stage canvas.

    Radii are scaled from the configured video-pixel values into canvas pixels so
    a stage render looks like the source render rather than needing new tuning.
    """
    rendering = config.get("rendering", {}) if isinstance(config, Mapping) else {}
    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    roi = map_view.get("roi", {}) if isinstance(map_view, Mapping) else {}
    try:
        roi_width = float(roi["x2"]) - float(roi["x1"])
    except (KeyError, TypeError, ValueError):
        roi_width = 0.0
    span = canvas_size - 2 * margin
    scale = (span / roi_width) if roi_width > 0 else 1.0

    radius = max(1, int(round(float(rendering.get("heat_point_radius_px", 22)) * scale)))
    sigma = max(1.0, float(rendering.get("heat_blur_sigma_px", 34)) * scale)

    heat = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    for point in points:
        x, y = stage_to_pixel(point["stage_x"], point["stage_y"], canvas_size=canvas_size, margin=margin)
        cv2.circle(heat, (x, y), radius, max(0.05, float(point.get("confidence", 1.0))), -1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

    active = heat[heat > 0]
    if active.size == 0:
        return heat
    percentile = min(max(float(rendering.get("heat_scale_percentile", 99.0)), 50.0), 100.0)
    peak = float(np.percentile(active, percentile))
    if peak <= 0:
        return heat
    return np.clip(heat / peak, 0.0, 1.0)


def blend_stage_heat(
    canvas: np.ndarray,
    heat: np.ndarray,
    color: tuple[int, int, int],
    config: Mapping[str, Any],
    *,
    max_alpha: float | None = None,
) -> np.ndarray:
    rendering = config.get("rendering", {}) if isinstance(config, Mapping) else {}
    alpha_limit = float(rendering.get("heat_max_alpha", 0.58)) if max_alpha is None else max_alpha
    display = adjust_heat_for_display(heat, {"rendering": dict(rendering)})
    layer = np.zeros_like(canvas, dtype=np.float32)
    layer[:, :] = np.array(color, dtype=np.float32)
    alpha = (display * alpha_limit)[:, :, None]
    return np.clip(canvas.astype(np.float32) * (1.0 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def group_by_team(points: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[point.get("team", "")].append(point)
    return dict(grouped)


def render_stage_routes(
    points: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
    label: str = "",
) -> np.ndarray:
    rendering = config.get("rendering", {}) if isinstance(config, Mapping) else {}
    thickness = max(1, int(rendering.get("route_line_thickness_px", 2)))
    radius = max(1, int(rendering.get("route_point_radius_px", 4)))
    # Only connect consecutive matched positions that are close enough to be the
    # same player moving. Without this the image fills with long jump lines
    # between unrelated tracks and stops being readable.
    max_draw_step = float(rendering.get("route_max_draw_step_px", 120))

    canvas = build_stage_canvas(canvas_size=canvas_size, margin=margin, label=label)
    tracks: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        tracks[(point.get("team", ""), point.get("track_slot", ""))].append(point)

    for (team, _slot), track in sorted(tracks.items()):
        color = team_display_color(team, dict(config))
        ordered = sorted(track, key=lambda item: str(item.get("time", "")))
        previous: tuple[int, int] | None = None
        for point in ordered:
            current = stage_to_pixel(point["stage_x"], point["stage_y"], canvas_size=canvas_size, margin=margin)
            step = str(point.get("step_distance", ""))
            try:
                connectable = step != "" and float(step) <= max_draw_step
            except ValueError:
                connectable = False
            if previous is not None and str(point.get("track_status", "")) == "matched" and connectable:
                cv2.line(canvas, previous, current, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, current, radius, color, -1, cv2.LINE_AA)
            previous = current
    return canvas


def render_stage_heatmaps(
    stage_csv: Path | str,
    config: Mapping[str, Any],
    output_dir: Path | str,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Render fixed-canvas stage heatmaps and routes from a stage-coordinate CSV."""
    rows = read_stage_rows(stage_csv)
    points = parse_stage_points(rows)
    match_id = str(config.get("match", {}).get("id", "")) if isinstance(config.get("match"), Mapping) else ""
    target_dir = Path(output_dir)

    if not points:
        return {
            "status": "no_points",
            "match_id": match_id,
            "input": display_path(Path(stage_csv)),
            "input_rows": len(rows),
            "rendered": {},
            "canvas_size": canvas_size,
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, str] = {}

    combined = build_stage_canvas(canvas_size=canvas_size, margin=margin, label=f"{match_id} stage space")
    by_team = group_by_team(points)
    for team, team_points in sorted(by_team.items()):
        heat = build_stage_heat(team_points, config, canvas_size=canvas_size, margin=margin)
        color = team_display_color(team, dict(config))
        combined = blend_stage_heat(combined, heat, color, config)

        team_canvas = build_stage_canvas(canvas_size=canvas_size, margin=margin, label=f"{match_id} {team} stage space")
        team_path = target_dir / f"stage_heatmap_{team or 'unknown'}.png"
        cv2.imwrite(str(team_path), blend_stage_heat(team_canvas, heat, color, config))
        rendered[f"heatmap_{team or 'unknown'}"] = display_path(team_path)

    combined_path = target_dir / "stage_heatmap_combined.png"
    cv2.imwrite(str(combined_path), combined)
    rendered["heatmap_combined"] = display_path(combined_path)

    routes_path = target_dir / "stage_routes.png"
    cv2.imwrite(
        str(routes_path),
        render_stage_routes(points, config, canvas_size=canvas_size, margin=margin, label=f"{match_id} routes"),
    )
    rendered["routes"] = display_path(routes_path)

    return {
        "status": "ready",
        "match_id": match_id,
        "input": display_path(Path(stage_csv)),
        "input_rows": len(rows),
        "stage_points": len(points),
        "dropped_rows": len(rows) - len(points),
        "teams": sorted(by_team),
        "canvas_size": canvas_size,
        "output_dir": display_path(target_dir),
        "rendered": rendered,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage Space Rendering",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- match: `{report.get('match_id', '')}`",
        f"- input: `{report.get('input', '')}`",
        f"- input rows: {report.get('input_rows', 0)}",
        f"- rendered points: {report.get('stage_points', 0)}",
        f"- dropped rows: {report.get('dropped_rows', 0)}",
        f"- canvas: {report.get('canvas_size', 0)}x{report.get('canvas_size', 0)}",
        f"- teams: {', '.join(report.get('teams', [])) or 'none'}",
        "",
    ]
    rendered = report.get("rendered", {})
    if rendered:
        lines.extend(["## Images", "", "| name | path |", "| --- | --- |"])
        lines.extend(f"| {name} | `{path}` |" for name, path in sorted(rendered.items()))
    else:
        lines.append("No stage coordinates were available. Promote a control-point asset first.")
    lines.extend(
        [
            "",
            "Every match renders onto the same fixed canvas, so these images can be",
            "compared or stacked across matches on the same stage.",
            "",
        ]
    )
    return "\n".join(lines)
