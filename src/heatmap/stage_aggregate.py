from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.data_registry import display_path
from src.heatmap.render_stage_space import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_MARGIN,
    blend_stage_heat,
    build_stage_canvas,
    build_stage_heat,
    group_by_team,
    parse_stage_points,
    read_stage_rows,
)
from src.heatmap.render_heatmaps import adjust_heat_for_display, team_display_color
from src.heatmap.stage_registry import stage_entry


# BGR. The left match reads warm, the right match reads cool.
LEFT_ONLY_COLOR = (80, 90, 240)
RIGHT_ONLY_COLOR = (240, 170, 60)
LABEL_COLOR = (170, 176, 186)


def load_match_points(stage_csv: Path | str) -> dict[str, Any]:
    rows = read_stage_rows(stage_csv)
    points = parse_stage_points(rows)
    return {
        "input": display_path(Path(stage_csv)),
        "input_rows": len(rows),
        "points": points,
        "status": "ready" if points else "no_points",
    }


def collect_stage_matches(
    stage_id: str,
    registry: Mapping[str, Any],
    stage_csv_paths: Mapping[str, Path | str],
) -> dict[str, Any]:
    """Gather stage-coordinate points for every match registered under one stage."""
    entry = stage_entry(registry, stage_id)
    if entry is None:
        return {"status": "unknown_stage", "stage_id": stage_id, "matches": [], "missing": []}

    matches: list[dict[str, Any]] = []
    missing: list[str] = []
    for match_id in entry.get("matches", []):
        csv_path = stage_csv_paths.get(match_id)
        if csv_path is None:
            missing.append(match_id)
            continue
        loaded = load_match_points(csv_path)
        if loaded["status"] != "ready":
            missing.append(match_id)
            continue
        matches.append({"match_id": match_id, **loaded})

    return {
        "status": "ready" if matches else "no_data",
        "stage_id": stage_id,
        "matches": matches,
        "missing": missing,
    }


def aggregate_heat(
    matches: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    team: str | None = None,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> np.ndarray:
    """Average per-match heat so a long match cannot outvote a short one."""
    fields: list[np.ndarray] = []
    for match in matches:
        points = match["points"]
        if team is not None:
            points = [point for point in points if point.get("team") == team]
        if not points:
            continue
        fields.append(build_stage_heat(points, config, canvas_size=canvas_size, margin=margin))
    if not fields:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32)
    return np.clip(np.mean(np.stack(fields, axis=0), axis=0), 0.0, 1.0)


def render_occupancy(
    matches: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    stage_id: str,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> np.ndarray:
    label = f"{stage_id} occupancy over {len(matches)} matches"
    canvas = build_stage_canvas(canvas_size=canvas_size, margin=margin, label=label)
    teams = sorted({point.get("team", "") for match in matches for point in match["points"]})
    for team in teams:
        heat = aggregate_heat(matches, config, team=team, canvas_size=canvas_size, margin=margin)
        if float(np.max(heat)) <= 0:
            continue
        canvas = blend_stage_heat(canvas, heat, team_display_color(team, dict(config)), config)
    return canvas


def render_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    team: str | None = None,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> np.ndarray:
    """Where does one match hold ground the other does not?

    Only meaningful because both sides are already in the same normalized space.
    """
    left_heat = aggregate_heat([left], config, team=team, canvas_size=canvas_size, margin=margin)
    right_heat = aggregate_heat([right], config, team=team, canvas_size=canvas_size, margin=margin)
    delta = left_heat - right_heat

    label = f"{left['match_id']} vs {right['match_id']}"
    canvas = build_stage_canvas(canvas_size=canvas_size, margin=margin, label=label)
    rendering = {"rendering": dict(config.get("rendering", {}))}
    for signed, color in ((np.clip(delta, 0.0, 1.0), LEFT_ONLY_COLOR), (np.clip(-delta, 0.0, 1.0), RIGHT_ONLY_COLOR)):
        if float(np.max(signed)) <= 0:
            continue
        display = adjust_heat_for_display(signed, rendering)
        layer = np.zeros_like(canvas, dtype=np.float32)
        layer[:, :] = np.array(color, dtype=np.float32)
        alpha = (display * float(config.get("rendering", {}).get("heat_max_alpha", 0.58)))[:, :, None]
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - alpha) + layer * alpha, 0, 255).astype(np.uint8)
    # Legend sits inside the plot area, below the title band and clear of the
    # axis tick labels on every edge.
    legend_x = margin + 12
    legend_y = margin + 22
    cv2.putText(canvas, f"{left['match_id']} only", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, LEFT_ONLY_COLOR, 1)
    cv2.putText(
        canvas,
        f"{right['match_id']} only",
        (legend_x, legend_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        RIGHT_ONLY_COLOR,
        1,
    )
    return canvas


def render_side_by_side(
    matches: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
    max_columns: int = 4,
) -> np.ndarray:
    """One panel per match on the shared canvas, so panels are directly comparable."""
    panels: list[np.ndarray] = []
    for match in matches[:max_columns]:
        panel = build_stage_canvas(canvas_size=canvas_size, margin=margin, label=match["match_id"])
        for team, points in sorted(group_by_team(match["points"]).items()):
            heat = build_stage_heat(points, config, canvas_size=canvas_size, margin=margin)
            panel = blend_stage_heat(panel, heat, team_display_color(team, dict(config)), config)
        panels.append(panel)
    if not panels:
        return build_stage_canvas(canvas_size=canvas_size, margin=margin, label="no matches")
    return np.hstack(panels)


def build_stage_aggregate(
    stage_id: str,
    registry: Mapping[str, Any],
    stage_csv_paths: Mapping[str, Path | str],
    config: Mapping[str, Any],
    output_dir: Path | str,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    margin: int = DEFAULT_MARGIN,
) -> dict[str, Any]:
    collected = collect_stage_matches(stage_id, registry, stage_csv_paths)
    base = {
        "schema_version": 1,
        "stage_id": stage_id,
        "matches": [match["match_id"] for match in collected["matches"]],
        "match_count": len(collected["matches"]),
        "missing_matches": collected["missing"],
        "canvas_size": canvas_size,
        "rendered": {},
    }
    if collected["status"] != "ready":
        return {**base, "status": collected["status"]}

    matches = collected["matches"]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, str] = {}

    occupancy_path = target_dir / f"{stage_id}_occupancy.png"
    cv2.imwrite(
        str(occupancy_path),
        render_occupancy(matches, config, stage_id=stage_id, canvas_size=canvas_size, margin=margin),
    )
    rendered["occupancy"] = display_path(occupancy_path)

    if len(matches) > 1:
        panel_path = target_dir / f"{stage_id}_side_by_side.png"
        cv2.imwrite(
            str(panel_path),
            render_side_by_side(matches, config, canvas_size=canvas_size, margin=margin),
        )
        rendered["side_by_side"] = display_path(panel_path)

        diff_path = target_dir / f"{stage_id}_difference.png"
        cv2.imwrite(
            str(diff_path),
            render_difference(matches[0], matches[1], config, canvas_size=canvas_size, margin=margin),
        )
        rendered["difference"] = display_path(diff_path)

    return {
        **base,
        "status": "ready" if len(matches) > 1 else "single_match",
        "total_points": sum(len(match["points"]) for match in matches),
        "per_match_points": {match["match_id"]: len(match["points"]) for match in matches},
        "output_dir": display_path(target_dir),
        "rendered": rendered,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage Aggregate",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- stage: `{report.get('stage_id', '')}`",
        f"- matches: {', '.join(report.get('matches', [])) or 'none'}",
        f"- match count: {report.get('match_count', 0)}",
        f"- total points: {report.get('total_points', 0)}",
        f"- missing matches: {', '.join(report.get('missing_matches', [])) or 'none'}",
        "",
    ]
    per_match = report.get("per_match_points", {})
    if per_match:
        lines.extend(["## Points Per Match", "", "| match | stage points |", "| --- | --- |"])
        lines.extend(f"| {match} | {count} |" for match, count in sorted(per_match.items()))
        lines.append("")
    rendered = report.get("rendered", {})
    if rendered:
        lines.extend(["## Images", "", "| name | path |", "| --- | --- |"])
        lines.extend(f"| {name} | `{path}` |" for name, path in sorted(rendered.items()))
    if report.get("status") == "single_match":
        lines.extend(
            [
                "",
                "Only one match on this stage has stage coordinates, so comparison and",
                "difference images were skipped. Label a second match on the same stage",
                "to make the aggregate meaningful.",
            ]
        )
    elif report.get("status") in {"no_data", "unknown_stage"}:
        lines.extend(
            [
                "",
                "No stage coordinates were found for this stage. Promote a control-point",
                "asset and re-run the heatmap pipeline for at least one registered match.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
