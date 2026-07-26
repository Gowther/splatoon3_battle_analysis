from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.heatmap.stage_coordinates import (
    DEFAULT_REPROJECTION_TOLERANCE,
    StageBox,
    homography_from_control_points,
    normalize_point_homography,
    parse_control_point,
    reprojection_report,
    stage_box_from_config,
)


# A homography fitted to four clustered points reprojects those four points
# perfectly while mapping the rest of the ROI nowhere near the stage. Reprojection
# error alone cannot see that, so coverage and corner sanity are separate gates.
DEFAULT_MIN_COVERAGE = 0.15
DEFAULT_MAX_CORNER_EXCURSION = 0.35
DEFAULT_MAX_FRAME_DRIFT = 12.0


def convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) < 3:
        return list(unique)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += point[0] * following[1] - following[0] * point[1]
    return abs(total) / 2.0


def coverage_report(
    control_points: Sequence[Mapping[str, Any]],
    source_box: StageBox,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Fraction of the map ROI enclosed by the control point hull."""
    hull = convex_hull([(float(point["source_x"]), float(point["source_y"])) for point in control_points])
    roi_area = source_box.width * source_box.height
    area = polygon_area(hull)
    coverage = area / roi_area if roi_area > 0 else 0.0
    return {
        "status": "ready" if coverage >= min_coverage else "low_coverage",
        "coverage": coverage,
        "min_coverage": min_coverage,
        "hull_points": len(hull),
        "hull_area": area,
        "roi_area": roi_area,
    }


def corner_sanity_report(
    matrix: list[list[float]],
    source_box: StageBox,
    *,
    max_excursion: float = DEFAULT_MAX_CORNER_EXCURSION,
) -> dict[str, Any]:
    """Map the ROI corners through the homography; they should land near the 0..1 stage box."""
    corners = (
        ("top_left", source_box.x1, source_box.y1),
        ("top_right", source_box.x2, source_box.y1),
        ("bottom_right", source_box.x2, source_box.y2),
        ("bottom_left", source_box.x1, source_box.y2),
    )
    mapped: list[dict[str, Any]] = []
    worst = 0.0
    for name, x, y in corners:
        try:
            point = normalize_point_homography(x, y, matrix)
        except ValueError:
            return {"status": "invalid", "max_excursion": None, "limit": max_excursion, "corners": mapped}
        excursion = max(
            0.0,
            -point["stage_x"],
            point["stage_x"] - 1.0,
            -point["stage_y"],
            point["stage_y"] - 1.0,
        )
        worst = max(worst, excursion)
        mapped.append(
            {
                "name": name,
                "stage_x": point["stage_x"],
                "stage_y": point["stage_y"],
                "excursion": excursion,
            }
        )
    return {
        "status": "ready" if worst <= max_excursion else "corners_out_of_stage",
        "max_excursion": worst,
        "limit": max_excursion,
        "corners": mapped,
    }


def frame_drift_report(
    labeled_frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_drift: float = DEFAULT_MAX_FRAME_DRIFT,
) -> dict[str, Any]:
    """Compare the same named landmark across frames; large drift means the camera moved."""
    by_name: dict[str, list[tuple[str, float, float]]] = {}
    for frame_id, points in labeled_frames.items():
        for raw in points:
            point = parse_control_point(raw)
            name = str(point.get("name", "")).strip()
            if name:
                by_name.setdefault(name, []).append((frame_id, point["source_x"], point["source_y"]))

    shared = {name: entries for name, entries in by_name.items() if len(entries) > 1}
    if not shared:
        return {
            "status": "not_available",
            "max_drift": None,
            "limit": max_drift,
            "compared_landmarks": 0,
            "landmarks": [],
        }

    landmarks: list[dict[str, Any]] = []
    worst = 0.0
    for name, entries in sorted(shared.items()):
        xs = [entry[1] for entry in entries]
        ys = [entry[2] for entry in entries]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        drift = max(((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 for x, y in zip(xs, ys))
        worst = max(worst, drift)
        landmarks.append({"name": name, "frames": len(entries), "max_drift": drift})

    return {
        "status": "ready" if worst <= max_drift else "unstable",
        "max_drift": worst,
        "limit": max_drift,
        "compared_landmarks": len(landmarks),
        "landmarks": sorted(landmarks, key=lambda item: item["max_drift"], reverse=True),
    }


def build_control_point_quality_report(
    config: Mapping[str, Any],
    asset: Mapping[str, Any],
    *,
    labeled_frames: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    tolerance: float = DEFAULT_REPROJECTION_TOLERANCE,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_excursion: float = DEFAULT_MAX_CORNER_EXCURSION,
    max_drift: float = DEFAULT_MAX_FRAME_DRIFT,
) -> dict[str, Any]:
    source_box = stage_box_from_config(config)
    points = [parse_control_point(point) for point in asset.get("control_points", []) if isinstance(point, Mapping)]

    blockers: list[str] = []
    if asset.get("template"):
        blockers.append("template asset cannot be evaluated; replace the seed points first")
    if len(points) < 4:
        blockers.append(f"at least four control points are required, found {len(points)}")

    checks: dict[str, Any] = {}
    if not blockers:
        try:
            matrix = homography_from_control_points(points)
        except ValueError as exc:
            blockers.append(f"homography could not be solved: {exc}")
        else:
            checks["reprojection"] = reprojection_report(points, matrix, tolerance=tolerance)
            checks["coverage"] = coverage_report(points, source_box, min_coverage=min_coverage)
            checks["corners"] = corner_sanity_report(matrix, source_box, max_excursion=max_excursion)
    checks["frame_drift"] = frame_drift_report(labeled_frames or {}, max_drift=max_drift)

    failed = [name for name, check in checks.items() if check.get("status") not in {"ready", "not_available"}]
    status = "needs_control_points" if blockers else ("ready" if not failed else "needs_review")
    return {
        "schema_version": 1,
        "status": status,
        "stage_id": asset.get("stage_id", ""),
        "control_point_count": len(points),
        "source_roi": source_box.as_dict(),
        "checks": checks,
        "failed_checks": failed,
        "blockers": blockers,
    }


def render_quality_markdown(report: Mapping[str, Any]) -> str:
    checks = report.get("checks", {})
    reprojection = checks.get("reprojection", {})
    coverage = checks.get("coverage", {})
    corners = checks.get("corners", {})
    drift = checks.get("frame_drift", {})

    def optional(value: Any, spec: str = ".6f") -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "n/a"

    lines = [
        "# Stage Control Point Quality",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- stage_id: `{report.get('stage_id', '')}`",
        f"- control_point_count: {report.get('control_point_count', 0)}",
        f"- failed_checks: {', '.join(report.get('failed_checks', [])) or 'none'}",
        "",
        "## Checks",
        "",
        "| check | status | value | limit |",
        "| --- | --- | --- | --- |",
        f"| reprojection | `{reprojection.get('status', 'n/a')}` | {optional(reprojection.get('max_error'))} | {optional(reprojection.get('tolerance'), '.4f')} |",
        f"| coverage | `{coverage.get('status', 'n/a')}` | {optional(coverage.get('coverage'), '.4f')} | >= {optional(coverage.get('min_coverage'), '.4f')} |",
        f"| corners | `{corners.get('status', 'n/a')}` | {optional(corners.get('max_excursion'), '.4f')} | <= {optional(corners.get('limit'), '.4f')} |",
        f"| frame_drift | `{drift.get('status', 'n/a')}` | {optional(drift.get('max_drift'), '.2f')} | <= {optional(drift.get('limit'), '.2f')} |",
    ]
    if corners.get("corners"):
        lines.extend(["", "## ROI Corners In Stage Space", "", "| corner | stage_x | stage_y | excursion |", "| --- | --- | --- | --- |"])
        lines.extend(
            f"| {item['name']} | {item['stage_x']:.4f} | {item['stage_y']:.4f} | {item['excursion']:.4f} |"
            for item in corners["corners"]
        )
    if drift.get("landmarks"):
        lines.extend(["", "## Landmark Drift", "", "| landmark | frames | max_drift_px |", "| --- | --- | --- |"])
        lines.extend(
            f"| {item['name']} | {item['frames']} | {item['max_drift']:.2f} |" for item in drift["landmarks"]
        )
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in report["blockers"])
    lines.append("")
    return "\n".join(lines)
