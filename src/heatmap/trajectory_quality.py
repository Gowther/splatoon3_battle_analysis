from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from src.data_registry import display_path, resolve_project_path


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def average(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(mean(values), 4)


def route_image_count(path: Path | None) -> int:
    if path is None or not path.exists() or not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg"})


def count_by(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "") or "(empty)"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def compute_quality(
    player_tracks_path: Path | None,
    player_track_gaps_path: Path | None = None,
    player_routes_dir: Path | None = None,
    expected_teams: list[str] | None = None,
) -> dict[str, Any]:
    tracks = read_csv_rows(player_tracks_path)
    gaps = read_csv_rows(player_track_gaps_path)
    statuses = count_by(tracks, "track_status")
    teams = count_by(tracks, "team")
    players = count_by(tracks, "player_id")
    frames = {row.get("frame_index", "") for row in tracks if row.get("frame_index", "")}
    times = [value for value in (float_or_none(row.get("time")) for row in tracks) if value is not None]
    confidences = [value for value in (float_or_none(row.get("confidence")) for row in tracks) if value is not None]
    identity_confidences = [
        value for value in (float_or_none(row.get("identity_confidence")) for row in tracks) if value is not None
    ]
    step_distances = [value for value in (float_or_none(row.get("step_distance")) for row in tracks) if value is not None]
    track_rows = len(tracks)
    gap_rows = len(gaps)
    jump_reset_rows = statuses.get("jump_reset", 0)
    route_images = route_image_count(player_routes_dir)
    expected_teams = expected_teams or []
    missing_teams = [team for team in expected_teams if team not in teams]

    warnings: list[str] = []
    if track_rows == 0:
        warnings.append("player_tracks.csv has no rows")
    if missing_teams:
        warnings.append(f"missing expected teams: {', '.join(missing_teams)}")
    if players and route_images < len(players):
        warnings.append(f"route image count {route_images} is lower than player slot count {len(players)}")

    return {
        "player_tracks": display_path(player_tracks_path),
        "player_track_gaps": display_path(player_track_gaps_path),
        "player_routes_dir": display_path(player_routes_dir),
        "track_rows": track_rows,
        "gap_rows": gap_rows,
        "gap_ratio": ratio(gap_rows, track_rows),
        "jump_reset_rows": jump_reset_rows,
        "jump_reset_ratio": ratio(jump_reset_rows, track_rows),
        "matched_rows": statuses.get("matched", 0),
        "new_rows": statuses.get("new", 0),
        "unique_frames": len(frames),
        "time_start": min(times) if times else None,
        "time_end": max(times) if times else None,
        "teams": sorted(teams),
        "team_rows": teams,
        "player_slots": sorted(players),
        "player_slot_rows": players,
        "track_status_rows": statuses,
        "route_images": route_images,
        "average_confidence": average(confidences),
        "average_identity_confidence": average(identity_confidences),
        "average_step_distance": average(step_distances),
        "max_step_distance": round(max(step_distances), 4) if step_distances else None,
        "warnings": warnings,
    }


def evaluate_gates(metrics: dict[str, Any], gates: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    gates = gates or {}
    checks: dict[str, dict[str, Any]] = {}

    if "min_track_rows" in gates:
        expected = int(gates["min_track_rows"])
        actual = int(metrics["track_rows"])
        checks["min_track_rows"] = {"expected": f">= {expected}", "actual": actual, "ok": actual >= expected}
    if "max_gap_ratio" in gates:
        expected = float(gates["max_gap_ratio"])
        actual = float(metrics["gap_ratio"])
        checks["max_gap_ratio"] = {"expected": f"<= {expected}", "actual": actual, "ok": actual <= expected}
    if "max_jump_reset_ratio" in gates:
        expected = float(gates["max_jump_reset_ratio"])
        actual = float(metrics["jump_reset_ratio"])
        checks["max_jump_reset_ratio"] = {"expected": f"<= {expected}", "actual": actual, "ok": actual <= expected}
    if "min_route_images" in gates:
        expected = int(gates["min_route_images"])
        actual = int(metrics["route_images"])
        checks["min_route_images"] = {"expected": f">= {expected}", "actual": actual, "ok": actual >= expected}

    if metrics.get("warnings"):
        checks["trajectory_warnings"] = {
            "expected": "none",
            "actual": "; ".join(metrics["warnings"]),
            "ok": False,
        }
    return checks


def status_from_checks(checks: dict[str, dict[str, Any]]) -> str:
    return "failed" if any(not check["ok"] for check in checks.values()) else "passed"


def quality_from_registry_heatmap(heatmap: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics = compute_quality(
        resolve_project_path(heatmap.get("player_tracks")),
        resolve_project_path(heatmap.get("player_track_gaps")),
        resolve_project_path(heatmap.get("player_routes_dir")),
        expected_teams=list(heatmap.get("teams", [])),
    )
    checks = evaluate_gates(metrics, heatmap.get("quality_gates", {}))
    return metrics, checks


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown_report(
    path: Path,
    title: str,
    metrics: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- status: {status_from_checks(checks)}",
        f"- player tracks: `{metrics['player_tracks']}`",
        f"- player track gaps: `{metrics['player_track_gaps']}`",
        f"- player routes: `{metrics['player_routes_dir']}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "track_rows",
        "gap_rows",
        "gap_ratio",
        "jump_reset_rows",
        "jump_reset_ratio",
        "matched_rows",
        "new_rows",
        "unique_frames",
        "route_images",
        "average_confidence",
        "average_identity_confidence",
        "average_step_distance",
        "max_step_distance",
    ):
        lines.append(f"| {key} | {metrics.get(key)} |")
    lines.extend(["", "## Coverage", "", f"- teams: {', '.join(metrics['teams'])}", f"- player slots: {len(metrics['player_slots'])}"])
    lines.extend(["", "## Quality Gates", "", "| check | expected | actual | status |", "| --- | --- | --- | --- |"])
    if checks:
        for key, check in checks.items():
            status = "passed" if check["ok"] else "failed"
            lines.append(f"| {key} | {check['expected']} | {check['actual']} | {status} |")
    else:
        lines.append("| none |  |  | passed |")
    lines.extend(["", "## Warnings", ""])
    if metrics.get("warnings"):
        lines.extend(f"- {warning}" for warning in metrics["warnings"])
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
