from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.csv_contracts import DEATH_POSITION_CSV_CONTRACT
from src.death_events import (
    DEFAULT_ALIVE_STATE_IDS,
    DEFAULT_DEAD_STATE_IDS,
    DeathEvent,
    build_death_event_report,
    extract_death_events,
    read_csv_rows,
    write_event_csv,
)
from src.heatmap.extract_frames import resolve_path
from src.heatmap.config_loader import load_config
from src.heatmap.render_heatmaps import team_display_color
from src.heatmap.render_stage_space import DEFAULT_CANVAS_SIZE, DEFAULT_MARGIN, stage_to_pixel


POSITION_FIELDS = list(DEATH_POSITION_CSV_CONTRACT.fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract deaths and locate them on overhead-map player tracks.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--events", help="Optional external event CSV instead of extracting UI-state deaths.")
    return parser.parse_args()


def read_rows(path: Path | str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path | str, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _team_slot(global_slot: int | None) -> int | None:
    if global_slot is None or global_slot < 1:
        return None
    return ((global_slot - 1) % 4) + 1


def _event_value(event: DeathEvent | Mapping[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(event, DeathEvent):
        return getattr(event, key, default)
    return event.get(key, default)


def _track_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        team = str(raw.get("team", "")).strip()
        slot = str(raw.get("track_slot", "")).strip()
        time_value = _float(raw.get("time"))
        x = _float(raw.get("x"))
        y = _float(raw.get("y"))
        if not team or not slot or time_value is None or x is None or y is None:
            continue
        row = dict(raw)
        row["_time"] = time_value
        row["_x"] = x
        row["_y"] = y
        row["_tracking_confidence"] = _float(raw.get("tracking_confidence")) or _float(raw.get("confidence")) or 0.0
        grouped[f"{team}:{slot}"].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: float(item["_time"]))
    return dict(grouped)


def _candidate_for_track(
    rows: Sequence[Mapping[str, Any]],
    event_time: float,
    *,
    pre_window: float,
    post_window: float,
    expected_interval: float,
) -> dict[str, Any] | None:
    before = [row for row in rows if event_time - pre_window <= float(row["_time"]) <= event_time]
    if not before:
        return None
    before_row = max(before, key=lambda row: float(row["_time"]))
    before_delta = event_time - float(before_row["_time"])
    after = [row for row in rows if event_time <= float(row["_time"]) <= event_time + post_window]
    after_row = min(after, key=lambda row: float(row["_time"])) if after else None
    after_delta = None if after_row is None else float(after_row["_time"]) - event_time

    # A track that vanishes immediately after the event is a stronger victim
    # candidate than one that keeps producing points through the event.
    score = max(0.0, 1.0 - before_delta / max(pre_window, 1e-6))
    if after_row is None:
        score += 0.35
    elif after_delta is not None and after_delta > expected_interval * 1.75:
        score += 0.20
    else:
        score -= 0.15
    score *= 0.75 + 0.25 * float(before_row.get("_tracking_confidence", 0.0))
    return {
        "track_slot": str(before_row.get("track_slot", "")),
        "player_id": str(before_row.get("player_id", "")),
        "before": before_row,
        "before_delta": before_delta,
        "after": after_row,
        "after_delta": after_delta,
        "score": score,
    }


def _assign_unique_candidates(event_candidates: Sequence[Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any] | None]:
    """Choose the highest-scoring one-to-one track assignment for simultaneous deaths."""
    track_slots = sorted(
        {str(candidate.get("track_slot", "")) for candidates in event_candidates for candidate in candidates}
    )
    if not event_candidates or not track_slots:
        return [None for _ in event_candidates]

    slot_index = {slot: index for index, slot in enumerate(track_slots)}
    # One zero-score dummy per event lets a row remain unassigned when it has no
    # valid track without consuming a real player slot.
    scores = np.full((len(event_candidates), len(track_slots) + len(event_candidates)), -1.0, dtype=np.float64)
    lookups: list[dict[str, Mapping[str, Any]]] = []
    for row_index, candidates in enumerate(event_candidates):
        lookup = {str(candidate.get("track_slot", "")): candidate for candidate in candidates}
        lookups.append(lookup)
        for slot, candidate in lookup.items():
            scores[row_index, slot_index[slot]] = float(candidate.get("score", 0.0))
        scores[row_index, len(track_slots) + row_index] = 0.0

    row_indices, column_indices = linear_sum_assignment(-scores)
    assignments: list[Mapping[str, Any] | None] = [None for _ in event_candidates]
    for row_index, column_index in zip(row_indices, column_indices):
        if column_index >= len(track_slots) or scores[row_index, column_index] < 0.0:
            continue
        assignments[row_index] = lookups[row_index].get(track_slots[column_index])
    return assignments


def _stage_lookup(stage_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        team = str(row.get("team", "")).strip()
        slot = str(row.get("track_slot", "")).strip()
        time_value = _float(row.get("time"))
        if not team or not slot or time_value is None:
            continue
        item = dict(row)
        item["_time"] = time_value
        grouped[f"{team}:{slot}"].append(item)
    for values in grouped.values():
        values.sort(key=lambda item: float(item["_time"]))
    return dict(grouped)


def _stage_point(stage_rows: Sequence[Mapping[str, Any]], team: str, slot: str, time_value: float) -> Mapping[str, Any] | None:
    candidates = [row for row in stage_rows if float(row["_time"]) <= time_value]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["_time"]))


def _location_label(row: Mapping[str, Any]) -> str:
    try:
        seconds = int(round(float(row.get("event_time", 0.0))))
        time_label = f"{seconds // 60}:{seconds % 60:02d}"
    except (TypeError, ValueError):
        time_label = "?"
    slot = str(row.get("track_slot", "")).strip()
    return f"{time_label} S{slot or '?'}"


def _draw_death_marker(
    image: Any,
    point: tuple[int, int],
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    marker_size: int,
) -> None:
    x, y = point
    team_color = team_display_color(str(row.get("team", "")), dict(config))
    cv2.drawMarker(
        image,
        (x, y),
        (245, 245, 245),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=marker_size + 6,
        thickness=5,
        line_type=cv2.LINE_AA,
    )
    cv2.drawMarker(
        image,
        (x, y),
        team_color,
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=marker_size,
        thickness=3,
        line_type=cv2.LINE_AA,
    )
    label = _location_label(row)
    label_x = min(max(8, x + marker_size // 2), max(8, image.shape[1] - 118))
    label_y = min(max(24, y - marker_size // 2), image.shape[0] - 8)
    cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)


def render_death_positions(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay located deaths on both source-pixel and normalized route maps."""
    outputs = config.get("outputs", {})
    rendered: dict[str, str] = {}
    located = [row for row in rows if str(row.get("location_status", "")).startswith("located")]

    source_base = resolve_path(outputs.get("team_routes_with_deaths_base", str(Path(outputs["rendered_dir"]) / "team_routes.png")))
    source_output = resolve_path(outputs.get("routes_with_deaths", str(Path(outputs["rendered_dir"]) / "routes_with_deaths.png")))
    source_image = cv2.imread(str(source_base))
    if source_image is not None:
        for row in located:
            x = _float(row.get("x"))
            y = _float(row.get("y"))
            if x is None or y is None:
                continue
            _draw_death_marker(source_image, (int(round(x)), int(round(y))), row, config, marker_size=30)
        source_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(source_output), source_image)
        rendered["source_routes_with_deaths"] = str(source_output)

    stage_dir = resolve_path(outputs.get("rendered_stage_dir", str(Path(config["match"]["output_dir"]) / "rendered_stage")))
    stage_base = stage_dir / "stage_routes.png"
    stage_output = resolve_path(outputs.get("stage_routes_with_deaths", str(stage_dir / "stage_routes_with_deaths.png")))
    stage_image = cv2.imread(str(stage_base))
    if stage_image is not None:
        canvas_size = int(stage_image.shape[0]) if stage_image.shape[0] == stage_image.shape[1] else DEFAULT_CANVAS_SIZE
        margin = int(config.get("rendering", {}).get("stage_margin_px", DEFAULT_MARGIN))
        for row in located:
            stage_x = _float(row.get("stage_x"))
            stage_y = _float(row.get("stage_y"))
            if stage_x is None or stage_y is None or not (0.0 <= stage_x <= 1.0 and 0.0 <= stage_y <= 1.0):
                continue
            point = stage_to_pixel(stage_x, stage_y, canvas_size=canvas_size, margin=margin)
            _draw_death_marker(stage_image, point, row, config, marker_size=24)
        stage_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(stage_output), stage_image)
        rendered["stage_routes_with_deaths"] = str(stage_output)

    return {"status": "ready" if rendered else "no_base_images", "located_markers": len(located), "rendered": rendered}


def build_death_position_rows(
    events: Sequence[DeathEvent | Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    *,
    stage_rows: Sequence[Mapping[str, Any]] = (),
    sample_fps: float = 1.0,
    pre_window: float = 3.0,
    post_window: float = 2.0,
    max_point_delta: float = 2.0,
    verified_slot_mapping: bool = False,
    ambiguity_margin: float = 0.12,
) -> list[dict[str, Any]]:
    grouped = _track_rows(track_rows)
    stage_grouped = _stage_lookup(stage_rows)
    expected_interval = 1.0 / max(sample_fps, 1e-6)
    prepared: list[dict[str, Any]] = []
    for event_index, event in enumerate(events, start=1):
        event_time = _float(_event_value(event, "time"))
        global_slot = _int(_event_value(event, "victim_slot"))
        team = str(_event_value(event, "team")).strip()
        if event_time is None:
            continue
        candidates: list[dict[str, Any]] = []
        slot_hint = _team_slot(global_slot)
        keys = sorted(key for key in grouped if key.startswith(f"{team}:"))
        if verified_slot_mapping and slot_hint is not None:
            keys = [f"{team}:{slot_hint}"] if f"{team}:{slot_hint}" in grouped else []
        for key in keys:
            candidate = _candidate_for_track(
                grouped[key],
                event_time,
                pre_window=pre_window,
                post_window=post_window,
                expected_interval=expected_interval,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        prepared.append(
            {
                "event_index": event_index,
                "event": event,
                "event_time": event_time,
                "global_slot": global_slot,
                "team": team,
                "candidates": candidates,
            }
        )

    grouped_events: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        grouped_events[(str(item["team"]), float(item["event_time"]))].append(item)
    for simultaneous in grouped_events.values():
        assignments = _assign_unique_candidates([item["candidates"] for item in simultaneous])
        for item, assignment in zip(simultaneous, assignments):
            item["best"] = assignment

    output: list[dict[str, Any]] = []
    for item in prepared:
        event_index = int(item["event_index"])
        event = item["event"]
        event_time = float(item["event_time"])
        global_slot = item["global_slot"]
        team = str(item["team"])
        candidates = item["candidates"]
        best = item.get("best")
        alternative_scores = [
            float(candidate["score"])
            for candidate in candidates
            if best is None or str(candidate["track_slot"]) != str(best["track_slot"])
        ]
        second_score = max(alternative_scores) if alternative_scores else None
        margin = None if best is None or second_score is None else float(best["score"]) - second_score
        before = best["before"] if best else None
        point_delta = float(best["before_delta"]) if best else None
        located = before is not None and point_delta is not None and point_delta <= max_point_delta
        if not located:
            status = "unknown"
            if best is None:
                reason = "no_predeath_track_candidate" if not candidates else "simultaneous_assignment_unavailable"
            else:
                reason = "latest_track_point_too_old"
        elif verified_slot_mapping:
            status = "located"
            reason = "verified_hud_slot"
        elif margin is not None and margin < ambiguity_margin:
            status = "ambiguous"
            reason = "candidate_identity_ambiguous"
        else:
            status = "located_unverified"
            reason = "candidate_identity_unverified"

        stage_point = None
        if best:
            stage_point = _stage_point(
                stage_grouped.get(f"{team}:{best['track_slot']}", []),
                team,
                str(best["track_slot"]),
                float(before["_time"]) if before else event_time,
            )
        output.append(
            {
                "event_id": _event_value(event, "event_id") or f"death:{_event_value(event, 'match_id')}:{event_index:04d}",
                "match_id": _event_value(event, "match_id"),
                "event_time": f"{event_time:.3f}",
                "event": _event_value(event, "event"),
                "team": team,
                "victim": _event_value(event, "victim") or _event_value(event, "player"),
                "victim_slot": "" if global_slot is None else global_slot,
                "victim_weapon": _event_value(event, "victim_weapon"),
                "track_slot": best["track_slot"] if best else "",
                "player_id": best["player_id"] if best else "",
                "x": "" if not located else f"{float(before['_x']):.2f}",
                "y": "" if not located else f"{float(before['_y']):.2f}",
                "point_time": "" if before is None else f"{float(before['_time']):.3f}",
                "point_delta_seconds": "" if point_delta is None else f"{point_delta:.3f}",
                "after_point_time": "" if not best or best["after"] is None else f"{float(best['after']['_time']):.3f}",
                "after_point_delta_seconds": "" if not best or best["after_delta"] is None else f"{float(best['after_delta']):.3f}",
                "stage_x": "" if not located or stage_point is None else stage_point.get("stage_x", ""),
                "stage_y": "" if not located or stage_point is None else stage_point.get("stage_y", ""),
                "stage_inside_roi": "" if not located or stage_point is None else stage_point.get("stage_inside_roi", ""),
                "location_status": status,
                "location_reason": reason,
                "assignment_method": "verified_slot" if verified_slot_mapping else "disappearance_candidate",
                "assignment_confidence": "" if best is None else f"{max(0.0, min(1.0, float(best['score']))):.3f}",
                "candidate_count": len(candidates),
                "candidate_margin": "" if margin is None else f"{margin:.3f}",
                "source_frame": "" if before is None else before.get("frame_path", ""),
                "clip_start": _event_value(event, "clip_start"),
                "clip_end": _event_value(event, "clip_end"),
                "evidence": _event_value(event, "evidence"),
                "notes": "" if status in {"located", "located_unverified"} else "map track assignment remains reviewable",
            }
        )
    return output


def build_position_report(rows: Sequence[Mapping[str, Any]], *, match_id: str = "") -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    reason_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("location_status", "unknown"))] += 1
        reason = str(row.get("location_reason", "")).strip()
        if reason:
            reason_counts[reason] += 1
    return {
        "status": "ready" if rows else "empty",
        "match_id": match_id,
        "event_count": len(rows),
        "located_count": counts.get("located", 0) + counts.get("located_unverified", 0),
        "ambiguous_count": counts.get("ambiguous", 0),
        "unknown_count": counts.get("unknown", 0),
        "status_counts": dict(sorted(counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _config_ids(config: Mapping[str, Any]) -> tuple[Sequence[Any], Sequence[Any], int, bool]:
    section = config.get("death_events", {})
    if not isinstance(section, Mapping):
        section = {}
    dead = section.get("dead_state_ids", DEFAULT_DEAD_STATE_IDS)
    alive = section.get("alive_state_ids", DEFAULT_ALIVE_STATE_IDS)
    min_frames = int(section.get("min_dead_frames", 1))
    initial = bool(section.get("include_initial_dead", False))
    return dead, alive, min_frames, initial


def run_death_position_pipeline(config: Mapping[str, Any], event_csv: Path | str | None = None) -> dict[str, Any]:
    outputs = config.get("outputs", {})
    match = config.get("match", {})
    match_id = str(match.get("id", ""))
    state_path = resolve_path(config["state_join"]["state_csv"])
    tracks_path = resolve_path(outputs["player_tracks_csv"])
    stage_path = resolve_path(outputs.get("player_tracks_stage_csv", "")) if outputs.get("player_tracks_stage_csv") else None
    event_path = resolve_path(event_csv) if event_csv else resolve_path(outputs.get("death_events_csv", match.get("output_dir", "") + "/death_events.csv"))
    event_json_path = resolve_path(outputs.get("death_events_json", str(event_path.with_suffix(".json"))))
    position_path = resolve_path(outputs.get("death_positions_csv", str(event_path.with_name("death_positions.csv"))))
    report_path = resolve_path(outputs.get("death_position_report_json", str(event_path.with_name("death_position_report.json"))))

    if event_csv:
        event_rows: list[Mapping[str, Any]] = read_rows(event_path)
        events: list[DeathEvent | Mapping[str, Any]] = event_rows
        event_report: Mapping[str, Any] = {"status": "external", "event_count": len(event_rows), "match_id": match_id}
    else:
        state_rows = read_csv_rows(state_path)
        dead_ids, alive_ids, min_frames, include_initial = _config_ids(config)
        teams = tuple(str(team) for team in (config.get("teams", {}) or {}).keys())
        events = extract_death_events(
            state_rows,
            match_id=match_id,
            dead_state_ids=dead_ids,
            alive_state_ids=alive_ids,
            min_dead_frames=min_frames,
            include_initial_dead=include_initial,
            team_names=teams,
        )
        write_event_csv(event_path, events)
        event_report = build_death_event_report(
            state_rows,
            match_id=match_id,
            events=events,
            dead_state_ids=dead_ids,
            alive_state_ids=alive_ids,
            team_names=teams,
        )
        event_json_path.parent.mkdir(parents=True, exist_ok=True)
        event_json_path.write_text(json.dumps(event_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    track_rows = read_rows(tracks_path)
    stage_rows = read_rows(stage_path) if stage_path else []
    tracking = config.get("identity_tracking", {})
    event_config = config.get("event_join", {})
    position_rows = build_death_position_rows(
        events,
        track_rows,
        stage_rows=stage_rows,
        sample_fps=float(config.get("sampling", {}).get("sample_fps", 1.0)),
        pre_window=float(tracking.get("death_pre_window_seconds", event_config.get("time_window_seconds", 2.0))),
        post_window=float(tracking.get("death_post_window_seconds", 2.0)),
        max_point_delta=float(tracking.get("death_max_point_delta_seconds", 2.0)),
        verified_slot_mapping=bool(tracking.get("slot_mapping_verified", False)),
        ambiguity_margin=float(tracking.get("death_ambiguity_margin", 0.12)),
    )
    write_rows(position_path, POSITION_FIELDS, position_rows)
    position_report = build_position_report(position_rows, match_id=match_id)
    rendering_report = render_death_positions(position_rows, config)
    position_report["event_csv"] = str(event_path)
    position_report["event_json"] = str(event_json_path)
    position_report["tracks_csv"] = str(tracks_path)
    position_report["rendering"] = rendering_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(position_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **position_report,
        "event_csv": str(event_path),
        "event_json": str(event_json_path),
        "position_csv": str(position_path),
        "report_json": str(report_path),
        "event_report": dict(event_report),
        "rendering": rendering_report,
    }


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    report = run_death_position_pipeline(config, args.events)
    print(f"death events: {report['event_count']}")
    print(f"located: {report['located_count']}")
    print(f"ambiguous: {report['ambiguous_count']}")
    print(f"unknown: {report['unknown_count']}")
    print(f"death positions csv: {report['position_csv']}")
    print(f"death position report: {report['report_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
