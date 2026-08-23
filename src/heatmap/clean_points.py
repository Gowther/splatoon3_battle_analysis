from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.heatmap.detect_markers import load_mask
from src.heatmap.extract_frames import load_config, resolve_path


Point = Dict[str, object]
FrameKey = Tuple[float, str]


@dataclass
class TrackState:
    x: float
    y: float
    time: float
    vx: float = 0.0
    vy: float = 0.0
    observations: int = 1

    def predicted_position(self, time_value: float, horizon_seconds: float) -> Tuple[float, float]:
        elapsed = max(0.0, min(float(time_value) - self.time, horizon_seconds))
        return self.x + self.vx * elapsed, self.y + self.vy * elapsed


CLEAN_FIELDNAMES = [
    "match_id",
    "time",
    "frame_index",
    "team",
    "player_id",
    "track_slot_hint",
    "x",
    "y",
    "confidence",
    "source",
    "clean_stage",
    "frame_path",
]

REJECT_FIELDNAMES = [
    "match_id",
    "time",
    "frame_index",
    "team",
    "player_id",
    "track_slot_hint",
    "x",
    "y",
    "confidence",
    "source",
    "area",
    "label_distance",
    "frame_path",
    "reject_reason",
]

TRACK_FIELDNAMES = [
    "match_id",
    "time",
    "frame_index",
    "team",
    "track_slot",
    "player_id",
    "x",
    "y",
    "confidence",
    "track_status",
    "step_distance",
    "time_delta",
    "prediction_error",
    "tracking_confidence",
    "observation_count",
    "source",
    "frame_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw overhead-map marker candidates.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reject_row(row: Dict[str, object], reason: str) -> Dict[str, object]:
    rejected = dict(row)
    rejected["reject_reason"] = reason
    return rejected


def parse_point(row: Dict[str, str], teams: Sequence[str]) -> Tuple[Optional[Point], Optional[Dict[str, object]]]:
    try:
        x = float(row["x"])
        y = float(row["y"])
        confidence = float(row["confidence"])
        time_value = float(row["time"])
    except (KeyError, TypeError, ValueError):
        return None, reject_row(row, "parse_error")

    team = row.get("team", "")
    if team not in teams:
        return None, reject_row(row, "unknown_team")

    point: Point = {
        "match_id": row.get("match_id", ""),
        "time": f"{time_value:.3f}",
        "frame_index": row.get("frame_index", ""),
        "team": team,
        "player_id": row.get("player_id", ""),
        "track_slot_hint": row.get("track_slot_hint", ""),
        "x": round(x, 2),
        "y": round(y, 2),
        "confidence": round(confidence, 4),
        "source": row.get("source", ""),
        "area": row.get("area", ""),
        "label_distance": row.get("label_distance", ""),
        "frame_path": row.get("frame_path", ""),
    }
    return point, None


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def point_inside_mask(point: Point, mask: np.ndarray) -> bool:
    x = int(round(float(point["x"])))
    y = int(round(float(point["y"])))
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return False
    return bool(mask[y, x] > 0)


def clean_raw_points(config: Dict) -> Tuple[List[Point], List[Dict[str, object]]]:
    raw_rows = read_csv(resolve_path(config["outputs"]["raw_points_csv"]))
    mask = load_mask(config)
    cleaning = config["point_cleaning"]
    min_confidence = float(cleaning["min_confidence"])
    merge_distance = float(cleaning["merge_distance_px"])
    max_per_team = int(cleaning["max_points_per_team_per_frame"])
    teams = list(config["teams"].keys())

    grouped: DefaultDict[Tuple[str, str, str], List[Point]] = defaultdict(list)
    rejected: List[Dict[str, object]] = []

    for row in raw_rows:
        point, rejection = parse_point(row, teams)
        if rejection is not None:
            rejected.append(rejection)
            continue
        assert point is not None
        if float(point["confidence"]) < min_confidence:
            rejected.append(reject_row(point, "low_confidence"))
            continue
        if not point_inside_mask(point, mask):
            rejected.append(reject_row(point, "outside_map_mask"))
            continue
        group_key = (str(point["time"]), str(point["frame_index"]), str(point["team"]))
        grouped[group_key].append(point)

    clean_points: List[Point] = []
    for group_key in sorted(grouped, key=lambda item: (float(item[0]), item[2])):
        kept: List[Point] = []
        for point in sorted(grouped[group_key], key=lambda row: float(row["confidence"]), reverse=True):
            if any(
                point_distance(point, other) < merge_distance
                and (
                    not point.get("player_id")
                    or not other.get("player_id")
                    or point.get("player_id") == other.get("player_id")
                )
                for other in kept
            ):
                rejected.append(reject_row(point, "duplicate_nearby_point"))
                continue
            if len(kept) >= max_per_team:
                rejected.append(reject_row(point, "over_team_frame_limit"))
                continue
            clean_point = dict(point)
            clean_point["clean_stage"] = "confidence_mask_limit"
            kept.append(clean_point)
        clean_points.extend(kept)

    clean_points.sort(key=lambda row: (float(row["time"]), str(row["team"]), -float(row["confidence"])))
    return clean_points, rejected


def group_points_by_frame(clean_points: Sequence[Point]) -> DefaultDict[Tuple[str, str, str], List[Point]]:
    grouped: DefaultDict[Tuple[str, str, str], List[Point]] = defaultdict(list)
    for point in clean_points:
        grouped[(str(point["time"]), str(point["frame_index"]), str(point["team"]))].append(point)
    return grouped


def classify_track_status(
    previous: Optional[TrackState],
    time_delta: Optional[float],
    distance: Optional[float],
    tracking_config: Dict,
) -> str:
    """Classify continuity before a row is allowed to count as matched."""
    if previous is None:
        return "new"
    max_gap = float(tracking_config.get("max_track_gap_seconds", 3.0))
    if time_delta is None or not 0.0 < time_delta <= max_gap:
        return "reacquired"
    max_matched_gap = float(tracking_config.get("max_matched_gap_seconds", max_gap))
    if time_delta > max_matched_gap:
        return "reacquired"
    max_matched_step = float(
        tracking_config.get("max_matched_step_px", tracking_config.get("max_track_step_px", float("inf")))
    )
    if distance is not None and distance > max_matched_step:
        return "jump_reset"
    return "matched"


def assign_tracks_for_team(
    candidates: Sequence[Point],
    states: Dict[int, Optional[TrackState]],
    time_value: float,
    tracking_config: Dict,
) -> List[Dict[str, object]]:
    """Assign one frame of candidates to stable slots with global matching.

    The previous implementation accepted the shortest pair greedily. That can
    swap two slots when players cross because an early local choice prevents a
    better assignment for the remaining points. A four-by-four Hungarian match
    is cheap and lets velocity, elapsed time, and detection confidence
    contribute to one global decision.
    """
    candidates = sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True)
    assigned_slots: Dict[int, Tuple[int, str, Optional[float], Optional[float], Optional[float]]] = {}
    assigned_candidate_indexes: set[int] = set()
    max_speed = float(tracking_config.get("max_track_speed_px_per_second", tracking_config["max_track_step_px"]))
    max_gap = float(tracking_config.get("max_track_gap_seconds", 3.0))
    prediction_horizon = float(tracking_config.get("prediction_horizon_seconds", 1.0))
    confidence_weight = float(tracking_config.get("assignment_confidence_weight_px", 30.0))
    velocity_alpha = float(tracking_config.get("velocity_smoothing_alpha", 0.65))
    min_gate = float(tracking_config.get("min_assignment_gate_px", 45.0))

    # A calibrated name template identifies the player before spatial tracking.
    # Bind those observations to their configured slot and use motion only as a
    # confidence signal. Anonymous detections still use global assignment below.
    for candidate_index, point in enumerate(candidates):
        try:
            slot = int(point.get("track_slot_hint", ""))
        except (TypeError, ValueError):
            continue
        if slot not in states or slot in assigned_slots:
            continue
        previous = states[slot]
        if previous is None:
            assigned_slots[slot] = (candidate_index, "new", None, None, None)
        else:
            time_delta = time_value - previous.time
            distance = math.hypot(float(point["x"]) - previous.x, float(point["y"]) - previous.y)
            predicted_x, predicted_y = previous.predicted_position(time_value, prediction_horizon)
            prediction_error = math.hypot(
                float(point["x"]) - predicted_x,
                float(point["y"]) - predicted_y,
            )
            status = classify_track_status(previous, time_delta, distance, tracking_config)
            assigned_slots[slot] = (candidate_index, status, distance, time_delta, prediction_error)
        assigned_candidate_indexes.add(candidate_index)

    active_slots = [
        slot
        for slot, state in sorted(states.items())
        if slot not in assigned_slots and state is not None and 0.0 < time_value - state.time <= max_gap
    ]
    if active_slots and candidates:
        invalid_cost = 1e9
        costs = np.full((len(active_slots), len(candidates)), invalid_cost, dtype=np.float64)
        metadata: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        for slot_index, slot in enumerate(active_slots):
            state = states[slot]
            assert state is not None
            time_delta = time_value - state.time
            predicted_x, predicted_y = state.predicted_position(time_value, prediction_horizon)
            gate = max(min_gate, max_speed * time_delta)
            for candidate_index, point in enumerate(candidates):
                if candidate_index in assigned_candidate_indexes:
                    continue
                prediction_error = math.hypot(
                    float(point["x"]) - predicted_x,
                    float(point["y"]) - predicted_y,
                )
                if prediction_error > gate:
                    continue
                step_distance = math.hypot(
                    float(point["x"]) - state.x,
                    float(point["y"]) - state.y,
                )
                confidence_penalty = (1.0 - float(point["confidence"])) * confidence_weight
                costs[slot_index, candidate_index] = prediction_error + confidence_penalty
                metadata[(slot_index, candidate_index)] = (step_distance, time_delta, prediction_error)

        slot_indexes, candidate_indexes = linear_sum_assignment(costs)
        for slot_index, candidate_index in zip(slot_indexes.tolist(), candidate_indexes.tolist()):
            if costs[slot_index, candidate_index] >= invalid_cost:
                continue
            slot = active_slots[slot_index]
            step_distance, time_delta, prediction_error = metadata[(slot_index, candidate_index)]
            previous = states[slot]
            assigned_slots[slot] = (
                candidate_index,
                classify_track_status(previous, time_delta, step_distance, tracking_config),
                step_distance,
                time_delta,
                prediction_error,
            )
            assigned_candidate_indexes.add(candidate_index)

    for index, point in enumerate(candidates):
        if index in assigned_candidate_indexes:
            continue
        available_slots = [slot for slot in sorted(states) if slot not in assigned_slots]
        if not available_slots:
            continue
        empty_slots = [slot for slot in available_slots if states[slot] is None]
        if empty_slots:
            slot = empty_slots[0]
            assigned_slots[slot] = (index, "new", None, None, None)
        else:
            stale_slots = [
                slot
                for slot in available_slots
                if states[slot] is not None and time_value - states[slot].time > max_gap
            ]
            if not stale_slots:
                continue
            slot = min(stale_slots, key=lambda slot_number: states[slot_number].time if states[slot_number] else 0.0)
            previous = states[slot]
            assert previous is not None
            distance = math.hypot(float(point["x"]) - previous.x, float(point["y"]) - previous.y)
            assigned_slots[slot] = (index, "reacquired", distance, time_value - previous.time, None)
        assigned_candidate_indexes.add(index)

    track_rows: List[Dict[str, object]] = []
    for slot, (index, status, distance, time_delta, prediction_error) in sorted(assigned_slots.items()):
        point = candidates[index]
        previous = states[slot]
        observations = 1
        vx = 0.0
        vy = 0.0
        if previous is not None and time_delta is not None and time_delta > 0.0:
            observed_vx = (float(point["x"]) - previous.x) / time_delta
            observed_vy = (float(point["y"]) - previous.y) / time_delta
            vx = velocity_alpha * observed_vx + (1.0 - velocity_alpha) * previous.vx
            vy = velocity_alpha * observed_vy + (1.0 - velocity_alpha) * previous.vy
            observations = previous.observations + 1
        states[slot] = TrackState(
            x=float(point["x"]),
            y=float(point["y"]),
            time=time_value,
            vx=vx,
            vy=vy,
            observations=observations,
        )
        detection_confidence = float(point["confidence"])
        if status == "matched" and prediction_error is not None and time_delta is not None:
            gate = max(min_gate, max_speed * time_delta)
            continuity_score = max(0.0, 1.0 - prediction_error / gate)
            tracking_confidence = 0.55 * detection_confidence + 0.45 * continuity_score
        elif status == "new":
            tracking_confidence = 0.55 * detection_confidence
        else:
            tracking_confidence = 0.40 * detection_confidence
        track_rows.append(
            {
                "match_id": point["match_id"],
                "time": point["time"],
                "frame_index": point["frame_index"],
                "team": point["team"],
                "track_slot": slot,
                "player_id": point.get("player_id", ""),
                "x": point["x"],
                "y": point["y"],
                "confidence": point["confidence"],
                "track_status": status,
                "step_distance": "" if distance is None else round(distance, 2),
                "time_delta": "" if time_delta is None else round(time_delta, 3),
                "prediction_error": "" if prediction_error is None else round(prediction_error, 2),
                "tracking_confidence": round(max(0.0, min(1.0, tracking_confidence)), 4),
                "observation_count": observations,
                "source": point.get("source", ""),
                "frame_path": point["frame_path"],
            }
        )
    return track_rows


def build_tracks(clean_points: Sequence[Point], config: Dict) -> List[Dict[str, object]]:
    cleaning = config["point_cleaning"]
    slots_per_team = int(cleaning["track_slots_per_team"])
    teams = list(config["teams"].keys())
    grouped = group_points_by_frame(clean_points)
    frame_keys = sorted({(time, frame_index) for time, frame_index, _ in grouped}, key=lambda item: float(item[0]))
    states: Dict[str, Dict[int, Optional[TrackState]]] = {
        team: {slot: None for slot in range(1, slots_per_team + 1)} for team in teams
    }

    tracks: List[Dict[str, object]] = []
    for time_value, frame_index in frame_keys:
        for team in teams:
            candidates = grouped.get((time_value, frame_index, team), [])
            tracks.extend(assign_tracks_for_team(candidates, states[team], float(time_value), cleaning))
    return tracks


def metric_rows(
    raw_count: int,
    clean_points: Sequence[Point],
    rejected: Sequence[Dict[str, object]],
    tracks: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = [
        {"metric": "raw_points", "value": raw_count},
        {"metric": "clean_points", "value": len(clean_points)},
        {"metric": "rejected_points", "value": len(rejected)},
        {"metric": "track_rows", "value": len(tracks)},
        {"metric": "track_unassigned_points", "value": max(0, len(clean_points) - len(tracks))},
        {
            "metric": "track_coverage_ratio",
            "value": round(len(tracks) / len(clean_points), 4) if clean_points else 0.0,
        },
    ]
    tracking_confidences = [float(row["tracking_confidence"]) for row in tracks if row.get("tracking_confidence") != ""]
    rows.append(
        {
            "metric": "mean_tracking_confidence",
            "value": round(sum(tracking_confidences) / len(tracking_confidences), 4)
            if tracking_confidences
            else 0.0,
        }
    )

    for team, count in sorted(Counter(str(point["team"]) for point in clean_points).items()):
        rows.append({"metric": f"clean_team_{team}", "value": count})
    for reason, count in sorted(Counter(str(row["reject_reason"]) for row in rejected).items()):
        rows.append({"metric": f"rejected_{reason}", "value": count})
    status_counts = Counter(str(row["track_status"]) for row in tracks)
    for status in ("matched", "jump_reset", "new", "reacquired"):
        rows.append({"metric": f"track_status_{status}", "value": status_counts.pop(status, 0)})
    for status, count in sorted(status_counts.items()):
        rows.append({"metric": f"track_status_{status}", "value": count})

    frame_counts = Counter(str(point["time"]) for point in clean_points)
    for point_count, frame_count in sorted(Counter(frame_counts.values()).items()):
        rows.append({"metric": f"frames_with_{point_count}_points", "value": frame_count})
    return rows


def choose_reference_frame(points: Sequence[Point], reference_time: float) -> Optional[Path]:
    with_paths = [point for point in points if point.get("frame_path")]
    if not with_paths:
        return None
    chosen = min(with_paths, key=lambda row: abs(float(row["time"]) - reference_time))
    return resolve_path(str(chosen["frame_path"]))


def draw_cleaning_debug(
    clean_points: Sequence[Point],
    rejected: Sequence[Dict[str, object]],
    config: Dict,
) -> Optional[Path]:
    output_dir = resolve_path(config["outputs"]["cleaning_debug_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_time = float(config["point_cleaning"]["debug_reference_time_seconds"])
    reference_path = choose_reference_frame(clean_points, reference_time)
    if reference_path is None:
        return None

    base = cv2.imread(str(reference_path))
    if base is None:
        return None
    overlay = base.copy()
    team_colors = {"yellow": (0, 255, 255), "blue": (255, 60, 0)}
    for point in clean_points:
        x = int(round(float(point["x"])))
        y = int(round(float(point["y"])))
        color = team_colors.get(str(point["team"]), (255, 255, 255))
        cv2.circle(overlay, (x, y), 3, color, -1)

    for point in rejected:
        try:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
        except (KeyError, TypeError, ValueError):
            continue
        cv2.drawMarker(overlay, (x, y), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=9, thickness=1)

    output = cv2.addWeighted(overlay, 0.70, base, 0.30, 0)
    cv2.putText(
        output,
        f"clean={len(clean_points)} rejected={len(rejected)} ref={reference_time:.1f}s",
        (28, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"clean={len(clean_points)} rejected={len(rejected)} ref={reference_time:.1f}s",
        (28, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    output_path = output_dir / "cleaned_points_overview.jpg"
    cv2.imwrite(str(output_path), output)
    return output_path


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    raw_count = len(read_csv(resolve_path(config["outputs"]["raw_points_csv"])))
    clean_points, rejected = clean_raw_points(config)
    tracks = build_tracks(clean_points, config)

    write_csv(resolve_path(config["outputs"]["clean_points_csv"]), CLEAN_FIELDNAMES, clean_points)
    write_csv(resolve_path(config["outputs"]["rejected_points_csv"]), REJECT_FIELDNAMES, rejected)
    write_csv(resolve_path(config["outputs"]["tracks_csv"]), TRACK_FIELDNAMES, tracks)
    write_csv(
        resolve_path(config["outputs"]["cleaning_report_csv"]),
        ["metric", "value"],
        metric_rows(raw_count, clean_points, rejected, tracks),
    )
    debug_path = draw_cleaning_debug(clean_points, rejected, config)

    print(f"raw points: {raw_count}")
    print(f"clean points: {len(clean_points)}")
    print(f"rejected points: {len(rejected)}")
    print(f"track rows: {len(tracks)}")
    print(f"clean points csv: {resolve_path(config['outputs']['clean_points_csv'])}")
    print(f"tracks csv: {resolve_path(config['outputs']['tracks_csv'])}")
    if debug_path is not None:
        print(f"cleaning debug: {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
