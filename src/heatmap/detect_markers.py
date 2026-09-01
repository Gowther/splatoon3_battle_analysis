from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.csv_contracts import RAW_MARKER_CSV_CONTRACT
from src.heatmap.extract_frames import load_config, resolve_path, team_mask


Point = Dict[str, object]
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PlayerTemplate:
    player_id: str
    team: str
    track_slot: int
    reference_box: Box
    reference_marker: Tuple[int, int]
    marker_offset: Tuple[int, int]
    label_edges: np.ndarray
    marker_edges: np.ndarray


@dataclass
class PlayerTemplateState:
    x: float
    y: float
    time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect team marker candidates on overhead-map frames.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--limit", type=int, help="Only process the first N valid frames.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_points(path: Path, rows: Sequence[Point]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_MARKER_CSV_CONTRACT.fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_mask(config: Dict) -> np.ndarray:
    mask_path = resolve_path(config["outputs"]["map_mask"])
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return mask


def component_boxes(mask: np.ndarray) -> List[Tuple[Box, int, Tuple[float, float]]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    output: List[Tuple[Box, int, Tuple[float, float]]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        cx, cy = centroids[index]
        output.append(((int(x), int(y), int(x + width), int(y + height)), int(area), (float(cx), float(cy))))
    return output


def detect_label_boxes(frame: np.ndarray, map_mask: np.ndarray, config: Dict) -> List[Tuple[Box, Tuple[float, float]]]:
    label_config = config["label_detection"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(label_config["white_hsv_lower"], dtype=np.uint8)
    upper = np.array(label_config["white_hsv_upper"], dtype=np.uint8)
    white = cv2.inRange(hsv, lower, upper)
    white = cv2.bitwise_and(white, white, mask=map_mask)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 9), dtype=np.uint8))

    labels: List[Tuple[Box, Tuple[float, float]]] = []
    for box, area, centroid in component_boxes(white):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if not (label_config["min_label_area"] <= area <= label_config["max_label_area"]):
            continue
        if not (label_config["min_label_width"] <= width <= label_config["max_label_width"]):
            continue
        if not (label_config["min_label_height"] <= height <= label_config["max_label_height"]):
            continue
        labels.append((box, centroid))
    return labels


def nearest_label(
    centroid: Tuple[float, float],
    labels: Sequence[Tuple[Box, Tuple[float, float]]],
) -> Tuple[Optional[Tuple[Box, Tuple[float, float]]], float]:
    if not labels:
        return None, float("inf")
    cx, cy = centroid
    best_label = None
    best_distance = float("inf")
    for label in labels:
        _, (lx, ly) = label
        distance = math.hypot(cx - lx, cy - ly)
        if distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label, best_distance


def merge_points(points: Sequence[Point], distance_px: float, limit: int) -> List[Point]:
    kept: List[Point] = []
    for point in sorted(points, key=lambda row: float(row["confidence"]), reverse=True):
        px = float(point["x"])
        py = float(point["y"])
        if any(math.hypot(px - float(other["x"]), py - float(other["y"])) < distance_px for other in kept):
            continue
        kept.append(point)
        if len(kept) >= limit:
            break
    return kept


def detect_team_points(
    frame: np.ndarray,
    map_mask: np.ndarray,
    labels: Sequence[Tuple[Box, Tuple[float, float]]],
    team: str,
    team_config: Dict,
    config: Dict,
) -> List[Point]:
    marker_config = config["marker_detection"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = team_mask(hsv, team_config["hsv_ranges"])
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=map_mask)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    raw_points: List[Point] = []
    max_area = float(marker_config["max_component_area"])
    proximity = float(marker_config["label_proximity_px"])
    for box, area, centroid in component_boxes(color_mask):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if not (marker_config["min_component_area"] <= area <= marker_config["max_component_area"]):
            continue
        if not (marker_config["min_component_width"] <= width <= marker_config["max_component_width"]):
            continue
        if not (marker_config["min_component_height"] <= height <= marker_config["max_component_height"]):
            continue

        _, label_distance = nearest_label(centroid, labels)
        if label_distance > proximity:
            continue
        area_score = min(1.0, area / max_area)
        distance_score = max(0.0, 1.0 - label_distance / proximity)
        confidence = 0.35 * area_score + 0.65 * distance_score
        if confidence < float(marker_config["min_confidence"]):
            continue
        raw_points.append(
            {
                "team": team,
                "player_id": "",
                "x": round(float(centroid[0]), 2),
                "y": round(float(centroid[1]), 2),
                "confidence": round(confidence, 4),
                "source": "label_guided_color_component",
                "area": area,
                "label_distance": round(label_distance, 2),
            }
        )

    return merge_points(
        raw_points,
        float(marker_config["merge_distance_px"]),
        int(marker_config["max_points_per_team_per_frame"]),
    )


def template_edge_frame(frame: np.ndarray, marker_config: Dict) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(
        gray,
        int(marker_config.get("template_edge_low", 80)),
        int(marker_config.get("template_edge_high", 160)),
    )


def read_video_frame_at(config: Dict, time_seconds: float) -> np.ndarray:
    video_path = resolve_path(config["match"]["input_video"])
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read template reference frame at {time_seconds:.3f}s: {video_path}")
    return frame


def build_player_templates(reference_frame: np.ndarray, config: Dict) -> List[PlayerTemplate]:
    marker_config = config["marker_detection"]
    edges = template_edge_frame(reference_frame, marker_config)
    marker_radius = int(marker_config.get("template_marker_radius_px", 10))
    height, width = edges.shape
    templates: List[PlayerTemplate] = []

    for item in marker_config.get("player_templates", []):
        x1, y1, x2, y2 = (int(value) for value in item["reference_box"])
        if "reference_marker" in item:
            marker_x, marker_y = (int(value) for value in item["reference_marker"])
            offset_x = marker_x - x1
            offset_y = marker_y - y1
        else:
            offset_x, offset_y = (int(value) for value in item["marker_offset"])
            marker_x = x1 + offset_x
            marker_y = y1 + offset_y
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid player template box for {item['player_id']}: {item['reference_box']}")
        mx1 = marker_x - marker_radius
        my1 = marker_y - marker_radius
        mx2 = marker_x + marker_radius + 1
        my2 = marker_y + marker_radius + 1
        if not (0 <= mx1 < mx2 <= width and 0 <= my1 < my2 <= height):
            raise ValueError(f"Invalid marker offset for {item['player_id']}: {item['marker_offset']}")
        label_edges = edges[y1:y2, x1:x2].copy()
        if not np.any(label_edges):
            raise ValueError(f"Empty edge template for {item['player_id']}")
        templates.append(
            PlayerTemplate(
                player_id=str(item["player_id"]),
                team=str(item["team"]),
                track_slot=int(item["track_slot"]),
                reference_box=(x1, y1, x2, y2),
                reference_marker=(marker_x, marker_y),
                marker_offset=(offset_x, offset_y),
                label_edges=label_edges,
                marker_edges=edges[my1:my2, mx1:mx2].copy(),
            )
        )
    if not templates:
        raise ValueError("marker_detection.player_templates must contain at least one template")
    return templates


def _template_search_box(
    template: PlayerTemplate,
    state: Optional[PlayerTemplateState],
    time_value: float,
    frame_shape: Tuple[int, int],
    config: Dict,
) -> Tuple[Box, bool, float]:
    height, width = frame_shape
    map_roi = config["map_view"]["roi"]
    x1 = max(0, int(map_roi["x1"]))
    y1 = max(0, int(map_roi["y1"]))
    x2 = min(width, int(map_roi["x2"]))
    y2 = min(height, int(map_roi["y2"]))
    marker_config = config["marker_detection"]
    max_gap = float(marker_config.get("template_max_gap_seconds", 3.0))
    time_delta = abs(time_value - state.time) if state is not None else float("inf")
    tracked = state is not None and 0.0 < time_delta <= max_gap
    gate = float(marker_config.get("template_min_search_gate_px", 90.0))
    if tracked and state is not None:
        gate = max(gate, float(marker_config.get("template_max_speed_px_per_second", 420.0)) * time_delta)
        expected_x = state.x - template.marker_offset[0]
        expected_y = state.y - template.marker_offset[1]
        x1 = max(x1, int(math.floor(expected_x - gate)))
        y1 = max(y1, int(math.floor(expected_y - gate)))
        x2 = min(x2, int(math.ceil(expected_x + gate + template.label_edges.shape[1])))
        y2 = min(y2, int(math.ceil(expected_y + gate + template.label_edges.shape[0])))
    return (x1, y1, x2, y2), tracked, gate


def _same_size_match_score(candidate: np.ndarray, template: np.ndarray) -> float:
    if candidate.shape != template.shape or not np.any(template):
        return 0.0
    score = float(cv2.matchTemplate(candidate, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score if math.isfinite(score) else 0.0


def match_marker_near(
    frame_edges: np.ndarray,
    marker_edges: np.ndarray,
    center: Tuple[int, int],
    search_radius: int,
) -> Tuple[float, Tuple[int, int]]:
    template_height, template_width = marker_edges.shape
    half_width = template_width // 2
    half_height = template_height // 2
    center_x, center_y = center
    x1 = max(0, center_x - half_width - search_radius)
    y1 = max(0, center_y - half_height - search_radius)
    x2 = min(frame_edges.shape[1], center_x + (template_width - half_width) + search_radius)
    y2 = min(frame_edges.shape[0], center_y + (template_height - half_height) + search_radius)
    search = frame_edges[y1:y2, x1:x2]
    if search.shape[0] < template_height or search.shape[1] < template_width:
        return 0.0, center
    response = cv2.matchTemplate(search, marker_edges, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    if not math.isfinite(score):
        return 0.0, center
    marker_x = x1 + int(location[0]) + half_width
    marker_y = y1 + int(location[1]) + half_height
    return float(score), (marker_x, marker_y)


def match_player_template(
    frame_edges: np.ndarray,
    map_mask: np.ndarray,
    template: PlayerTemplate,
    state: Optional[PlayerTemplateState],
    time_value: float,
    config: Dict,
) -> Optional[Point]:
    search_box, tracked, gate = _template_search_box(template, state, time_value, frame_edges.shape, config)
    x1, y1, x2, y2 = search_box
    template_height, template_width = template.label_edges.shape
    search = frame_edges[y1:y2, x1:x2]
    if search.shape[0] < template_height or search.shape[1] < template_width:
        return None
    response = cv2.matchTemplate(search, template.label_edges, cv2.TM_CCOEFF_NORMED)
    _, label_score, _, location = cv2.minMaxLoc(response)
    label_x = x1 + int(location[0])
    label_y = y1 + int(location[1])
    expected_marker = (
        label_x + template.marker_offset[0],
        label_y + template.marker_offset[1],
    )
    marker_config = config["marker_detection"]
    marker_score, (marker_x, marker_y) = match_marker_near(
        frame_edges,
        template.marker_edges,
        expected_marker,
        int(marker_config.get("template_marker_search_radius_px", 8)),
    )
    if not (0 <= marker_x < map_mask.shape[1] and 0 <= marker_y < map_mask.shape[0]):
        return None
    if map_mask[marker_y, marker_x] == 0:
        return None

    distance = None
    time_delta = None
    continuity = 0.0
    if tracked and state is not None:
        distance = math.hypot(marker_x - state.x, marker_y - state.y)
        time_delta = abs(time_value - state.time)
        continuity = max(0.0, 1.0 - distance / gate)

    min_score_key = "template_tracked_min_score" if tracked else "template_reacquire_min_score"
    if label_score < float(marker_config.get(min_score_key, 0.10 if tracked else 0.18)):
        return None
    marker_min_score_key = "template_marker_tracked_min_score" if tracked else "template_marker_reacquire_min_score"
    marker_min_score = float(marker_config.get(marker_min_score_key, 0.18 if tracked else 0.30))
    if marker_score < marker_min_score:
        return None
    if tracked and distance is not None and distance > gate:
        return None
    quality = 0.35 * max(0.0, float(label_score)) + 0.35 * max(0.0, marker_score) + 0.30 * continuity
    confidence = max(0.0, min(1.0, 0.50 + 0.50 * quality))
    return {
        "team": template.team,
        "player_id": template.player_id,
        "track_slot_hint": template.track_slot,
        "x": round(float(marker_x), 2),
        "y": round(float(marker_y), 2),
        "confidence": round(confidence, 4),
        "source": "seeded_name_marker_template",
        "area": int(np.count_nonzero(template.label_edges)),
        "label_distance": round(1.0 - float(label_score), 4),
        "_label_score": float(label_score),
        "_marker_score": marker_score,
        "_continuity": continuity,
        "_label_box": (label_x, label_y, label_x + template_width, label_y + template_height),
        "_time_delta": time_delta,
        "_distance": distance,
    }


def nearest_state_row(state_rows: Sequence[Dict[str, str]], time_value: float) -> Optional[Dict[str, str]]:
    if not state_rows:
        return None
    return min(state_rows, key=lambda row: abs(float(row["elapsed_time"]) - time_value))


def state_row_at_or_before(
    state_rows: Sequence[Dict[str, str]], time_value: float
) -> Optional[Dict[str, str]]:
    """Return the latest HUD sample that cannot contain future information."""
    if not state_rows:
        return None
    for row in reversed(state_rows):
        if float(row["elapsed_time"]) <= time_value + 1e-6:
            return row
    # There is no causal HUD evidence before the first sample. Returning the
    # first row here would leak a future death state into earlier map frames.
    return None


def alive_limits(config: Dict, state_rows: Sequence[Dict[str, str]], time_value: float) -> Dict[str, int]:
    state = state_row_at_or_before(state_rows, time_value)
    if state is None:
        return {}
    dead_ids = {str(value) for value in config.get("death_events", {}).get("dead_state_ids", [1, 3])}
    teams = list(config["teams"])
    output: Dict[str, int] = {}
    for team_index, team in enumerate(teams):
        values = [state.get(f"player_state_{team_index * 4 + slot}", "") for slot in range(1, 5)]
        if all(value != "" for value in values):
            output[team] = sum(value not in dead_ids for value in values)
    return output


def player_template_is_alive(
    template: PlayerTemplate,
    state: Optional[Dict[str, str]],
    config: Dict,
) -> bool:
    if state is None:
        return True
    teams = list(config["teams"])
    try:
        team_index = teams.index(template.team)
    except ValueError:
        return True
    value = str(state.get(f"player_state_{team_index * 4 + template.track_slot}", "")).strip()
    if not value:
        return True
    dead_ids = {str(item) for item in config.get("death_events", {}).get("dead_state_ids", [1, 3])}
    return value not in dead_ids


def select_template_points(
    candidates: Sequence[Point],
    limits: Dict[str, int],
) -> List[Point]:
    selected: List[Point] = []
    teams = sorted({str(point["team"]) for point in candidates})
    for team in teams:
        team_points = [point for point in candidates if point["team"] == team]
        team_points.sort(key=lambda point: float(point["confidence"]), reverse=True)
        limit = limits.get(team, len(team_points))
        selected.extend(team_points[:limit])
    return selected


def resolve_template_collisions(candidates: Sequence[Point], config: Dict) -> List[Point]:
    merge_distance = float(config["marker_detection"].get("template_collision_distance_px", 26.0))
    kept: List[Point] = []
    for point in sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True):
        if any(
            math.hypot(float(point["x"]) - float(other["x"]), float(point["y"]) - float(other["y"]))
            < merge_distance
            for other in kept
        ):
            continue
        kept.append(point)
    return kept


def load_state_rows(config: Dict) -> List[Dict[str, str]]:
    state_path_value = config.get("state_join", {}).get("state_csv")
    if not state_path_value:
        return []
    state_path = resolve_path(state_path_value)
    if not state_path.exists():
        return []
    return read_csv(state_path)


def reference_template_points(
    templates: Sequence[PlayerTemplate],
    frame_row: Dict[str, str],
    config: Dict,
) -> List[Point]:
    points: List[Point] = []
    for template in templates:
        points.append(
            {
                "match_id": config["match"]["id"],
                "time": frame_row["time"],
                "frame_index": frame_row["frame_index"],
                "team": template.team,
                "player_id": template.player_id,
                "track_slot_hint": template.track_slot,
                "x": float(template.reference_marker[0]),
                "y": float(template.reference_marker[1]),
                "confidence": 1.0,
                "source": "reference_marker_seed",
                "area": int(np.count_nonzero(template.marker_edges)),
                "label_distance": 0.0,
                "frame_path": frame_row["frame_path"],
            }
        )
    return points


def track_template_direction(
    frame_rows: Sequence[Dict[str, str]],
    templates: Sequence[PlayerTemplate],
    map_mask: np.ndarray,
    state_rows: Sequence[Dict[str, str]],
    config: Dict,
) -> List[Point]:
    reference_time = float(config["marker_detection"]["reference_time_seconds"])
    states = {
        template.player_id: PlayerTemplateState(
            x=float(template.reference_marker[0]),
            y=float(template.reference_marker[1]),
            time=reference_time,
        )
        for template in templates
    }
    output: List[Point] = []
    for frame_row in frame_rows:
        frame = cv2.imread(str(resolve_path(frame_row["frame_path"])))
        if frame is None:
            continue
        time_value = float(frame_row["time"])
        frame_edges = template_edge_frame(frame, config["marker_detection"])
        state_row = state_row_at_or_before(state_rows, time_value)
        candidates: List[Point] = []
        for template in templates:
            if not player_template_is_alive(template, state_row, config):
                continue
            candidate = match_player_template(
                frame_edges,
                map_mask,
                template,
                states.get(template.player_id),
                time_value,
                config,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates = resolve_template_collisions(candidates, config)
        frame_points = select_template_points(candidates, alive_limits(config, state_rows, time_value))
        for point in frame_points:
            point["match_id"] = config["match"]["id"]
            point["time"] = frame_row["time"]
            point["frame_index"] = frame_row["frame_index"]
            point["frame_path"] = frame_row["frame_path"]
            states[str(point["player_id"])] = PlayerTemplateState(
                x=float(point["x"]),
                y=float(point["y"]),
                time=time_value,
            )
        output.extend(frame_points)
    return output


def draw_debug(
    frame: np.ndarray,
    labels: Sequence[Tuple[Box, Tuple[float, float]]],
    points: Sequence[Point],
    output_path: Path,
) -> None:
    output = frame.copy()
    for box, _ in labels:
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), (210, 210, 210), 1)
    colors = {"yellow": (0, 255, 255), "blue": (255, 40, 0)}
    for point in points:
        x = int(round(float(point["x"])))
        y = int(round(float(point["y"])))
        color = colors.get(str(point["team"]), (255, 255, 255))
        label_box = point.get("_label_box")
        if label_box:
            x1, y1, x2, y2 = (int(value) for value in label_box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.circle(output, (x, y), 12, color, 2)
        cv2.putText(
            output,
            f"{point.get('player_id') or point['team']} {float(point['confidence']):.2f}",
            (x + 14, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)


def detect_markers(config: Dict, limit: Optional[int] = None) -> List[Point]:
    valid_rows = read_csv(resolve_path(config["outputs"]["valid_frames_csv"]))
    if limit is not None:
        valid_rows = valid_rows[:limit]
    map_mask = load_mask(config)
    debug_dir = resolve_path(config["outputs"]["debug_markers_dir"])
    debug_limit = int(config["marker_detection"].get("debug_frame_limit", 0))
    method = str(config["marker_detection"].get("method", "label_guided_color_components"))
    all_points: List[Point] = []
    player_templates: List[PlayerTemplate] = []
    state_rows: List[Dict[str, str]] = []

    if method in {"player_name_template_tracking", "seeded_name_marker_tracking"}:
        reference_time = float(config["marker_detection"]["reference_time_seconds"])
        reference_frame = read_video_frame_at(config, reference_time)
        player_templates = build_player_templates(reference_frame, config)
        state_rows = load_state_rows(config)

    if debug_limit > 0 and debug_dir.exists():
        for old_debug in debug_dir.glob("markers_*.jpg"):
            old_debug.unlink()

    if method in {"player_name_template_tracking", "seeded_name_marker_tracking"}:
        reference_row = min(valid_rows, key=lambda row: abs(float(row["time"]) - reference_time))
        before_rows = [row for row in valid_rows if float(row["time"]) < float(reference_row["time"])]
        after_rows = [row for row in valid_rows if float(row["time"]) > float(reference_row["time"])]
        all_points.extend(
            track_template_direction(list(reversed(before_rows)), player_templates, map_mask, state_rows, config)
        )
        all_points.extend(reference_template_points(player_templates, reference_row, config))
        all_points.extend(track_template_direction(after_rows, player_templates, map_mask, state_rows, config))
        all_points.sort(key=lambda row: (float(row["time"]), str(row["team"]), int(row["track_slot_hint"])))

        if debug_limit > 0:
            points_by_frame: Dict[str, List[Point]] = {}
            for point in all_points:
                points_by_frame.setdefault(str(point["frame_path"]), []).append(point)
            for frame_row in valid_rows[:debug_limit]:
                frame = cv2.imread(str(resolve_path(frame_row["frame_path"])))
                if frame is None:
                    continue
                output_path = debug_dir / f"markers_{float(frame_row['time']):07.3f}s.jpg"
                draw_debug(frame, [], points_by_frame.get(frame_row["frame_path"], []), output_path)
        return all_points

    for frame_number, frame_row in enumerate(valid_rows, start=1):
        frame_path = resolve_path(frame_row["frame_path"])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        labels: List[Tuple[Box, Tuple[float, float]]] = []
        frame_points: List[Point] = []
        labels = detect_label_boxes(frame, map_mask, config)
        for team, team_config in config["teams"].items():
            frame_points.extend(detect_team_points(frame, map_mask, labels, team, team_config, config))

        for point in frame_points:
            point["match_id"] = config["match"]["id"]
            point["time"] = frame_row["time"]
            point["frame_index"] = frame_row["frame_index"]
            point["frame_path"] = frame_row["frame_path"]
        all_points.extend(frame_points)

        if frame_number <= debug_limit:
            output_path = debug_dir / f"markers_{float(frame_row['time']):07.3f}s.jpg"
            draw_debug(frame, labels, frame_points, output_path)

    return all_points


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    points = detect_markers(config, args.limit)
    write_points(resolve_path(config["outputs"]["raw_points_csv"]), points)
    print(f"raw points: {len(points)}")
    print(f"raw points csv: {resolve_path(config['outputs']['raw_points_csv'])}")
    print(f"debug markers dir: {resolve_path(config['outputs']['debug_markers_dir'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
