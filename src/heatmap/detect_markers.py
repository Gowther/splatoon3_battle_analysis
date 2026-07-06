from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.heatmap.extract_frames import load_config, resolve_path, team_mask


Point = Dict[str, object]
Box = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect team marker candidates on overhead-map frames.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--limit", type=int, help="Only process the first N valid frames.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_points(path: Path, rows: Sequence[Point]) -> None:
    fieldnames = [
        "match_id",
        "time",
        "frame_index",
        "team",
        "player_id",
        "x",
        "y",
        "confidence",
        "source",
        "area",
        "label_distance",
        "frame_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        cv2.circle(output, (x, y), 12, color, 2)
        cv2.putText(
            output,
            f"{point['team']} {float(point['confidence']):.2f}",
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
    all_points: List[Point] = []

    if debug_limit > 0 and debug_dir.exists():
        for old_debug in debug_dir.glob("markers_*.jpg"):
            old_debug.unlink()

    for frame_number, frame_row in enumerate(valid_rows, start=1):
        frame_path = resolve_path(frame_row["frame_path"])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        labels = detect_label_boxes(frame, map_mask, config)
        frame_points: List[Point] = []
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
