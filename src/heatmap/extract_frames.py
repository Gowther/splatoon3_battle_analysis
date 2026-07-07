from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import cv2
import numpy as np

from src.heatmap.config_loader import load_config


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract overhead-map frames for heatmap analysis.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--no-save-frames", action="store_true", help="Only write CSV/contact output.")
    parser.add_argument("--contact-limit", type=int, default=36, help="Maximum frames shown in the contact sheet.")
    return parser.parse_args()


def resolve_path(path: Union[str, Path]) -> Path:
    output = Path(path).expanduser()
    return output if output.is_absolute() else ROOT / output


def seconds_range(start: float, stop: float, sample_fps: float) -> Iterable[float]:
    step = 1.0 / sample_fps
    count = int(math.floor((stop - start) / step)) + 1
    for i in range(max(0, count)):
        yield round(start + i * step, 3)


def in_ranges(value: float, ranges: Sequence[Sequence[float]]) -> bool:
    return any(float(start) <= value <= float(stop) for start, stop in ranges)


def clip_box(box: Dict[str, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(box["x1"]), width))
    y1 = max(0, min(int(box["y1"]), height))
    x2 = max(0, min(int(box["x2"]), width))
    y2 = max(0, min(int(box["y2"]), height))
    return x1, y1, x2, y2


def apply_excludes(mask: np.ndarray, roi: Tuple[int, int, int, int], exclude_regions: Sequence[Dict[str, int]]) -> None:
    rx1, ry1, rx2, ry2 = roi
    for region in exclude_regions:
        x1, y1, x2, y2 = clip_box(region, rx2, ry2)
        ix1, iy1 = max(x1, rx1), max(y1, ry1)
        ix2, iy2 = min(x2, rx2), min(y2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        mask[iy1 - ry1 : iy2 - ry1, ix1 - rx1 : ix2 - rx1] = 0


def team_mask(hsv_roi: np.ndarray, ranges: Sequence[Dict[str, Sequence[int]]]) -> np.ndarray:
    output = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    for color_range in ranges:
        lower = np.array(color_range["lower"], dtype=np.uint8)
        upper = np.array(color_range["upper"], dtype=np.uint8)
        output = cv2.bitwise_or(output, cv2.inRange(hsv_roi, lower, upper))
    return output


def quality_metrics(frame: np.ndarray, config: Dict) -> Dict[str, float]:
    height, width = frame.shape[:2]
    roi = clip_box(config["map_view"]["roi"], width, height)
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]
    if roi_frame.size == 0:
        return {"yellow_ratio": 0.0, "blue_ratio": 0.0, "ink_ratio": 0.0}

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    exclude_regions = config["map_view"].get("exclude_regions", [])
    masks: Dict[str, np.ndarray] = {}
    for team, team_config in config["teams"].items():
        mask = team_mask(hsv, team_config["hsv_ranges"])
        apply_excludes(mask, roi, exclude_regions)
        masks[team] = mask

    area = float(roi_frame.shape[0] * roi_frame.shape[1])
    team_names = list(config["teams"].keys())
    ratios = [
        cv2.countNonZero(masks.get(team, np.zeros_like(hsv[:, :, 0]))) / area
        for team in team_names
    ]
    yellow_ratio = ratios[0] if len(ratios) > 0 else 0.0
    blue_ratio = ratios[1] if len(ratios) > 1 else 0.0
    return {
        "yellow_ratio": round(yellow_ratio, 6),
        "blue_ratio": round(blue_ratio, 6),
        "ink_ratio": round(sum(ratios), 6),
    }


def classify_frame(time_seconds: float, metrics: Dict[str, float], config: Dict) -> Tuple[bool, str]:
    invalid_ranges = config["frame_quality"].get("invalid_ranges_seconds", [])
    if in_ranges(time_seconds, invalid_ranges):
        return False, "configured_invalid_range"
    if metrics["ink_ratio"] < float(config["frame_quality"]["min_ink_ratio"]):
        return False, "low_team_color_ratio"
    return True, "map_view"


def annotate_frame(frame: np.ndarray, time_seconds: float, metrics: Dict[str, float], valid: bool, reason: str) -> np.ndarray:
    output = frame.copy()
    color = (0, 220, 0) if valid else (0, 0, 255)
    lines = [
        f"{time_seconds:.1f}s {'valid' if valid else 'invalid'}",
        f"ink={metrics['ink_ratio']:.3f} y={metrics['yellow_ratio']:.3f} b={metrics['blue_ratio']:.3f}",
        reason,
    ]
    y = 46
    for line in lines:
        cv2.putText(output, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        y += 38
    return output


def make_contact_sheet(image_paths: Sequence[Path], output_path: Path, limit: int) -> None:
    selected = list(image_paths[: max(0, limit)])
    if not selected:
        return
    thumbs: List[np.ndarray] = []
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height = 220
        width = int(image.shape[1] * height / image.shape[0])
        thumbs.append(cv2.resize(image, (width, height)))
    if not thumbs:
        return

    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    cell_h = max(image.shape[0] for image in thumbs)
    cell_w = max(image.shape[1] for image in thumbs)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for index, image in enumerate(thumbs):
        row, col = divmod(index, cols)
        y = row * cell_h
        x = col * cell_w
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "match_id",
        "time",
        "frame_index",
        "frame_path",
        "valid",
        "reason",
        "ink_ratio",
        "yellow_ratio",
        "blue_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_frames(config: Dict, save_frames: bool, contact_limit: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    input_video = resolve_path(config["match"]["input_video"])
    if not input_video.exists():
        raise FileNotFoundError(input_video)

    frames_dir = resolve_path(config["outputs"]["frames_dir"])
    probes_dir = resolve_path(config["outputs"]["probes_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    probes_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    valid_rows: List[Dict[str, object]] = []
    invalid_rows: List[Dict[str, object]] = []
    saved_valid_paths: List[Path] = []
    try:
        for time_seconds in seconds_range(
            float(config["sampling"]["start_seconds"]),
            float(config["sampling"]["stop_seconds"]),
            float(config["sampling"]["sample_fps"]),
        ):
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            metrics = quality_metrics(frame, config)
            valid, reason = classify_frame(time_seconds, metrics, config)
            row: Dict[str, object] = {
                "match_id": config["match"]["id"],
                "time": f"{time_seconds:.3f}",
                "frame_index": frame_index,
                "frame_path": "",
                "valid": valid,
                "reason": reason,
                **metrics,
            }

            if save_frames:
                annotated = annotate_frame(frame, time_seconds, metrics, valid, reason)
                frame_path = frames_dir / f"frame_{time_seconds:07.3f}s.jpg"
                cv2.imwrite(str(frame_path), annotated)
                row["frame_path"] = str(frame_path.relative_to(ROOT))
                if valid:
                    saved_valid_paths.append(frame_path)

            if valid:
                valid_rows.append(row)
            else:
                invalid_rows.append(row)
    finally:
        cap.release()

    contact_path = probes_dir / "contact.jpg"
    make_contact_sheet(saved_valid_paths, contact_path, contact_limit)
    return valid_rows, invalid_rows


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    valid_rows, invalid_rows = extract_frames(config, not args.no_save_frames, args.contact_limit)
    write_csv(resolve_path(config["outputs"]["valid_frames_csv"]), valid_rows)
    write_csv(resolve_path(config["outputs"]["invalid_frames_csv"]), invalid_rows)
    print(f"valid frames: {len(valid_rows)}")
    print(f"invalid frames: {len(invalid_rows)}")
    print(f"frames dir: {resolve_path(config['outputs']['frames_dir'])}")
    print(f"contact sheet: {resolve_path(config['outputs']['probes_dir']) / 'contact.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
