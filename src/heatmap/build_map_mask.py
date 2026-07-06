from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from src.heatmap.extract_frames import clip_box, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the overhead-map ROI mask and debug image.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--reference-time", type=float, help="Video timestamp used for the debug background.")
    return parser.parse_args()


def read_frame(video_path: Path, time_seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame at {time_seconds:.3f}s from {video_path}")
        return frame
    finally:
        cap.release()


def build_mask(width: int, height: int, config: Dict) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    roi = clip_box(config["map_view"]["roi"], width, height)
    x1, y1, x2, y2 = roi
    mask[y1:y2, x1:x2] = 255
    for region in config["map_view"].get("exclude_regions", []):
        ex1, ey1, ex2, ey2 = clip_box(region, width, height)
        mask[ey1:ey2, ex1:ex2] = 0
    return mask


def draw_region(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 3,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1 + 8, max(24, y1 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def debug_image(frame: np.ndarray, mask: np.ndarray, config: Dict, reference_time: float) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    overlay[mask > 0] = (70, 180, 70)
    output = cv2.addWeighted(overlay, 0.24, output, 0.76, 0)

    height, width = frame.shape[:2]
    draw_region(output, clip_box(config["map_view"]["roi"], width, height), (0, 255, 0), "map_roi")
    for region in config["map_view"].get("exclude_regions", []):
        draw_region(output, clip_box(region, width, height), (0, 0, 255), region.get("name", "exclude"), 2)

    cv2.putText(
        output,
        f"coordinate_space={config['map_view'].get('coordinate_space', 'video_pixels')} reference={reference_time:.1f}s",
        (30, height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"coordinate_space={config['map_view'].get('coordinate_space', 'video_pixels')} reference={reference_time:.1f}s",
        (30, height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    reference_time = (
        args.reference_time
        if args.reference_time is not None
        else float(config["map_view"].get("reference_time_seconds", config["sampling"]["start_seconds"]))
    )
    frame = read_frame(resolve_path(config["match"]["input_video"]), reference_time)
    height, width = frame.shape[:2]
    mask = build_mask(width, height, config)
    debug = debug_image(frame, mask, config, reference_time)

    mask_path = resolve_path(config["outputs"]["map_mask"])
    debug_path = resolve_path(config["outputs"]["map_roi_debug"])
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(debug_path), debug)

    usable_ratio = float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])
    print(f"map mask: {mask_path}")
    print(f"roi debug: {debug_path}")
    print(f"usable mask ratio: {usable_ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
