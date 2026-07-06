from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frame_iter(
    input_path: Path,
    sample_fps: float,
    every_frame: bool,
    start_seconds: float,
    stop_seconds: Optional[float],
):
    if is_image(input_path):
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {input_path}")
        yield 0, 0.0, frame
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = 1 if every_frame else max(1, int(round(source_fps / sample_fps)))
    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
    frame_index = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_index = max(pos_frame, frame_index + 1)
            if frame_index % interval != 0:
                continue
            elapsed = round(frame_index / source_fps, 3)
            if stop_seconds is not None and elapsed > stop_seconds:
                break
            yield frame_index, elapsed, frame
    finally:
        cap.release()
