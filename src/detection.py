from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
import torch

from src.core.paths import ROOT


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("Requested MPS, but torch reports it is unavailable. Falling back to CPU.", file=sys.stderr)
            return "cpu"
        return "mps"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_yolo_model(path: Path, device: str, conf: float, iou: float):
    model = torch.hub.load(
        str(ROOT / "yolov5"),
        "custom",
        path=str(path),
        source="local",
        device=device,
        _verbose=False,
    )
    model.conf = conf
    model.iou = iou
    return model


def torch_load(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def class_ids(names: Dict[int, str], required: Sequence[str]) -> Dict[str, int]:
    reverse = {name: idx for idx, name in names.items()}
    missing = [name for name in required if name not in reverse]
    if missing:
        raise ValueError(f"Detection model is missing required classes: {', '.join(missing)}")
    return {name: reverse[name] for name in required}


def detections(results) -> np.ndarray:
    arr = results.xyxy[0].detach().cpu().numpy()
    if arr.size == 0:
        return np.zeros((0, 6), dtype=float)
    return arr


def by_class(arr: np.ndarray, cls_id: int) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros((0, 6), dtype=float)
    return arr[arr[:, 5] == cls_id]


def player_lamps(arr: np.ndarray, ids: Dict[str, int]) -> np.ndarray:
    lamps = [
        by_class(arr, ids["alive"]),
        by_class(arr, ids["dead"]),
        by_class(arr, ids["special"]),
    ]
    non_empty = [lamp for lamp in lamps if len(lamp) > 0]
    if not non_empty:
        return np.zeros((0, 6), dtype=float)
    combined = np.concatenate(non_empty)
    return combined[np.argsort(combined[:, 0])]


def image_width(results) -> int:
    return int(results.ims[0].shape[1])


def center_x(results, arr: np.ndarray, ids: Dict[str, int]) -> float:
    lamps = player_lamps(arr, ids)
    if len(lamps) == 8:
        return float(np.sum(lamps[:, 0] + lamps[:, 2]) / 16.0)
    return image_width(results) / 2.0


def crop_result(results, box: Sequence[float]) -> np.ndarray:
    img = results.ims[0]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 = max(0, min(x1, img.shape[1]))
    x2 = max(0, min(x2, img.shape[1]))
    y1 = max(0, min(y1, img.shape[0]))
    y2 = max(0, min(y2, img.shape[0]))
    return img[y1:y2, x1:x2]


def draw_preview(frame_bgr: np.ndarray, results, names: Dict[int, str]) -> np.ndarray:
    output = frame_bgr.copy()
    for x1, y1, x2, y2, conf, cls in detections(results):
        x1, y1, x2, y2 = [int(v) for v in (x1, y1, x2, y2)]
        label = f"{names.get(int(cls), int(cls))} {conf:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output
