from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np

from src.detection import by_class, center_x, crop_result, detections


MESSAGE_CHARS = {
    0: "1",
    1: "2",
    2: "3",
    3: "ア",
    4: "ば",
    5: "バ",
    6: "チ",
    7: "ちゅう",
    8: "だ",
    9: "第",
    10: "ど",
    11: "ド",
    12: "エ",
    13: "防",
    14: "が",
    15: "ガ",
    16: "グ",
    17: "保",
    18: "ホ",
    19: "い",
    20: "カ",
    21: "確",
    22: "け",
    23: "こ",
    24: "コ",
    25: "みな",
    26: "も",
    27: "モ",
    28: "ン",
    29: "に",
    30: "お",
    31: "おう",
    32: "破",
    33: "プ",
    34: "ラ",
    35: "れ",
    36: "リ",
    37: "る",
    38: "さ",
    39: "し",
    40: "ス",
    41: "スー",
    42: "た",
    43: "ト",
    44: "突",
    45: "到",
    46: "っ",
    47: "ツ",
    48: "着",
    49: "う",
    50: "ウ",
    51: "失",
    52: "わ",
    53: "を",
    54: "ヲ",
    55: "ヤ",
}


def ocr_number(ocr_model, img: np.ndarray, min_digit_conf: float) -> str:
    if img.size == 0:
        return ""
    results = ocr_model(img, 64)
    arr = detections(results)
    arr = arr[(arr[:, 5] < 11) & (arr[:, 4] >= min_digit_conf)]
    arr = arr[np.argsort(arr[:, 0])]
    return "".join(str(results.names[int(row[5])]) for row in arr)


def parse_number(text: str, upper_bound: int) -> Optional[int]:
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    value = int(digits)
    if 0 <= value < upper_bound:
        return value
    return None


def side_numbers(
    results,
    boxes: np.ndarray,
    center: float,
    ocr_model,
    upper_bound: int,
    min_box_conf: float,
    min_digit_conf: float,
) -> List[Optional[int]]:
    output: List[Optional[int]] = [None, None]
    boxes = boxes[boxes[:, 4] >= min_box_conf]
    for box in boxes[np.argsort(boxes[:, 0])]:
        value = parse_number(ocr_number(ocr_model, crop_result(results, box), min_digit_conf), upper_bound)
        if value is None:
            continue
        midpoint = (box[0] + box[2]) / 2.0
        output[0 if midpoint < center else 1] = value
    return output


def count_numbers(
    results,
    arr: np.ndarray,
    ids: Dict[str, int],
    ocr_model,
    min_box_conf: float,
    min_digit_conf: float,
) -> List[Optional[int]]:
    moving = by_class(arr, ids["moving_count"])
    fixed = by_class(arr, ids["fixed_count"])
    boxes = np.concatenate([moving, fixed]) if len(moving) or len(fixed) else np.zeros((0, 6), dtype=float)
    if len(boxes) == 0:
        return [None, None]
    return side_numbers(results, boxes, center_x(results, arr, ids), ocr_model, 101, min_box_conf, min_digit_conf)


def penalty_numbers(
    results,
    arr: np.ndarray,
    ids: Dict[str, int],
    ocr_model,
    min_box_conf: float,
    min_digit_conf: float,
) -> List[Optional[int]]:
    boxes = by_class(arr, ids["penalty"])
    if len(boxes) == 0:
        return [None, None]
    return side_numbers(results, boxes, center_x(results, arr, ids), ocr_model, 100, min_box_conf, min_digit_conf)


def message_text(message_model, img: np.ndarray, min_char_conf: float) -> str:
    if img.size == 0:
        return ""
    results = message_model(img, 640)
    arr = detections(results)
    arr = arr[arr[:, 4] >= min_char_conf]
    arr = arr[np.argsort(arr[:, 0])]
    chars = []
    for row in arr:
        idx = int(row[5])
        if hasattr(results.names, "get"):
            fallback = results.names.get(idx, "")
        else:
            fallback = results.names[idx] if idx < len(results.names) else ""
        chars.append(MESSAGE_CHARS.get(idx, str(fallback)))
    return "".join(chars)


def first_image_for_class(results, arr: np.ndarray, cls_id: int, min_conf: float = 0.0) -> Optional[np.ndarray]:
    boxes = by_class(arr, cls_id)
    boxes = boxes[boxes[:, 4] >= min_conf]
    if len(boxes) == 0:
        return None
    return crop_result(results, boxes[0])
