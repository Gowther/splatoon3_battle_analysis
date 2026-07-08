from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.analysis_preview import PreviewSaveState, maybe_save_preview
from src.analysis_runtime import AnalysisRunResult, preview_dir_from_arg
from src.analysis_warmup import WeaponWarmupState, update_weapon_warmup
from src.core.paths import ROOT, model_path
from src.detection import (
    by_class,
    class_ids,
    detections,
    draw_preview,
    load_yolo_model,
    player_lamps,
    torch_load,
)
from src.media import frame_iter
from src.ocr import count_numbers, first_image_for_class, message_text, penalty_numbers
from src.weapons import ImageTransform, load_weapon_names, weapon_model_output_count


REQUIRED_DETECTION_CLASSES = [
    "alive",
    "dead",
    "special",
    "moving_count",
    "fixed_count",
    "penalty",
    "message",
    "asari_object",
    "hoko_canmon",
    "area_object",
    "yagura_kanmon",
    "player",
]


@dataclass
class DetectionModels:
    detect_model: Any
    ocr_model: Any
    message_model: Any
    ids: Dict[str, int]


@dataclass
class WeaponRuntime:
    names: List[str]
    model: Any
    transform: ImageTransform


def load_detection_models(args: argparse.Namespace, device: str) -> DetectionModels:
    detect_model = load_yolo_model(model_path("the_model.pt"), device, args.conf, args.iou)
    ocr_model = load_yolo_model(model_path("ocr_model.pt"), device, args.conf, args.iou)
    message_model = load_yolo_model(model_path("message_ocr_model.pt"), device, args.conf, args.iou)
    ids = class_ids(detect_model.names, REQUIRED_DETECTION_CLASSES)
    return DetectionModels(detect_model, ocr_model, message_model, ids)


def print_model_names(models: DetectionModels) -> None:
    print("detection:", models.detect_model.names)
    print("number_ocr:", models.ocr_model.names)
    print("message_ocr:", models.message_model.names)


def load_weapon_runtime(device: str) -> WeaponRuntime:
    weapon_names = load_weapon_names(ROOT / "main_weapon_list.txt")
    weapon_model = torch_load(model_path("main_weapons_classification_weight.pth"), device)
    weapon_model.eval()
    weapon_output_count = weapon_model_output_count(weapon_model)
    if weapon_output_count is not None and weapon_output_count != len(weapon_names):
        print(
            "Warning: weapon model output count "
            f"({weapon_output_count}) does not match main_weapon_list.txt ({len(weapon_names)})."
        )
    return WeaponRuntime(weapon_names, weapon_model, ImageTransform())


def analyze_results(
    results,
    elapsed_time: Optional[float],
    analysis_time: str,
    ids: Dict[str, int],
    ocr_model,
    message_model,
    weapon_list: Optional[List[str]],
    count_box_conf: float,
    digit_conf: float,
    message_box_conf: float,
    message_char_conf: float,
) -> List[object]:
    arr = detections(results)
    row: List[object] = [None] * 33
    row[0] = elapsed_time
    row[29] = analysis_time

    lamps = player_lamps(arr, ids)
    if len(lamps) == 8:
        for i, lamp in enumerate(lamps):
            row[1 + i] = int(lamp[5])

    if weapon_list:
        for i, weapon in enumerate(weapon_list[:8]):
            row[13 + i] = weapon

    row[22] = int(len(by_class(arr, ids["asari_object"])))
    row[23] = int(len(by_class(arr, ids["hoko_canmon"])))
    row[24] = int(len(by_class(arr, ids["area_object"])))
    row[25] = int(len(by_class(arr, ids["yagura_kanmon"])))
    row[27] = bool(len(by_class(arr, ids["player"])) > 0) or None

    counts = count_numbers(results, arr, ids, ocr_model, count_box_conf, digit_conf)
    row[9], row[10] = counts

    penalties = penalty_numbers(results, arr, ids, ocr_model, count_box_conf, digit_conf)
    row[11], row[12] = penalties

    message_img = first_image_for_class(results, arr, ids["message"], message_box_conf)
    if message_img is not None:
        message = message_text(message_model, message_img, message_char_conf)
        row[26] = message or None

    return row


def analyze_frame_stream(
    args: argparse.Namespace,
    input_path: Path,
    device: str,
    models: DetectionModels,
    weapons: WeaponRuntime,
) -> AnalysisRunResult:
    rows: List[List[object]] = []
    weapon_warmup = WeaponWarmupState()
    analyzed = 0
    analysis_time = dt.datetime.now().isoformat(timespec="seconds")
    preview_save = PreviewSaveState(preview_dir_from_arg(args.save_preview_dir), args.save_preview_limit)

    for _, elapsed, frame_bgr in frame_iter(
        input_path,
        args.sample_fps,
        args.every_frame,
        args.start_seconds,
        args.stop_seconds,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = models.detect_model(frame_rgb, 640)
        weapon_warmup = update_weapon_warmup(
            results,
            warmup_frames=args.warmup_frames,
            detection_ids=models.ids,
            weapon_model=weapons.model,
            weapon_names=weapons.names,
            weapon_transform=weapons.transform,
            device=device,
            state=weapon_warmup,
        )

        rows.append(
            analyze_results(
                results,
                elapsed,
                analysis_time,
                models.ids,
                models.ocr_model,
                models.message_model,
                weapon_warmup.final_weapons,
                args.count_box_conf,
                args.digit_conf,
                args.message_box_conf,
                args.message_char_conf,
            )
        )
        analyzed += 1

        if args.preview or preview_save.enabled:
            preview = draw_preview(frame_bgr, results, models.detect_model.names)

        if preview_save.can_save:
            maybe_save_preview(preview, preview_save, analyzed, elapsed)

        if args.preview:
            if args.preview_scale != 1.0:
                preview = cv2.resize(preview, None, fx=args.preview_scale, fy=args.preview_scale)
            cv2.imshow("splatoon3 analysis", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.max_frames and analyzed >= args.max_frames:
            break

    if args.preview:
        cv2.destroyAllWindows()

    return AnalysisRunResult(rows, analyzed, weapon_warmup.final_weapons)
