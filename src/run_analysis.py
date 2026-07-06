from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from src.core.paths import ROOT, configure_environment, default_output_path, model_path, project_path

configure_environment()

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch\\.cuda\\.amp\\.autocast.*")

import cv2

from src.detection import (
    by_class,
    class_ids,
    choose_device,
    detections,
    draw_preview,
    load_yolo_model,
    player_lamps,
    torch_load,
)
from src.media import frame_iter
from src.ocr import count_numbers, first_image_for_class, message_text, penalty_numbers
from src.weapons import ImageTransform, classify_weapons, load_weapon_names, vote_weapons, weapon_model_output_count


CSV_HEADER = [
    "elapsed_time",
    "player_state_1",
    "player_state_2",
    "player_state_3",
    "player_state_4",
    "player_state_5",
    "player_state_6",
    "player_state_7",
    "player_state_8",
    "count_left",
    "count_right",
    "penalty_left",
    "penalty_right",
    "weapon_1",
    "weapon_2",
    "weapon_3",
    "weapon_4",
    "weapon_5",
    "weapon_6",
    "weapon_7",
    "weapon_8",
    "stage",
    "asari_count",
    "hoko_count",
    "area_count",
    "yagura_count",
    "message",
    "player_detected",
    "reserved_28",
    "timestamp",
    "reserved_30",
    "reserved_31",
    "reserved_32",
]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Splatoon 3 footage.")
    parser.add_argument("--input", required=True, help="Image or video file to analyze.")
    parser.add_argument("--output", help="CSV output path. Defaults to outputs/<input>_<timestamp>.csv.")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Frames per second to analyze for videos.")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Skip input before this timestamp.")
    parser.add_argument("--stop-seconds", type=float, help="Stop analyzing after this timestamp.")
    parser.add_argument("--every-frame", action="store_true", help="Analyze every video frame.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many analyzed frames.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Valid 8-player frames used for weapon voting.")
    parser.add_argument("--preview", action="store_true", help="Show an OpenCV preview window.")
    parser.add_argument("--preview-scale", type=float, default=0.5, help="Preview resize scale.")
    parser.add_argument("--save-preview-dir", help="Save annotated preview frames to this directory.")
    parser.add_argument("--save-preview-limit", type=int, default=20, help="Maximum annotated frames to save.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.3, help="YOLO NMS IoU threshold.")
    parser.add_argument("--count-box-conf", type=float, default=0.5, help="Minimum count/penalty box confidence.")
    parser.add_argument("--digit-conf", type=float, default=0.5, help="Minimum OCR digit confidence.")
    parser.add_argument("--message-box-conf", type=float, default=0.5, help="Minimum message box confidence.")
    parser.add_argument("--message-char-conf", type=float, default=0.55, help="Minimum message OCR character confidence.")
    parser.add_argument("--no-header", action="store_true", help="Do not write a CSV header row.")
    parser.add_argument("--list-model-names", action="store_true", help="Print loaded model class names and exit.")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    input_path = project_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = project_path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    detect_model = load_yolo_model(model_path("the_model.pt"), device, args.conf, args.iou)
    ocr_model = load_yolo_model(model_path("ocr_model.pt"), device, args.conf, args.iou)
    message_model = load_yolo_model(model_path("message_ocr_model.pt"), device, args.conf, args.iou)
    ids = class_ids(detect_model.names, REQUIRED_DETECTION_CLASSES)

    if args.list_model_names:
        print("detection:", detect_model.names)
        print("number_ocr:", ocr_model.names)
        print("message_ocr:", message_model.names)
        return 0

    weapon_names = load_weapon_names(ROOT / "main_weapon_list.txt")
    weapon_model = torch_load(model_path("main_weapons_classification_weight.pth"), device)
    weapon_model.eval()
    weapon_output_count = weapon_model_output_count(weapon_model)
    if weapon_output_count is not None and weapon_output_count != len(weapon_names):
        print(
            "Warning: weapon model output count "
            f"({weapon_output_count}) does not match main_weapon_list.txt ({len(weapon_names)})."
        )
    transform = ImageTransform()

    rows: List[List[object]] = []
    weapon_votes: List[Optional[List[str]]] = []
    final_weapons: Optional[List[str]] = None
    analyzed = 0
    saved_previews = 0
    analysis_time = dt.datetime.now().isoformat(timespec="seconds")
    save_preview_dir = Path(args.save_preview_dir).expanduser() if args.save_preview_dir else None
    if save_preview_dir and not save_preview_dir.is_absolute():
        save_preview_dir = ROOT / save_preview_dir
    if save_preview_dir:
        save_preview_dir.mkdir(parents=True, exist_ok=True)

    for _, elapsed, frame_bgr in frame_iter(
        input_path,
        args.sample_fps,
        args.every_frame,
        args.start_seconds,
        args.stop_seconds,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detect_model(frame_rgb, 640)

        if final_weapons is None and len(weapon_votes) < args.warmup_frames:
            vote = classify_weapons(results, weapon_model, weapon_names, device, transform, ids)
            if vote:
                weapon_votes.append(vote)
                print(f"Weapon warmup frame {len(weapon_votes)}/{args.warmup_frames}: {vote}")
            if len(weapon_votes) >= args.warmup_frames:
                final_weapons = vote_weapons(weapon_votes)
                print(f"Weapon warmup complete: {final_weapons}")

        rows.append(
            analyze_results(
                results,
                elapsed,
                analysis_time,
                ids,
                ocr_model,
                message_model,
                final_weapons,
                args.count_box_conf,
                args.digit_conf,
                args.message_box_conf,
                args.message_char_conf,
            )
        )
        analyzed += 1

        if args.preview or save_preview_dir:
            preview = draw_preview(frame_bgr, results, detect_model.names)

        if save_preview_dir and saved_previews < args.save_preview_limit:
            cv2.imwrite(str(save_preview_dir / f"frame_{analyzed:05d}_{elapsed:.1f}s.jpg"), preview)
            saved_previews += 1

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

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not args.no_header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"Analyzed frames: {analyzed}")
    print(f"Wrote CSV: {output_path}")
    if final_weapons is None:
        print("Weapon warmup did not complete; CSV weapon fields may be empty.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
