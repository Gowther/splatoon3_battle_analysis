from __future__ import annotations

import argparse
import sys
import warnings

from src.analysis_runtime import CSV_HEADER, AnalysisRunResult, preview_dir_from_arg, resolve_io_paths, write_analysis_csv
from src.analysis_pipeline import (
    REQUIRED_DETECTION_CLASSES,
    DetectionModels,
    WeaponRuntime,
    analyze_frame_stream,
    analyze_results,
    load_detection_models,
    load_weapon_runtime,
    print_model_names,
    update_weapon_warmup,
)
from src.core.paths import configure_environment

configure_environment()

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch\\.cuda\\.amp\\.autocast.*")

from src.detection import choose_device


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


def main() -> int:
    args = parse_args()
    input_path, output_path = resolve_io_paths(args)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    models = load_detection_models(args, device)

    if args.list_model_names:
        print_model_names(models)
        return 0

    weapons = load_weapon_runtime(device)
    result = analyze_frame_stream(args, input_path, device, models, weapons)
    write_analysis_csv(output_path, result.rows, include_header=not args.no_header)

    print(f"Analyzed frames: {result.analyzed}")
    print(f"Wrote CSV: {output_path}")
    if result.final_weapons is None:
        print("Weapon warmup did not complete; CSV weapon fields may be empty.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
