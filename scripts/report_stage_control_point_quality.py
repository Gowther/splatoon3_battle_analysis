from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.config_loader import load_config, resolve_path
from src.heatmap.stage_coordinates import (
    DEFAULT_REPROJECTION_TOLERANCE,
    discover_control_point_asset,
    load_control_point_asset,
)
from src.heatmap.stage_quality import (
    DEFAULT_MAX_CORNER_EXCURSION,
    DEFAULT_MAX_FRAME_DRIFT,
    DEFAULT_MIN_COVERAGE,
    build_control_point_quality_report,
    render_quality_markdown,
)
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate stage control-point quality: reprojection, ROI coverage, corner sanity, and cross-frame drift."
    )
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml", help="Heatmap config YAML.")
    parser.add_argument("--control-points", type=Path, help="Control-point asset. Defaults to the discovered asset for this config.")
    parser.add_argument(
        "--labeled-frames",
        type=Path,
        help="Optional JSON mapping frame id to a control_points list, used for cross-frame drift.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_REPROJECTION_TOLERANCE)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--max-corner-excursion", type=float, default=DEFAULT_MAX_CORNER_EXCURSION)
    parser.add_argument("--max-frame-drift", type=float, default=DEFAULT_MAX_FRAME_DRIFT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage_control_point_quality.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "stage_control_point_quality.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every check passes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))

    if args.control_points:
        asset = load_control_point_asset(args.control_points)
    else:
        asset = discover_control_point_asset(config)
        if asset is None:
            print("no promoted control-point asset found for this config")
            print("label control points in the /stage-labeling page, or pass --control-points")
            return 1 if args.strict else 0

    labeled_frames = {}
    if args.labeled_frames:
        labeled_frames = json.loads(args.labeled_frames.expanduser().read_text(encoding="utf-8"))

    report = build_control_point_quality_report(
        config,
        asset,
        labeled_frames=labeled_frames,
        tolerance=args.tolerance,
        min_coverage=args.min_coverage,
        max_excursion=args.max_corner_excursion,
        max_drift=args.max_frame_drift,
    )

    write_text_report(args.output.expanduser(), render_quality_markdown(report))
    write_json_report(args.json_output.expanduser(), report)

    print(f"stage control point quality: {report['status']}")
    for name in report["failed_checks"]:
        check = report["checks"][name]
        print(f"- {name}: {check.get('status')}")
    for blocker in report["blockers"]:
        print(f"- {blocker}")
    return strict_exit_code(report["status"], args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
