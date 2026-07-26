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
    build_control_point_asset,
    load_control_point_asset,
    render_control_point_markdown,
    roi_corner_control_points,
    stage_box_from_config,
    validate_control_point_asset,
)
from src.report_io import strict_exit_code, write_json_report, write_text_report


DEFAULT_ASSET_DIR = ROOT / "config" / "stage_control_points"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate a stage control-point asset for heatmap homography.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml", help="Heatmap config YAML providing the map ROI.")
    parser.add_argument("--stage-id", help="Stage identifier. Required unless --validate is used.")
    parser.add_argument("--control-points", type=Path, help="Existing asset to validate instead of generating one.")
    parser.add_argument("--points-json", type=Path, help="JSON file with a control_points list to build the asset from.")
    parser.add_argument("--output", type=Path, help="Asset output path. Defaults to config/stage_control_points/<stage_id>.json.")
    parser.add_argument("--report", type=Path, help="Optional Markdown validation report.")
    parser.add_argument("--json-report", type=Path, help="Optional JSON validation report.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_REPROJECTION_TOLERANCE, help="Max allowed reprojection error.")
    parser.add_argument("--validate", action="store_true", help="Only validate, never write an asset.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing asset file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the asset is ready.")
    return parser.parse_args()


def load_points_json(path: Path) -> list[dict]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    points = payload.get("control_points", payload) if isinstance(payload, dict) else payload
    if not isinstance(points, list):
        raise SystemExit("points JSON must contain a control_points list")
    return points


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    source_box = stage_box_from_config(config)

    if args.control_points:
        asset = load_control_point_asset(args.control_points)
        asset = build_control_point_asset(
            asset.get("stage_id", ""),
            asset.get("control_points", []),
            template=bool(asset.get("template")),
            notes=asset.get("notes") or [],
        )
        output: Path | None = None
    else:
        if not args.stage_id:
            raise SystemExit("--stage-id is required when generating an asset")
        points = load_points_json(args.points_json) if args.points_json else roi_corner_control_points(source_box)
        notes = (
            []
            if args.points_json
            else [
                "Seeded from the config map ROI corners, which only reproduces linear normalization.",
                "Replace each source point with a visible stage landmark before trusting the homography.",
            ]
        )
        asset = build_control_point_asset(args.stage_id, points, notes=notes)
        output = (args.output or DEFAULT_ASSET_DIR / f"{args.stage_id}.json").expanduser()

    report = validate_control_point_asset(asset, source_box=source_box, tolerance=args.tolerance)

    if output is not None and not args.validate:
        if output.exists() and not args.force:
            raise SystemExit(f"{output} already exists; pass --force to overwrite")
        write_json_report(output, asset)

    if args.report:
        write_text_report(args.report.expanduser(), render_control_point_markdown(report))
    if args.json_report:
        write_json_report(args.json_report.expanduser(), report)

    print(f"control point status: {report['status']}")
    for error in report["errors"]:
        print(f"- {error}")
    return strict_exit_code(report["status"], args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
