from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.config_loader import load_config, resolve_path
from src.heatmap.render_stage_space import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_MARGIN,
    render_markdown,
    render_stage_heatmaps,
)
from src.heatmap.run_pipeline import stage_tracks_csv_path
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed-canvas stage-space heatmaps and routes from normalized stage coordinates."
    )
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml", help="Heatmap config YAML.")
    parser.add_argument("--stage-csv", type=Path, help="Stage-coordinate CSV. Defaults to the pipeline stage tracks output.")
    parser.add_argument("--output-dir", type=Path, help="Image output directory. Defaults to <output_dir>/rendered_stage.")
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--output", type=Path, help="Optional Markdown report.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless images were rendered.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    stage_csv = args.stage_csv or stage_tracks_csv_path(config)
    output_dir = args.output_dir or (resolve_path(config["match"]["output_dir"]) / "rendered_stage")

    report = render_stage_heatmaps(
        stage_csv,
        config,
        output_dir,
        canvas_size=args.canvas_size,
        margin=args.margin,
    )

    if args.output:
        write_text_report(args.output.expanduser(), render_markdown(report))
    if args.json_output:
        write_json_report(args.json_output.expanduser(), report)

    print(f"stage rendering: {report['status']}")
    if report["status"] == "ready":
        print(f"rendered {len(report['rendered'])} images from {report['stage_points']} points -> {report['output_dir']}")
    else:
        print(f"no stage coordinates in {report['input']}")
        print("promote a control-point asset and re-run the heatmap pipeline first")
    return strict_exit_code(report["status"], args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
