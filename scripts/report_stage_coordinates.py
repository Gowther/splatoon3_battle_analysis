from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import resolve_project_path
from src.heatmap.config_loader import load_config, resolve_path
from src.heatmap.stage_coordinates import build_stage_coordinate_report, render_markdown, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report and optionally export normalized heatmap stage coordinates.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml", help="Heatmap config YAML.")
    parser.add_argument("--points-csv", type=Path, help="CSV with x/y columns. Defaults to player_tracks or team_points.")
    parser.add_argument("--normalized-output", type=Path, help="Optional CSV output with stage_x/stage_y columns.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage_coordinates.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "stage_coordinates.json")
    return parser.parse_args()


def default_points_csv(config: dict) -> Path | None:
    outputs = config.get("outputs", {})
    if not isinstance(outputs, dict):
        return None
    value = outputs.get("player_tracks_csv") or outputs.get("clean_points_csv")
    return resolve_project_path(value) if value else None


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    points_csv = args.points_csv or default_points_csv(config)
    report = build_stage_coordinate_report(config, points_csv=points_csv, normalized_csv=args.normalized_output)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(render_markdown(report), encoding="utf-8")
    write_json(args.json_output.expanduser(), report)
    print(f"stage coordinate status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
