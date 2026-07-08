from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, load_registry, resolve_project_path
from src.heatmap.quality_loop import build_quality_loop_report, render_markdown
from src.report_io import strict_exit_code, write_json_report, write_text_report


DEFAULT_CONFIG = ROOT / "config" / "annotation_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the heatmap annotation and evaluation quality loop report.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--package-dir", type=Path, default=ROOT / "outputs" / "heatmap_quality_loop")
    parser.add_argument("--annotation-csv", type=Path, help="Existing annotation CSV to evaluate.")
    parser.add_argument("--match-id", action="append", default=[], help="Heatmap match id to include. May be repeated.")
    parser.add_argument("--frames-per-match", type=int, help="Frames exported per match.")
    parser.add_argument("--export-package", action="store_true", help="Export annotation frames/templates before reporting.")
    parser.add_argument("--threshold-px", type=float, help="Override matching distance threshold.")
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--max-mean-error-px", type=float)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_quality_loop.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_quality_loop.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless quality loop status is passed.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    target = resolve_project_path(path) or path.expanduser()
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    frames_per_match = args.frames_per_match or int(defaults.get("frames_per_match", 5))
    threshold_px = args.threshold_px or float(defaults.get("annotation_distance_threshold_px", 80.0))
    match_ids = args.match_id or list(config.get("heatmap_matches", []))
    registry = load_registry(args.registry)

    report = build_quality_loop_report(
        registry,
        package_dir=args.package_dir.expanduser(),
        annotation_csv=args.annotation_csv.expanduser() if args.annotation_csv else None,
        frames_per_match=frames_per_match,
        match_ids=match_ids,
        export_package=args.export_package,
        threshold_px=threshold_px,
        min_recall=args.min_recall,
        max_mean_error_px=args.max_mean_error_px,
    )
    write_text_report(args.output, render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"heatmap quality loop status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"passed"})


if __name__ == "__main__":
    raise SystemExit(main())
