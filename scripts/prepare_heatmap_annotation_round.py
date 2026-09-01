from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY
from src.heatmap.annotation_ui import build_annotation_ui
from src.heatmap.annotation_round import build_annotation_round_report, render_markdown
from src.report_io import write_json_report, write_text_report


DEFAULT_CONFIG = ROOT / "config" / "annotation_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a named heatmap manual annotation round.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--round-id", default="first_manual_loop")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "outputs" / "heatmap_annotation_round1")
    parser.add_argument("--annotation-csv", type=Path, help="Existing filled annotation CSV to evaluate.")
    parser.add_argument("--match-id", action="append", default=[], help="Override round match ids. May be repeated.")
    parser.add_argument("--frames-per-match", type=int, help="Override frames exported per match.")
    parser.add_argument("--threshold-px", type=float, help="Override annotation matching threshold.")
    parser.add_argument("--min-labeled-rows", type=int, help="Minimum manual labels required for a passed progress gate.")
    parser.add_argument("--min-complete-groups", type=int, help="Minimum complete frame/team groups required for a passed progress gate.")
    parser.add_argument("--build-ui", action="store_true", help="Also build the static annotation HTML helper for this round.")
    parser.add_argument("--ui-output", type=Path, help="Annotation UI output path. Defaults inside --package-dir.")
    parser.add_argument("--priority-limit", type=int, help="Move the top N priority tasks to the beginning of the annotation UI.")
    parser.add_argument("--no-export", action="store_true", help="Only summarize/evaluate an existing package.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_annotation_round1.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_annotation_round1.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless progress gates pass and labels exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_annotation_round_report(
        registry_path=args.registry,
        config_path=args.config,
        round_id=args.round_id,
        package_dir=args.package_dir.expanduser(),
        match_ids=args.match_id or None,
        frames_per_match=args.frames_per_match,
        export_package=not args.no_export,
        annotation_csv=args.annotation_csv.expanduser() if args.annotation_csv else None,
        threshold_px=args.threshold_px,
        min_labeled_rows=args.min_labeled_rows,
        min_complete_groups=args.min_complete_groups,
    )
    if args.build_ui:
        package_dir = args.package_dir.expanduser()
        annotation_csv = args.annotation_csv.expanduser() if args.annotation_csv else package_dir / "annotation_template.csv"
        ui_output = args.ui_output.expanduser() if args.ui_output else package_dir / "annotation_ui.html"
        report["annotation_ui"] = build_annotation_ui(
            annotation_csv,
            ui_output,
            title=f"热力图人工标注：{args.round_id}",
            priority_limit=args.priority_limit,
        )
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"heatmap annotation round status: {report['status']}")
    failed_checks = any(not check["ok"] for check in report.get("progress_checks", {}).values())
    if args.strict and (report["status"] == "needs_labels" or failed_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
