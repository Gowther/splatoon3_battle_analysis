from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY
from src.heatmap.annotation_round import build_annotation_round_report
from src.heatmap.annotation_ui import build_annotation_ui
from src.heatmap.labeling_workbench import (
    build_labeling_workbench_report,
    render_markdown,
    write_json,
    write_priority_queue,
)
from src.report_io import strict_exit_code, write_json_report, write_text_report


DEFAULT_CONFIG = ROOT / "config" / "annotation_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or continue a heatmap manual labeling workbench.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--round-id", default="first_manual_loop")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "outputs" / "heatmap_annotation_round1")
    parser.add_argument("--annotation-csv", type=Path, help="Existing annotation CSV. Defaults to package annotation_template.csv.")
    parser.add_argument("--match-id", action="append", default=[], help="Override round match ids. May be repeated.")
    parser.add_argument("--frames-per-match", type=int, help="Override frames exported per match.")
    parser.add_argument("--threshold-px", type=float, help="Override annotation matching threshold.")
    parser.add_argument("--priority-limit", type=int, default=24)
    parser.add_argument("--priority-queue", type=Path, help="Priority queue CSV path. Defaults inside --package-dir.")
    parser.add_argument("--ui-output", type=Path, help="Annotation UI output path. Defaults inside --package-dir.")
    parser.add_argument("--no-build-ui", action="store_true", help="Do not build the static annotation UI.")
    parser.add_argument("--refresh-package", action="store_true", help="Re-export frames and reset annotation_template.csv.")
    parser.add_argument("--manifest-output", type=Path, help="Workbench manifest JSON path. Defaults inside --package-dir.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_labeling_workbench.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_labeling_workbench.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the workbench is ready to label or tune.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.expanduser()
    annotation_csv = args.annotation_csv.expanduser() if args.annotation_csv else package_dir / "annotation_template.csv"
    export_package = args.refresh_package or (args.annotation_csv is None and not annotation_csv.exists())
    round_report = build_annotation_round_report(
        registry_path=args.registry,
        config_path=args.config,
        round_id=args.round_id,
        package_dir=package_dir,
        match_ids=args.match_id or None,
        frames_per_match=args.frames_per_match,
        export_package=export_package,
        annotation_csv=annotation_csv,
        threshold_px=args.threshold_px,
    )

    priority_queue_path = args.priority_queue.expanduser() if args.priority_queue else package_dir / "priority_queue.csv"
    priority_queue = write_priority_queue(
        priority_queue_path,
        round_report.get("priority_tasks", []),
        limit=args.priority_limit,
    )
    annotation_ui = {}
    if not args.no_build_ui and annotation_csv.exists():
        ui_output = args.ui_output.expanduser() if args.ui_output else package_dir / "annotation_ui.html"
        annotation_ui = build_annotation_ui(
            annotation_csv,
            ui_output,
            title=f"Heatmap Labeling: {args.round_id}",
            priority_limit=args.priority_limit,
        )

    report = build_labeling_workbench_report(round_report, priority_queue=priority_queue, annotation_ui=annotation_ui)
    manifest_output = args.manifest_output.expanduser() if args.manifest_output else package_dir / "labeling_manifest.json"
    write_json(manifest_output, report)
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"heatmap labeling workbench status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"ready_to_label", "ready_for_tuning"})


if __name__ == "__main__":
    raise SystemExit(main())
