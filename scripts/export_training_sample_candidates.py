from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY
from src.report_io import strict_exit_code, write_json_report, write_text_report
from src.training_sample_export import build_training_sample_package, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export failed or low-quality recognition samples as labeling queues.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--evaluation-results",
        type=Path,
        default=ROOT / "outputs" / "validation_suite" / "evaluation" / "evaluation_results.json",
    )
    parser.add_argument(
        "--model-training-plan",
        type=Path,
        default=ROOT / "outputs" / "validation_suite" / "model_training_plan.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "training_sample_candidates")
    parser.add_argument("--heatmap-match-id", action="append", default=[], help="Heatmap match id to include. May repeat.")
    parser.add_argument("--max-rows-per-reason", type=int, default=24)
    parser.add_argument("--max-heatmap-items-per-match", type=int, default=24)
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap anomaly export.")
    parser.add_argument("--no-frame-export", action="store_true", help="Write queues without extracting source video frames.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "training_sample_candidates.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "training_sample_candidates.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if no candidates were exported.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_training_sample_package(
        registry_path=args.registry,
        evaluation_results_path=args.evaluation_results,
        model_training_plan_path=args.model_training_plan,
        output_dir=args.output_dir,
        include_heatmap=not args.no_heatmap,
        heatmap_match_ids=args.heatmap_match_id or None,
        max_rows_per_reason=args.max_rows_per_reason,
        max_heatmap_items_per_match=args.max_heatmap_items_per_match,
        export_frames=not args.no_frame_export,
    )
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"training sample candidates status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"ready"})


if __name__ == "__main__":
    raise SystemExit(main())
