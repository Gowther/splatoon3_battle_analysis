from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY
from src.heatmap.tuning import build_tuning_report, render_markdown
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest heatmap tracker tuning actions from manual annotations.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=ROOT / "outputs" / "heatmap_annotation_round1" / "annotation_template.csv",
    )
    parser.add_argument("--heatmap-comparison", type=Path, help="Optional heatmap comparison JSON.")
    parser.add_argument("--threshold-px", type=float, default=80.0)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--max-mean-error-px", type=float)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_tuning_suggestions.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_tuning_suggestions.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when labels are missing or checks fail.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_tuning_report(
        registry_path=args.registry,
        annotation_csv=args.annotation_csv,
        threshold_px=args.threshold_px,
        min_recall=args.min_recall,
        max_mean_error_px=args.max_mean_error_px,
        heatmap_comparison_json=args.heatmap_comparison,
    )
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"heatmap tuning status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"ready"})


if __name__ == "__main__":
    raise SystemExit(main())
