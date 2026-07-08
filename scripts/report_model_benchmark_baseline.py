from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_benchmark_baseline import build_baseline_snapshot, load_optional_json, render_markdown
from src.report_io import write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot current validation outputs as the model benchmark baseline.")
    parser.add_argument("--evaluation-results", type=Path, default=ROOT / "outputs" / "evaluation" / "evaluation_results.json")
    parser.add_argument("--model-errors", type=Path, default=ROOT / "outputs" / "validation_suite" / "model_error_report_smoothed.json")
    parser.add_argument("--heatmap-comparison", type=Path, default=ROOT / "outputs" / "validation_suite" / "heatmap_comparison.json")
    parser.add_argument("--heatmap-quality-loop", type=Path, default=ROOT / "outputs" / "heatmap_quality_loop.json")
    parser.add_argument("--benchmark-plan", type=Path, default=ROOT / "outputs" / "model_benchmark_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_benchmarks" / "baseline_snapshot.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_benchmarks" / "baseline_snapshot.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_baseline_snapshot(
        evaluation_results=load_optional_json(args.evaluation_results),
        model_errors=load_optional_json(args.model_errors),
        heatmap_comparison=load_optional_json(args.heatmap_comparison),
        heatmap_quality_loop=load_optional_json(args.heatmap_quality_loop),
        benchmark_plan=load_optional_json(args.benchmark_plan),
    )
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"model benchmark baseline status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
