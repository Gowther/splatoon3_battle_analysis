from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.productization import build_productization_report, render_markdown, source_report, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report heatmap productization readiness and next milestones.")
    parser.add_argument("--annotation-round", type=Path, default=ROOT / "outputs" / "heatmap_annotation_round1.json")
    parser.add_argument("--tuning-report", type=Path, default=ROOT / "outputs" / "heatmap_tuning_suggestions.json")
    parser.add_argument("--heatmap-comparison", type=Path, default=ROOT / "outputs" / "validation_suite" / "heatmap_comparison.json")
    parser.add_argument("--runtime-benchmarks", type=Path, default=ROOT / "outputs" / "runtime" / "runtime_benchmarks.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_productization.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_productization.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_productization_report(
        annotation_round=source_report(args.annotation_round),
        tuning_report=source_report(args.tuning_report),
        heatmap_comparison=source_report(args.heatmap_comparison),
        runtime_benchmarks=source_report(args.runtime_benchmarks),
    )
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(render_markdown(report), encoding="utf-8")
    write_json(args.json_output.expanduser(), report)
    print(f"heatmap productization status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
