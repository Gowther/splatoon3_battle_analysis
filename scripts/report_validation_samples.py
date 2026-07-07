from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, resolve_project_path
from src.validation_sample_report import build_report, render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a summary report for registered validation samples.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evaluation-results", type=Path, default=ROOT / "outputs" / "evaluation" / "evaluation_results.json")
    parser.add_argument("--heatmap-comparison", type=Path, default=ROOT / "outputs" / "heatmap_comparison.json")
    parser.add_argument("--analysis-window-scan", type=Path, default=ROOT / "outputs" / "analysis_window_scan" / "analysis_window_scan.json")
    parser.add_argument("--model-error-report", type=Path, default=ROOT / "outputs" / "model_error_report_smoothed.json")
    parser.add_argument("--heatmap-quality-loop", type=Path, help="Optional heatmap annotation/evaluation quality loop JSON.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "validation_samples.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "validation_samples.json")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_json(path: Path, fallback: Any) -> Any:
    target = resolve_project_path(path) or path.expanduser()
    if not target.exists():
        return fallback
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, content: str) -> None:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote: {target}")


def main() -> int:
    args = parse_args()
    registry = read_json(args.registry, {"matches": []})
    evaluation_results = read_json(args.evaluation_results, [])
    heatmap_comparison = read_json(args.heatmap_comparison, {})
    analysis_scan = read_json(args.analysis_window_scan, {})
    model_error_report = read_json(args.model_error_report, {})
    heatmap_quality_loop = read_json(args.heatmap_quality_loop, {}) if args.heatmap_quality_loop else None

    report = build_report(
        registry,
        evaluation_results,
        heatmap_comparison,
        analysis_scan,
        model_error_report,
        heatmap_quality_loop,
    )
    write_text(args.output, render_markdown(report))
    write_text(args.json_output, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"validation sample report status: {report['status']}")
    return 1 if args.strict and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
