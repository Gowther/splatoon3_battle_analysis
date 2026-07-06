from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, load_registry, resolve_project_path
from src.heatmap.annotation_eval import evaluate_annotations, evaluate_gates, write_json, write_markdown


DEFAULT_CONFIG = ROOT / "config" / "annotation_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heatmap player positions against manual annotations.")
    parser.add_argument("annotation_csv", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="JSON output path. Defaults next to annotation CSV.")
    parser.add_argument("--report", type=Path, help="Markdown output path. Defaults next to annotation CSV.")
    parser.add_argument("--threshold-px", type=float, help="Override matching distance threshold.")
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--max-mean-error-px", type=float)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when labels are missing or checks fail.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = resolve_project_path(path) or path.expanduser()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    threshold_px = args.threshold_px or float(config.get("defaults", {}).get("annotation_distance_threshold_px", 80.0))
    annotation_csv = args.annotation_csv.expanduser()
    output = args.output.expanduser() if args.output else annotation_csv.with_suffix(".evaluation.json")
    report = args.report.expanduser() if args.report else annotation_csv.with_suffix(".evaluation.md")
    registry = load_registry(args.registry)

    metrics = evaluate_annotations(annotation_csv, registry, threshold_px=threshold_px)
    checks = evaluate_gates(metrics, min_recall=args.min_recall, max_mean_error_px=args.max_mean_error_px)
    payload = {"metrics": metrics, "checks": checks}
    write_json(output, payload)
    write_markdown(report, metrics, checks)
    print(f"wrote annotation evaluation: {output}")
    print(f"wrote annotation report: {report}")
    print(f"labels: {metrics['labeled_rows']} matched: {metrics['matched_labels']} recall: {metrics['recall']}")

    failed_checks = any(not check["ok"] for check in checks.values())
    if args.strict and (metrics["status"] == "no_labels" or failed_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
