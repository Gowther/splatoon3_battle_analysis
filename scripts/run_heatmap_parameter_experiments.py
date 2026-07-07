from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY
from src.heatmap.parameter_experiments import build_parameter_experiment_plan, render_markdown, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan heatmap parameter experiments from manual annotations.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=ROOT / "outputs" / "heatmap_annotation_round1" / "annotation_template.csv",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "heatmap_parameter_experiments")
    parser.add_argument("--write-configs", action="store_true", help="Write candidate YAML configs and candidate registries.")
    parser.add_argument("--threshold-px", type=float, default=80.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_parameter_experiments.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_parameter_experiments.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_parameter_experiment_plan(
        annotation_csv=args.annotation_csv.expanduser(),
        registry_path=args.registry,
        output_root=args.output_root.expanduser(),
        write_configs=args.write_configs,
        threshold_px=args.threshold_px,
    )
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(render_markdown(plan), encoding="utf-8")
    write_json(args.json_output.expanduser(), plan)
    print(f"heatmap parameter experiment status: {plan['status']}")
    print(f"candidates: {plan['summary']['candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
