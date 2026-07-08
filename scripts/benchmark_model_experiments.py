from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, project_path
from src.data_registry import DEFAULT_REGISTRY, load_registry
from src.model_experiments import (
    build_benchmark_plan,
    build_experiment_plan,
    load_json,
    load_optional_json,
    render_benchmark_markdown,
)
from src.report_io import write_json_report, write_text_report
from src.validation_suite import validation_ids


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a repeatable benchmark matrix for planned model experiments.")
    parser.add_argument("--experiment-plan", type=Path, help="Existing model_experiment_plan.json. Built from config when omitted.")
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "config" / "model_experiments.json")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evaluation-results", type=Path, default=ROOT / "outputs" / "evaluation" / "evaluation_results.json")
    parser.add_argument("--model-errors", type=Path, help="Optional model_errors.json used when building a plan from config.")
    parser.add_argument("--heatmap-comparison", type=Path, help="Optional heatmap_comparison.json used when building a plan from config.")
    parser.add_argument("--benchmark-root", default="outputs/model_benchmarks")
    parser.add_argument("--include-baseline-priority", action="store_true", help="Also include baseline-priority experiments.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_benchmark_plan.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_benchmark_plan.json")
    return parser.parse_args()


def read_experiment_plan(args: argparse.Namespace) -> dict:
    if args.experiment_plan:
        return load_json(args.experiment_plan)
    config = load_json(args.experiment_config)
    return build_experiment_plan(
        config,
        model_errors=load_optional_json(args.model_errors),
        heatmap_comparison=load_optional_json(args.heatmap_comparison),
    )


def main() -> int:
    args = parse_args()
    experiment_plan = read_experiment_plan(args)
    registry = load_registry(args.registry)
    evaluation_results = load_optional_json(args.evaluation_results) or []
    plan = build_benchmark_plan(
        experiment_plan,
        evaluation_results=evaluation_results,
        validation_ids=validation_ids(registry),
        benchmark_root=args.benchmark_root,
        include_baseline_priority=args.include_baseline_priority,
    )

    output = write_text_report(args.output, render_benchmark_markdown(plan))
    write_json_report(args.json_output, plan)

    print(f"wrote model benchmark plan: {output}")
    print(f"wrote model benchmark plan json: {project_path(args.json_output)}")
    print(f"model benchmark runs: {plan['summary']['run_count']}")
    for run in plan["runs"]:
        print(f"- {run['priority']}: {run['id']} ({run['candidate']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
