from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data_readiness import build_model_data_readiness_report, load_optional_json, render_markdown
from src.report_io import write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize readiness for the next model/data experiment phase.")
    parser.add_argument("--annotation-round", type=Path, default=ROOT / "outputs" / "heatmap_annotation_round1.json")
    parser.add_argument("--parameter-experiments", type=Path, default=ROOT / "outputs" / "heatmap_parameter_experiments.json")
    parser.add_argument("--runtime-benchmarks", type=Path, default=ROOT / "outputs" / "runtime" / "runtime_benchmarks.json")
    parser.add_argument("--validation-suite", type=Path, default=ROOT / "outputs" / "validation_suite.json")
    parser.add_argument("--dataset-governance", type=Path, default=ROOT / "outputs" / "dataset_governance.json")
    parser.add_argument("--model-registry", type=Path, default=ROOT / "outputs" / "model_registry.json")
    parser.add_argument("--model-training-plan", type=Path, default=ROOT / "outputs" / "model_training_plan.json")
    parser.add_argument("--model-training-datasets", type=Path, default=ROOT / "outputs" / "model_training_datasets.json")
    parser.add_argument("--model-experiment-plan", type=Path, default=ROOT / "outputs" / "model_experiment_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_data_readiness.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_data_readiness.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_model_data_readiness_report(
        annotation_round=load_optional_json(args.annotation_round),
        parameter_experiments=load_optional_json(args.parameter_experiments),
        runtime_benchmarks=load_optional_json(args.runtime_benchmarks),
        validation_suite=load_optional_json(args.validation_suite),
        dataset_governance=load_optional_json(args.dataset_governance),
        model_registry=load_optional_json(args.model_registry),
        model_training_plan=load_optional_json(args.model_training_plan),
        model_training_datasets=load_optional_json(args.model_training_datasets),
        model_experiment_plan=load_optional_json(args.model_experiment_plan),
    )
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"model/data readiness status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
