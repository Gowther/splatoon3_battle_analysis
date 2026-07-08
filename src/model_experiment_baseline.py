from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BaselineStep:
    name: str
    command: Sequence[object]
    allow_failure: bool = False


def baseline_paths(output_dir: Path, *, validation_suite_ran: bool = False) -> dict[str, Path]:
    validation_root = output_dir / "validation_suite" if validation_suite_ran else Path("outputs") / "validation_suite"
    return {
        "model_registry_md": output_dir / "model_registry.md",
        "model_registry_json": output_dir / "model_registry.json",
        "model_training_plan_md": output_dir / "model_training_plan.md",
        "model_training_plan_json": output_dir / "model_training_plan.json",
        "model_training_datasets_md": output_dir / "model_training_datasets.md",
        "model_training_datasets_json": output_dir / "model_training_datasets.json",
        "validation_suite_json": output_dir / "validation_suite.json",
        "validation_suite_work_dir": output_dir / "validation_suite",
        "evaluation_results_json": validation_root / "evaluation" / "evaluation_results.json",
        "model_errors_json": validation_root / "model_error_report_smoothed.json",
        "heatmap_comparison_json": validation_root / "heatmap_comparison.json",
        "baseline_md": output_dir / "baseline_snapshot.md",
        "baseline_json": output_dir / "baseline_snapshot.json",
        "readiness_md": output_dir / "model_data_readiness.md",
        "readiness_json": output_dir / "model_data_readiness.json",
        "manifest_md": output_dir / "experiment_manifest.md",
        "manifest_json": output_dir / "experiment_manifest.json",
        "summary_json": output_dir / "run_summary.json",
    }


def build_baseline_steps(
    *,
    python: Path,
    output_dir: Path,
    run_validation_suite: bool = False,
) -> list[BaselineStep]:
    paths = baseline_paths(output_dir, validation_suite_ran=run_validation_suite)
    steps: list[BaselineStep] = [
        BaselineStep(
            "model registry",
            [
                python,
                "scripts/report_model_registry.py",
                "--output",
                paths["model_registry_md"],
                "--json-output",
                paths["model_registry_json"],
                "--strict",
            ],
        ),
        BaselineStep(
            "model training plan",
            [
                python,
                "scripts/plan_model_training.py",
                "--output",
                paths["model_training_plan_md"],
                "--json-output",
                paths["model_training_plan_json"],
            ],
        ),
        BaselineStep(
            "model training datasets",
            [
                python,
                "scripts/validate_model_training_datasets.py",
                "--output",
                paths["model_training_datasets_md"],
                "--json-output",
                paths["model_training_datasets_json"],
            ],
        ),
    ]
    if run_validation_suite:
        steps.append(
            BaselineStep(
                "validation suite",
                [
                    python,
                    "scripts/run_validation_suite.py",
                    "--output",
                    paths["validation_suite_json"],
                    "--work-dir",
                    paths["validation_suite_work_dir"],
                ],
            )
        )
    steps.extend(
        [
            BaselineStep(
                "model benchmark baseline",
                [
                    python,
                    "scripts/report_model_benchmark_baseline.py",
                    "--evaluation-results",
                    paths["evaluation_results_json"],
                    "--model-errors",
                    paths["model_errors_json"],
                    "--heatmap-comparison",
                    paths["heatmap_comparison_json"],
                    "--output",
                    paths["baseline_md"],
                    "--json-output",
                    paths["baseline_json"],
                ],
            ),
            BaselineStep(
                "model data readiness",
                [
                    python,
                    "scripts/report_model_data_readiness.py",
                    "--validation-suite",
                    paths["validation_suite_json"] if run_validation_suite else Path("outputs") / "validation_suite.json",
                    "--model-registry",
                    paths["model_registry_json"],
                    "--model-training-plan",
                    paths["model_training_plan_json"],
                    "--model-training-datasets",
                    paths["model_training_datasets_json"],
                    "--output",
                    paths["readiness_md"],
                    "--json-output",
                    paths["readiness_json"],
                ],
            ),
            BaselineStep(
                "experiment manifest",
                [
                    python,
                    "scripts/write_experiment_manifest.py",
                    "--experiment-id",
                    "model_experiment_baseline",
                    "--artifact",
                    f"model_registry={paths['model_registry_json']}",
                    "--artifact",
                    f"model_training_plan={paths['model_training_plan_json']}",
                    "--artifact",
                    f"model_training_datasets={paths['model_training_datasets_json']}",
                    "--artifact",
                    f"baseline_snapshot={paths['baseline_json']}",
                    "--artifact",
                    f"model_data_readiness={paths['readiness_json']}",
                    "--verification",
                    "scripts/run_model_experiment_baseline.py",
                    "--output",
                    paths["manifest_md"],
                    "--json-output",
                    paths["manifest_json"],
                ],
            ),
        ]
    )
    return steps


def summarize_steps(steps: list[dict[str, object]]) -> dict[str, object]:
    failed = [step for step in steps if step.get("status") == "failed"]
    needs_review = [step for step in steps if step.get("status") == "needs_review"]
    return {
        "status": "failed" if failed else "passed",
        "step_count": len(steps),
        "failed": len(failed),
        "needs_review": len(needs_review),
    }
