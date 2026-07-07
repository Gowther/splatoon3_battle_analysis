from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CheckStep:
    name: str
    command: Sequence[object]


def heatmap_annotation_steps(python: Path, work_dir: Path) -> list[CheckStep]:
    annotation_csv = work_dir / "heatmap_annotation_round1" / "annotation_template.csv"
    return [
        CheckStep(
            "heatmap annotation round helper",
            [
                python,
                "scripts/prepare_heatmap_annotation_round.py",
                "--package-dir",
                work_dir / "heatmap_annotation_round1",
                "--output",
                work_dir / "heatmap_annotation_round1.md",
                "--json-output",
                work_dir / "heatmap_annotation_round1.json",
            ],
        ),
        CheckStep(
            "heatmap annotation UI helper",
            [
                python,
                "scripts/build_heatmap_annotation_ui.py",
                "--annotation-csv",
                annotation_csv,
                "--output",
                work_dir / "heatmap_annotation_round1" / "annotation_ui.html",
                "--json-output",
                work_dir / "heatmap_annotation_ui.json",
            ],
        ),
        CheckStep(
            "heatmap tuning suggestion helper",
            [
                python,
                "scripts/suggest_heatmap_tuning.py",
                "--annotation-csv",
                annotation_csv,
                "--heatmap-comparison",
                work_dir / "heatmap_comparison.json",
                "--output",
                work_dir / "heatmap_tuning_suggestions.md",
                "--json-output",
                work_dir / "heatmap_tuning_suggestions.json",
            ],
        ),
        CheckStep(
            "heatmap parameter experiment helper",
            [
                python,
                "scripts/run_heatmap_parameter_experiments.py",
                "--annotation-csv",
                annotation_csv,
                "--output-root",
                work_dir / "heatmap_parameter_experiments",
                "--write-configs",
                "--output",
                work_dir / "heatmap_parameter_experiments.md",
                "--json-output",
                work_dir / "heatmap_parameter_experiments.json",
            ],
        ),
    ]


def experiment_delivery_steps(python: Path, root: Path, work_dir: Path) -> list[CheckStep]:
    return [
        CheckStep(
            "runtime wrapper helper",
            [
                python,
                "scripts/run_with_runtime_report.py",
                "--name",
                "summarize_sample_csv",
                "--output",
                work_dir / "runtime_summary.json",
                "--markdown-output",
                work_dir / "runtime_summary.md",
                "--",
                str(python),
                "scripts/summarize_csv.py",
                str(work_dir / "sample.csv"),
            ],
        ),
        CheckStep(
            "runtime benchmark report helper",
            [
                python,
                "scripts/report_runtime_benchmarks.py",
                "--runtime-report",
                f"summarize_sample_csv={work_dir / 'runtime_summary.json'}",
                "--output",
                work_dir / "runtime_benchmarks.md",
                "--json-output",
                work_dir / "runtime_benchmarks.json",
            ],
        ),
        CheckStep(
            "model experiment plan helper",
            [
                python,
                "scripts/plan_model_experiments.py",
                "--model-errors",
                work_dir / "model_errors.json",
                "--heatmap-comparison",
                work_dir / "heatmap_comparison.json",
                "--output",
                work_dir / "model_experiment_plan.md",
                "--json-output",
                work_dir / "model_experiment_plan.json",
            ],
        ),
        CheckStep(
            "model benchmark plan helper",
            [
                python,
                "scripts/benchmark_model_experiments.py",
                "--experiment-plan",
                work_dir / "model_experiment_plan.json",
                "--heatmap-comparison",
                work_dir / "heatmap_comparison.json",
                "--output",
                work_dir / "model_benchmark_plan.md",
                "--json-output",
                work_dir / "model_benchmark_plan.json",
            ],
        ),
        CheckStep(
            "model benchmark baseline helper",
            [
                python,
                "scripts/report_model_benchmark_baseline.py",
                "--evaluation-results",
                root / "outputs" / "evaluation" / "evaluation_results.json",
                "--model-errors",
                work_dir / "model_errors.json",
                "--heatmap-comparison",
                work_dir / "heatmap_comparison.json",
                "--heatmap-quality-loop",
                work_dir / "heatmap_quality_loop.json",
                "--benchmark-plan",
                work_dir / "model_benchmark_plan.json",
                "--output",
                work_dir / "model_benchmark_baseline.md",
                "--json-output",
                work_dir / "model_benchmark_baseline.json",
            ],
        ),
        CheckStep(
            "experiment manifest helper",
            [
                python,
                "scripts/write_experiment_manifest.py",
                "--experiment-id",
                "check_project_tooling",
                "--artifact",
                f"model_benchmark_baseline={work_dir / 'model_benchmark_baseline.json'}",
                "--artifact",
                f"heatmap_parameter_experiments={work_dir / 'heatmap_parameter_experiments.json'}",
                "--verification",
                "check_project --tooling helper smoke",
                "--output",
                work_dir / "experiment_manifest.md",
                "--json-output",
                work_dir / "experiment_manifest.json",
            ],
        ),
        CheckStep(
            "heatmap productization helper",
            [
                python,
                "scripts/report_heatmap_productization.py",
                "--annotation-round",
                work_dir / "heatmap_annotation_round1.json",
                "--tuning-report",
                work_dir / "heatmap_tuning_suggestions.json",
                "--heatmap-comparison",
                work_dir / "heatmap_comparison.json",
                "--runtime-benchmarks",
                work_dir / "runtime_benchmarks.json",
                "--output",
                work_dir / "heatmap_productization.md",
                "--json-output",
                work_dir / "heatmap_productization.json",
            ],
        ),
        CheckStep(
            "stage coordinate normalization helper",
            [
                python,
                "scripts/report_stage_coordinates.py",
                "--config",
                "src/heatmap/config_match9.yaml",
                "--output",
                work_dir / "stage_coordinates.md",
                "--json-output",
                work_dir / "stage_coordinates.json",
            ],
        ),
        CheckStep(
            "change package helper",
            [
                python,
                "scripts/report_change_package.py",
                "--verification",
                "check_project --tooling helper smoke",
                "--output",
                work_dir / "change_package.md",
                "--json-output",
                work_dir / "change_package.json",
            ],
        ),
    ]
