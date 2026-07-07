from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_registry import resolve_project_path


def load_optional_json(path: Path | str | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    target = resolve_project_path(path) or Path(path).expanduser()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def report_status(payload: dict[str, Any] | None) -> str:
    return str(payload.get("status", "missing")) if payload else "missing"


def build_model_data_readiness_report(
    *,
    annotation_round: dict[str, Any] | None = None,
    parameter_experiments: dict[str, Any] | None = None,
    runtime_benchmarks: dict[str, Any] | None = None,
    validation_suite: dict[str, Any] | None = None,
    dataset_governance: dict[str, Any] | None = None,
    model_registry: dict[str, Any] | None = None,
    model_training_plan: dict[str, Any] | None = None,
    model_experiment_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    annotation_progress = (annotation_round or {}).get("progress", {})
    labeled_rows = int(annotation_progress.get("labeled_rows") or 0)
    if labeled_rows < 30:
        blockers.append(
            {
                "area": "heatmap_labels",
                "detail": f"Need at least 30 labeled heatmap rows before parameter comparisons; current={labeled_rows}.",
            }
        )
    if report_status(parameter_experiments) == "needs_labels":
        blockers.append({"area": "heatmap_parameter_experiments", "detail": "Parameter experiments are waiting for manual labels."})
    if report_status(validation_suite) not in {"passed", "ready"}:
        blockers.append({"area": "validation", "detail": f"Validation suite status is {report_status(validation_suite)}."})
    if report_status(runtime_benchmarks) not in {"ready"}:
        warnings.append({"area": "runtime", "detail": f"Runtime benchmark status is {report_status(runtime_benchmarks)}."})
    if report_status(dataset_governance) not in {"passed"}:
        warnings.append({"area": "dataset_governance", "detail": f"Dataset governance status is {report_status(dataset_governance)}."})
    if report_status(model_registry) not in {"passed"}:
        blockers.append({"area": "model_registry", "detail": f"Model registry status is {report_status(model_registry)}."})
    if report_status(model_training_plan) not in {"ready"}:
        warnings.append({"area": "model_training_plan", "detail": f"Model training plan status is {report_status(model_training_plan)}."})

    experiment_summary = (model_experiment_plan or {}).get("summary", {})
    actions = [
        "Fill the prioritized heatmap annotation rows and rerun parameter experiments.",
        "Keep runtime and validation reports attached to every model/data experiment.",
    ]
    if int(experiment_summary.get("high_priority") or 0) or int(experiment_summary.get("medium_priority") or 0):
        actions.append("Run benchmark_model_experiments.py for triggered model candidates before changing runtime defaults.")

    return {
        "schema_version": 1,
        "status": "needs_data" if blockers else "ready_for_model_data_experiments",
        "inputs": {
            "annotation_round": report_status(annotation_round),
            "parameter_experiments": report_status(parameter_experiments),
            "runtime_benchmarks": report_status(runtime_benchmarks),
            "validation_suite": report_status(validation_suite),
            "dataset_governance": report_status(dataset_governance),
            "model_registry": report_status(model_registry),
            "model_training_plan": report_status(model_training_plan),
            "model_experiment_plan": report_status(model_experiment_plan),
        },
        "heatmap_labels": {
            "labeled_rows": labeled_rows,
            "recommended_min_labeled_rows": 30,
            "complete_frame_team_groups": int(annotation_progress.get("complete_frame_team_groups") or 0),
        },
        "experiment_summary": experiment_summary,
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": actions,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model And Data Readiness",
        "",
        f"- status: `{report['status']}`",
        f"- labeled_rows: {report['heatmap_labels']['labeled_rows']} / {report['heatmap_labels']['recommended_min_labeled_rows']}",
        "",
        "## Inputs",
        "",
    ]
    for name, status in report["inputs"].items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item['area']}`: {item['detail']}" for item in report.get("blockers", []))
    if not report.get("blockers"):
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- `{item['area']}`: {item['detail']}" for item in report.get("warnings", []))
    if not report.get("warnings"):
        lines.append("- none")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report.get("next_actions", []))
    lines.append("")
    return "\n".join(lines)
