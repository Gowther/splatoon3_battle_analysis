from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path
from src.data_registry import display_path


DEFAULT_EXPERIMENT_CONFIG = ROOT / "config" / "model_experiments.json"


def load_json(path: Path) -> Any:
    with project_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_optional_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    resolved = project_path(path)
    if not resolved.exists():
        return None
    return load_json(resolved)


def issue_signal_counts(model_errors: dict[str, Any] | None, heatmap_comparison: dict[str, Any] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if model_errors:
        for category, count in model_errors.get("issue_counts", {}).items():
            counts[category] = counts.get(category, 0) + int(count)
    if heatmap_comparison:
        aggregate = heatmap_comparison.get("aggregate", {})
        anomaly_counts = aggregate.get("anomaly_counts", {})
        if anomaly_counts.get("track_gap"):
            counts["heatmap_gap"] = int(anomaly_counts["track_gap"])
        if anomaly_counts.get("jump_reset"):
            counts["heatmap_jump_reset"] = int(anomaly_counts["jump_reset"])
    return counts


def priority_for_experiment(experiment: dict[str, Any], signals: dict[str, int]) -> str:
    triggered = [category for category in experiment.get("trigger_categories", []) if signals.get(category, 0) > 0]
    if not triggered:
        return "baseline"
    if any(signals[category] >= 10 for category in triggered):
        return "high"
    return "medium"


def build_experiment_plan(
    config: dict[str, Any],
    model_errors: dict[str, Any] | None = None,
    heatmap_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signals = issue_signal_counts(model_errors, heatmap_comparison)
    experiments: list[dict[str, Any]] = []
    for experiment in config.get("experiments", []):
        priority = priority_for_experiment(experiment, signals)
        triggered_by = [
            {"category": category, "count": signals[category]}
            for category in experiment.get("trigger_categories", [])
            if signals.get(category, 0) > 0
        ]
        experiments.append({**experiment, "priority": priority, "triggered_by": triggered_by})

    priority_order = {"high": 0, "medium": 1, "baseline": 2}
    experiments.sort(key=lambda item: (priority_order[item["priority"]], item["area"], item["id"]))
    return {
        "status": "planned",
        "signals": signals,
        "experiments": experiments,
        "summary": {
            "experiment_count": len(experiments),
            "high_priority": sum(1 for item in experiments if item["priority"] == "high"),
            "medium_priority": sum(1 for item in experiments if item["priority"] == "medium"),
            "baseline_priority": sum(1 for item in experiments if item["priority"] == "baseline"),
        },
    }


def status_counts(results: list[dict[str, Any]] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results or []:
        status = str(item.get("status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def benchmark_commands(experiment: dict[str, Any], benchmark_root: str) -> list[str]:
    experiment_dir = f"{benchmark_root}/{experiment['id']}"
    commands = [
        f"python scripts/run_validation_suite.py --output {experiment_dir}/baseline_validation_suite.json",
        f"python scripts/report_model_errors.py --evaluation-results {experiment_dir}/baseline_validation_suite/evaluation/evaluation_results.json --smoothed --output {experiment_dir}/baseline_model_errors.md --json-output {experiment_dir}/baseline_model_errors.json",
    ]
    if experiment.get("area") == "heatmap":
        commands.append(
            f"python scripts/report_heatmap_quality_loop.py --export-package --package-dir {experiment_dir}/annotation_package --output {experiment_dir}/heatmap_quality_loop.md --json-output {experiment_dir}/heatmap_quality_loop.json"
        )
    commands.extend(experiment.get("baseline_commands", []))
    return commands


def result_template(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment["id"],
        "candidate": experiment.get("candidate", ""),
        "status": "not_run",
        "baseline_metrics": {},
        "candidate_metrics": {},
        "decision": "pending",
        "notes": "",
    }


def build_benchmark_plan(
    experiment_plan: dict[str, Any],
    *,
    evaluation_results: list[dict[str, Any]] | None = None,
    validation_ids: list[str] | None = None,
    benchmark_root: str = "outputs/model_benchmarks",
    include_baseline_priority: bool = False,
) -> dict[str, Any]:
    selected_experiments = [
        experiment
        for experiment in experiment_plan.get("experiments", [])
        if include_baseline_priority or experiment.get("priority") != "baseline"
    ]
    runs = [
        {
            "id": experiment["id"],
            "priority": experiment.get("priority", ""),
            "area": experiment.get("area", ""),
            "candidate": experiment.get("candidate", ""),
            "validation_ids": validation_ids or [],
            "commands": benchmark_commands(experiment, benchmark_root),
            "metrics": experiment.get("metrics", []),
            "pass_criteria": experiment.get("pass_criteria", []),
            "result_template": result_template(experiment),
        }
        for experiment in selected_experiments
    ]
    return {
        "status": "ready" if runs else "empty",
        "benchmark_root": benchmark_root,
        "baseline_result_status_counts": status_counts(evaluation_results),
        "validation_ids": validation_ids or [],
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "high_priority": sum(1 for item in runs if item["priority"] == "high"),
            "medium_priority": sum(1 for item in runs if item["priority"] == "medium"),
            "baseline_priority": sum(1 for item in runs if item["priority"] == "baseline"),
        },
    }


def render_benchmark_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Model Benchmark Plan",
        "",
        f"- status: `{plan['status']}`",
        f"- benchmark_root: `{plan['benchmark_root']}`",
        f"- runs: {plan['summary']['run_count']}",
        f"- validation_ids: {', '.join(plan.get('validation_ids', []))}",
        f"- baseline_result_status_counts: {json.dumps(plan.get('baseline_result_status_counts', {}), ensure_ascii=False)}",
        "",
        "| priority | area | id | candidate | metrics |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for run in plan.get("runs", []):
        lines.append(
            "| {priority} | {area} | {id} | {candidate} | {metrics} |".format(
                priority=run.get("priority", ""),
                area=run.get("area", ""),
                id=run.get("id", ""),
                candidate=run.get("candidate", ""),
                metrics=len(run.get("metrics", [])),
            )
        )

    for run in plan.get("runs", []):
        lines.extend(["", f"## {run['id']}", "", "Commands:"])
        lines.extend(f"- `{command}`" for command in run.get("commands", []))
        lines.extend(["", "Pass criteria:"])
        lines.extend(f"- {criterion}" for criterion in run.get("pass_criteria", []))
        lines.extend(["", "Result template:", "", "```json", json.dumps(run["result_template"], indent=2, ensure_ascii=False), "```"])
    return "\n".join(lines) + "\n"


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Model Experiment Plan",
        "",
        f"- status: `{plan['status']}`",
        f"- experiments: {plan['summary']['experiment_count']}",
        f"- high_priority: {plan['summary']['high_priority']}",
        f"- medium_priority: {plan['summary']['medium_priority']}",
        f"- baseline_priority: {plan['summary']['baseline_priority']}",
        f"- signals: {json.dumps(plan['signals'], ensure_ascii=False)}",
        "",
        "## Experiments",
        "",
        "| priority | area | id | candidate | triggers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for experiment in plan["experiments"]:
        triggers = ", ".join(f"{item['category']}={item['count']}" for item in experiment["triggered_by"]) or "-"
        lines.append(
            "| {priority} | {area} | {id} | {candidate} | {triggers} |".format(
                priority=experiment["priority"],
                area=experiment["area"],
                id=experiment["id"],
                candidate=experiment["candidate"],
                triggers=triggers,
            )
        )

    for experiment in plan["experiments"]:
        lines.extend(
            [
                "",
                f"## {experiment['id']}",
                "",
                f"- priority: `{experiment['priority']}`",
                f"- area: `{experiment['area']}`",
                f"- candidate: {experiment['candidate']}",
                f"- status: {experiment['status']}",
                "",
                "Baseline commands:",
            ]
        )
        lines.extend(f"- `{command}`" for command in experiment.get("baseline_commands", []))
        lines.extend(["", "Metrics:"])
        lines.extend(f"- {metric}" for metric in experiment.get("metrics", []))
        lines.extend(["", "Pass criteria:"])
        lines.extend(f"- {criterion}" for criterion in experiment.get("pass_criteria", []))
        if experiment.get("notes"):
            lines.extend(["", f"Notes: {experiment['notes']}"])

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not replace a runtime model until the current baseline report and error report are saved.",
            "- Compare candidates on the same registered match windows before changing production defaults.",
            "- Keep existing YOLOv5/OCR/ResNet weights as controls until a candidate beats them on documented metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_input(path: Path | None) -> str:
    return display_path(project_path(path)) if path else ""
