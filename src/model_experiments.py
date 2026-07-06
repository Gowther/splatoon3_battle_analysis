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
