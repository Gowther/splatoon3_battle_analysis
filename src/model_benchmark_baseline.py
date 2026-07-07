from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_registry import resolve_project_path
from src.model_experiments import status_counts


def load_optional_json(path: Path | None) -> Any | None:
    if path is None:
        return None
    target = resolve_project_path(path) or path.expanduser()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def issue_counts(model_errors: dict[str, Any] | None) -> dict[str, int]:
    return {str(key): int(value) for key, value in (model_errors or {}).get("issue_counts", {}).items()}


def heatmap_counts(heatmap_comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not heatmap_comparison:
        return {}
    aggregate = heatmap_comparison.get("aggregate", {})
    return {
        "status": heatmap_comparison.get("status", ""),
        "matches": len(heatmap_comparison.get("matches", [])),
        "anomaly_counts": aggregate.get("anomaly_counts", {}),
    }


def quality_loop_counts(quality_loop: dict[str, Any] | None) -> dict[str, Any]:
    if not quality_loop:
        return {}
    metrics = quality_loop.get("metrics", {})
    return {
        "status": quality_loop.get("status", ""),
        "labeled_rows": metrics.get("labeled_rows"),
        "matched_labels": metrics.get("matched_labels"),
        "recall": metrics.get("recall"),
        "mean_error_px": metrics.get("mean_error_px"),
    }


def build_baseline_snapshot(
    *,
    evaluation_results: list[dict[str, Any]] | None,
    model_errors: dict[str, Any] | None,
    heatmap_comparison: dict[str, Any] | None,
    heatmap_quality_loop: dict[str, Any] | None,
    benchmark_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    missing_inputs = []
    if evaluation_results is None:
        missing_inputs.append("evaluation_results")
    if model_errors is None:
        missing_inputs.append("model_errors")
    if heatmap_comparison is None:
        missing_inputs.append("heatmap_comparison")
    return {
        "schema_version": 1,
        "status": "ready" if not missing_inputs else "needs_inputs",
        "missing_inputs": missing_inputs,
        "evaluation": {
            "result_count": len(evaluation_results or []),
            "status_counts": status_counts(evaluation_results),
        },
        "model_errors": {
            "status": (model_errors or {}).get("status", ""),
            "issue_counts": issue_counts(model_errors),
        },
        "heatmap_comparison": heatmap_counts(heatmap_comparison),
        "heatmap_quality_loop": quality_loop_counts(heatmap_quality_loop),
        "benchmark_plan": {
            "status": (benchmark_plan or {}).get("status", ""),
            "run_count": (benchmark_plan or {}).get("summary", {}).get("run_count", 0),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Model Benchmark Baseline",
            "",
            f"- status: `{report.get('status')}`",
            f"- missing_inputs: {', '.join(report.get('missing_inputs', [])) or '-'}",
            f"- evaluation_results: {report.get('evaluation', {}).get('result_count', 0)}",
            f"- evaluation_status_counts: {json.dumps(report.get('evaluation', {}).get('status_counts', {}), ensure_ascii=False)}",
            f"- model_issue_counts: {json.dumps(report.get('model_errors', {}).get('issue_counts', {}), ensure_ascii=False)}",
            f"- heatmap_anomaly_counts: {json.dumps(report.get('heatmap_comparison', {}).get('anomaly_counts', {}), ensure_ascii=False)}",
            f"- heatmap_quality_loop_status: `{report.get('heatmap_quality_loop', {}).get('status', '')}`",
            f"- benchmark_runs: {report.get('benchmark_plan', {}).get('run_count', 0)}",
            "",
        ]
    )
