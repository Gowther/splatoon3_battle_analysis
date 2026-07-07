from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_registry import display_path, resolve_project_path
from src.runtime_report import format_seconds


DEFAULT_BENCHMARKS = [
    {
        "id": "unit_tests",
        "command": "python scripts/run_with_runtime_report.py --name unit_tests --output outputs/runtime/unit_tests.json -- python -m unittest discover -s tests -q",
        "expected_report": "outputs/runtime/unit_tests.json",
    },
    {
        "id": "validation_suite",
        "command": "python scripts/run_with_runtime_report.py --name validation_suite --output outputs/runtime/validation_suite.json -- python scripts/run_validation_suite.py",
        "expected_report": "outputs/runtime/validation_suite.json",
    },
    {
        "id": "run_analysis_sample_image",
        "command": "python scripts/run_with_runtime_report.py --name run_analysis_sample_image --output outputs/runtime/run_analysis_sample_image.json -- python -m src.run_analysis --input sample/battle.png --output outputs/runtime/run_analysis_sample_image.csv --device cpu --max-frames 1",
        "expected_report": "outputs/runtime/run_analysis_sample_image.json",
    },
    {
        "id": "heatmap_pipeline_match9_report_only",
        "command": "python scripts/run_with_runtime_report.py --name heatmap_pipeline_match9_report_only --output outputs/runtime/heatmap_pipeline_match9_report_only.json -- python -m src.heatmap.run_pipeline --config src/heatmap/config_match9.yaml --only-report --disable-auto-colors",
        "expected_report": "outputs/runtime/heatmap_pipeline_match9_report_only.json",
    },
]


def load_json(path: Path) -> Any | None:
    target = resolve_project_path(path) or path.expanduser()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def report_item(label: str, path: Path) -> dict[str, Any]:
    payload = load_json(path)
    target = resolve_project_path(path) or path.expanduser()
    if not payload:
        return {"label": label, "path": display_path(target), "status": "missing"}
    return {
        "label": label,
        "path": display_path(target),
        "status": "ready",
        "name": payload.get("name", label),
        "total_seconds": payload.get("total_seconds", 0),
        "total_display": payload.get("total_display", format_seconds(float(payload.get("total_seconds", 0) or 0))),
        "step_count": payload.get("step_count", 0),
        "generated_at": payload.get("generated_at", ""),
    }


def build_runtime_benchmark_report(runtime_reports: list[tuple[str, Path]]) -> dict[str, Any]:
    items = [report_item(label, path) for label, path in runtime_reports]
    ready = [item for item in items if item["status"] == "ready"]
    missing = [item for item in items if item["status"] != "ready"]
    return {
        "schema_version": 1,
        "status": "ready" if ready and not missing else ("partial" if ready else "needs_reports"),
        "reports": items,
        "summary": {
            "ready": len(ready),
            "missing": len(missing),
            "slowest": max(ready, key=lambda item: float(item.get("total_seconds", 0)), default={}).get("label", ""),
        },
        "planned_benchmarks": DEFAULT_BENCHMARKS,
    }


def parse_runtime_report_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label, Path(path)
    path = Path(value)
    return path.stem, path


def default_runtime_reports() -> list[tuple[str, Path]]:
    return [(item["id"], Path(item["expected_report"])) for item in DEFAULT_BENCHMARKS]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Runtime Benchmarks",
        "",
        f"- status: `{report.get('status')}`",
        f"- ready: {report.get('summary', {}).get('ready', 0)}",
        f"- missing: {report.get('summary', {}).get('missing', 0)}",
        f"- slowest: `{report.get('summary', {}).get('slowest', '')}`",
        "",
        "| status | label | total | steps | path |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in report.get("reports", []):
        lines.append(
            f"| {item.get('status')} | {item.get('label')} | {item.get('total_display', '')} | "
            f"{item.get('step_count', '')} | `{item.get('path', '')}` |"
        )
    lines.extend(["", "## Planned Commands", ""])
    for item in report.get("planned_benchmarks", []):
        lines.append(f"- `{item['id']}`: `{item['command']}`")
    lines.append("")
    return "\n".join(lines)
