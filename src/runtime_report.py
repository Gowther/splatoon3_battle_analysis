from __future__ import annotations

import datetime as dt
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


class RuntimeRecorder:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    @contextmanager
    def step(self, label: str, **metadata: Any) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.steps.append(
                {
                    "label": label,
                    "duration_seconds": round(time.perf_counter() - started, 4),
                    **metadata,
                }
            )


def build_runtime_report(name: str, steps: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    total = round(sum(float(step.get("duration_seconds", 0.0)) for step in steps), 4)
    return {
        "schema_version": 1,
        "name": name,
        "generated_at": now_iso(),
        "total_seconds": total,
        "total_display": format_seconds(total),
        "step_count": len(steps),
        "metadata": metadata or {},
        "steps": steps,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('name', 'Runtime')} Runtime Report",
        "",
        f"- generated_at: `{report.get('generated_at', '')}`",
        f"- total: {report.get('total_display', '')}",
        f"- steps: {report.get('step_count', 0)}",
        "",
        "| step | duration | detail |",
        "| --- | ---: | --- |",
    ]
    for step in report.get("steps", []):
        detail = step.get("command") or step.get("detail") or ""
        duration = format_seconds(float(step.get("duration_seconds", 0.0)))
        lines.append(f"| {step.get('label', '')} | {duration} | `{detail}` |")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
