from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.paths import project_path


PASSING_STATUSES = {"passed", "ready"}


def resolve_report_path(path: Path) -> Path:
    return project_path(path)


def write_text_report(path: Path, content: str) -> Path:
    target = resolve_report_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote: {target}")
    return target


def write_json_report(path: Path, payload: Any) -> Path:
    return write_text_report(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def emit_markdown_or_stdout(output: Path | None, markdown: str) -> Path | None:
    if output:
        return write_text_report(output, markdown)
    print(markdown, end="")
    return None


def strict_exit_code(status: str, strict: bool, passing_statuses: set[str] | None = None) -> int:
    if not strict:
        return 0
    allowed = passing_statuses or PASSING_STATUSES
    return 0 if status in allowed else 1
