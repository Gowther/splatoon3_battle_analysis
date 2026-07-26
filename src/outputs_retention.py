from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Iterable

from src.core.paths import ROOT, project_path
from src.data_registry import display_path


DEFAULT_OUTPUTS_DIR = ROOT / "outputs"

# Regenerable intermediates: each is rebuilt by re-running the pipeline that wrote it.
REGENERABLE_DIR_NAMES = (
    "frames",
    "debug_markers",
    "cleaning_debug",
    "probes",
)

# Deliverables and provenance that must survive a sweep.
PROTECTED_DIR_NAMES = (
    "rendered",
    "player_routes",
)

PROTECTED_FILE_NAMES = (
    "run_manifest.json",
    "report.md",
)

# Annotation templates point at frame images by path. Sweeping a directory that an
# unfinished round still references would strand that round, so those are held back.
ANNOTATION_TEMPLATE_GLOB = "**/annotation_template.csv"
ANNOTATION_PATH_FIELDS = ("frame_path", "preview_path")
ANNOTATION_LABEL_FIELDS = ("x", "y")


def annotation_referenced_dirs(outputs_dir: Path) -> dict[str, str]:
    """Map each directory an unfinished annotation round depends on to the template holding it."""
    referenced: dict[str, str] = {}
    if not outputs_dir.exists():
        return referenced
    for template in sorted(outputs_dir.glob(ANNOTATION_TEMPLATE_GLOB)):
        try:
            with template.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            continue
        if not rows:
            continue
        complete = all(
            any((row.get(field) or "").strip() for field in ANNOTATION_LABEL_FIELDS) for row in rows
        )
        if complete:
            continue
        owner = display_path(template)
        for row in rows:
            for field in ANNOTATION_PATH_FIELDS:
                value = (row.get(field) or "").strip()
                if value:
                    referenced.setdefault(display_path(project_path(value).parent), owner)
    return referenced


def dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def is_protected(path: Path) -> bool:
    return path.name in PROTECTED_DIR_NAMES or path.name in PROTECTED_FILE_NAMES


def find_regenerable_dirs(
    outputs_dir: Path,
    *,
    dir_names: Iterable[str] = REGENERABLE_DIR_NAMES,
) -> list[dict[str, Any]]:
    """Find regenerable intermediate directories nested under outputs/."""
    if not outputs_dir.exists():
        return []
    wanted = set(dir_names)
    found: list[dict[str, Any]] = []
    for candidate in sorted(outputs_dir.rglob("*")):
        if not candidate.is_dir() or candidate.name not in wanted:
            continue
        if is_protected(candidate):
            continue
        found.append(
            {
                "path": display_path(candidate),
                "kind": candidate.name,
                "owner": candidate.parent.relative_to(outputs_dir).as_posix(),
                "size_bytes": dir_size(candidate),
                "file_count": sum(1 for item in candidate.rglob("*") if item.is_file()),
            }
        )
    return sorted(found, key=lambda item: item["size_bytes"], reverse=True)


def build_retention_plan(
    outputs_dir: Path | str = DEFAULT_OUTPUTS_DIR,
    *,
    dir_names: Iterable[str] = REGENERABLE_DIR_NAMES,
    min_size_bytes: int = 0,
) -> dict[str, Any]:
    resolved = project_path(outputs_dir)
    candidates = find_regenerable_dirs(resolved, dir_names=dir_names)
    referenced = annotation_referenced_dirs(resolved)

    reclaimable: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in candidates:
        owner = referenced.get(item["path"])
        if owner:
            held.append({**item, "referenced_by": owner})
        elif item["size_bytes"] < min_size_bytes:
            skipped.append(item)
        else:
            reclaimable.append(item)

    total = dir_size(resolved) if resolved.exists() else 0
    reclaimable_bytes = sum(item["size_bytes"] for item in reclaimable)
    return {
        "schema_version": 1,
        "status": "reclaimable" if reclaimable else "clean",
        "outputs_dir": display_path(resolved),
        "total_bytes": total,
        "total_display": format_size(total),
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_display": format_size(reclaimable_bytes),
        "reclaimable_percent": round(100.0 * reclaimable_bytes / total, 1) if total else 0.0,
        "candidates": reclaimable,
        "held_for_annotation": held,
        "skipped_below_threshold": skipped,
        "protected_dir_names": list(PROTECTED_DIR_NAMES),
        "protected_file_names": list(PROTECTED_FILE_NAMES),
        "regenerable_dir_names": list(dir_names),
    }


def apply_retention_plan(plan: dict[str, Any], *, dry_run: bool = True) -> dict[str, Any]:
    """Delete the planned directories. Refuses anything that is not a plan candidate."""
    removed: list[str] = []
    errors: list[str] = []
    for item in plan.get("candidates", []):
        target = project_path(item["path"])
        if not target.is_dir():
            errors.append(f"{item['path']} is not a directory")
            continue
        if is_protected(target):
            errors.append(f"{item['path']} is protected")
            continue
        if dry_run:
            removed.append(item["path"])
            continue
        try:
            shutil.rmtree(target)
            removed.append(item["path"])
        except OSError as exc:
            errors.append(f"{item['path']}: {exc}")
    return {
        "dry_run": dry_run,
        "removed": removed,
        "removed_count": len(removed),
        "freed_bytes": sum(
            item["size_bytes"] for item in plan.get("candidates", []) if item["path"] in set(removed)
        ),
        "errors": errors,
    }


def render_markdown(plan: dict[str, Any], result: dict[str, Any] | None = None) -> str:
    lines = [
        "# Outputs Retention",
        "",
        f"- status: `{plan.get('status', '')}`",
        f"- outputs_dir: `{plan.get('outputs_dir', '')}`",
        f"- total: {plan.get('total_display', '')}",
        f"- reclaimable: {plan.get('reclaimable_display', '')} ({plan.get('reclaimable_percent', 0)}%)",
        f"- candidate_dirs: {len(plan.get('candidates', []))}",
        "",
        "## Regenerable Candidates",
        "",
    ]
    candidates = plan.get("candidates", [])
    if candidates:
        lines.extend(["| path | kind | size | files |", "| --- | --- | --- | --- |"])
        lines.extend(
            f"| {item['path']} | {item['kind']} | {format_size(item['size_bytes'])} | {item['file_count']} |"
            for item in candidates
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Protected",
            "",
            f"- directories: {', '.join(plan.get('protected_dir_names', []))}",
            f"- files: {', '.join(plan.get('protected_file_names', []))}",
        ]
    )
    held = plan.get("held_for_annotation", [])
    if held:
        lines.extend(
            [
                "",
                "## Held For Annotation",
                "",
                "| path | size | referenced_by |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {item['path']} | {format_size(item['size_bytes'])} | {item['referenced_by']} |"
            for item in held
        )
    if result is not None:
        lines.extend(
            [
                "",
                "## Applied",
                "",
                f"- dry_run: {result.get('dry_run')}",
                f"- removed_count: {result.get('removed_count', 0)}",
                f"- freed: {format_size(result.get('freed_bytes', 0))}",
            ]
        )
        if result.get("errors"):
            lines.extend(["", "### Errors", ""])
            lines.extend(f"- {error}" for error in result["errors"])
    lines.append("")
    return "\n".join(lines)
