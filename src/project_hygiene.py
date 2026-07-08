from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Any


ACTIVE_ROOT_DIRS = {
    "config",
    "main_icons",
    "models",
    "sample",
    "scripts",
    "src",
    "tests",
    "yolov5",
}

LOCAL_DATA_ROOTS = {
    ".cache",
    ".idea",
    ".venv",
    ".vscode",
    "data",
    "footages",
    "main_training_dataset",
    "outputs",
}

LEGACY_REFERENCE_ROOTS = {
    ".models",
    "legacy",
    "notebooks",
}

ACTIVE_ROOT_FILES = {
    ".gitignore",
    "main_weapon_list.txt",
    "requirements-mac-m4.txt",
}

GENERATED_ROOT_FILE_PATTERNS = (
    "20*.csv",
    "*.log",
)

PYTHON_CACHE_EXCLUDES = {
    ".cache",
    ".git",
    ".venv",
}

ACTIVE_IMPORT_SCAN_ROOTS = ("src", "scripts")
BOUNDARY_IMPORT_PREFIXES = ("legacy", "yolov5")


def display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def root_entries(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((item for item in root.iterdir()), key=lambda path: path.name)


def is_allowed_root_entry(path: Path) -> bool:
    name = path.name
    if name == ".git" or name == ".DS_Store":
        return True
    if path.is_dir():
        return name in ACTIVE_ROOT_DIRS or name in LOCAL_DATA_ROOTS or name in LEGACY_REFERENCE_ROOTS
    if path.suffix == ".md":
        return True
    return name in ACTIVE_ROOT_FILES


def generated_root_files(root: Path) -> list[str]:
    generated: list[str] = []
    for path in root_entries(root):
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in GENERATED_ROOT_FILE_PATTERNS):
            generated.append(display_path(root, path))
    return generated


def unexpected_root_entries(root: Path) -> list[str]:
    return [display_path(root, path) for path in root_entries(root) if not is_allowed_root_entry(path)]


def directory_file_count(path: Path, limit: int = 10000) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            if count >= limit:
                return count
    return count


def find_stray_pycache_dirs(root: Path, limit: int = 50) -> list[str]:
    found: list[str] = []
    for current, dirs, _ in os.walk(root):
        current_path = Path(current)
        try:
            parts = set(current_path.relative_to(root).parts)
        except ValueError:
            continue
        if parts & PYTHON_CACHE_EXCLUDES:
            dirs[:] = []
            continue
        if "__pycache__" in dirs:
            path = current_path / "__pycache__"
            found.append(display_path(root, path))
            dirs.remove("__pycache__")
        if len(found) >= limit:
            break
    return sorted(found)


def active_boundary_imports(root: Path, limit: int = 50) -> list[str]:
    matches: list[str] = []
    for dirname in ACTIVE_IMPORT_SCAN_ROOTS:
        scan_root = root / dirname
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if any(
                    stripped == f"import {name}"
                    or stripped.startswith(f"import {name}.")
                    or stripped.startswith(f"from {name} ")
                    or stripped.startswith(f"from {name}.")
                    for name in BOUNDARY_IMPORT_PREFIXES
                ):
                    matches.append(f"{display_path(root, path)}:{line_number}: {stripped}")
                    break
            if len(matches) >= limit:
                return matches
    return matches


def issue(severity: str, category: str, detail: str, items: list[str]) -> dict[str, Any]:
    return {"severity": severity, "category": category, "detail": detail, "items": items}


def build_hygiene_report(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    unexpected = unexpected_root_entries(root)
    generated = generated_root_files(root)
    stray_pycache = find_stray_pycache_dirs(root)
    boundary_imports = active_boundary_imports(root)

    local_data = {
        name: {"exists": (root / name).exists(), "file_count": directory_file_count(root / name)}
        for name in sorted(LOCAL_DATA_ROOTS)
    }
    legacy_references = {
        name: {"exists": (root / name).exists(), "file_count": directory_file_count(root / name)}
        for name in sorted(LEGACY_REFERENCE_ROOTS)
    }

    issues: list[dict[str, Any]] = []
    if unexpected:
        issues.append(issue("warning", "root_layout", "unexpected root-level entries", unexpected))
    if generated:
        issues.append(issue("warning", "generated_outputs", "generated root-level files should live under outputs/", generated))
    if stray_pycache:
        issues.append(issue("info", "python_cache", "stray __pycache__ directories outside .cache/.venv", stray_pycache))
    if boundary_imports:
        issues.append(
            issue(
                "warning",
                "boundary_imports",
                "active code should not import legacy or vendored yolov5 packages directly",
                boundary_imports,
            )
        )
    if legacy_references.get(".models", {}).get("exists"):
        issues.append(
            issue(
                "info",
                "legacy_reference",
                ".models is a tracked historical export archive; supported runtime weights live under models/",
                [".models"],
            )
        )

    has_blocking_issue = any(item["severity"] in {"warning", "high"} for item in issues)
    return {
        "status": "needs_review" if has_blocking_issue else "passed",
        "root": root.as_posix(),
        "unexpected_root_entries": unexpected,
        "generated_root_files": generated,
        "stray_pycache_dirs": stray_pycache,
        "boundary_imports": boundary_imports,
        "active_roots": sorted(ACTIVE_ROOT_DIRS),
        "boundary_contract": {
            "active_code": sorted(ACTIVE_ROOT_DIRS - {"yolov5"}),
            "vendored_runtime": ["yolov5"],
            "legacy_reference": sorted(LEGACY_REFERENCE_ROOTS),
            "local_data": sorted(LOCAL_DATA_ROOTS),
        },
        "local_data": local_data,
        "legacy_references": legacy_references,
        "issues": issues,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Project Hygiene Report",
        "",
        f"- status: `{report['status']}`",
        f"- unexpected_root_entries: {len(report['unexpected_root_entries'])}",
        f"- generated_root_files: {len(report['generated_root_files'])}",
        f"- stray_pycache_dirs: {len(report['stray_pycache_dirs'])}",
        f"- boundary_imports: {len(report.get('boundary_imports', []))}",
        "",
        "## Issues",
        "",
    ]
    if not report["issues"]:
        lines.append("- No layout or output-governance issues found.")
    else:
        for item in report["issues"]:
            lines.append(f"- `{item['severity']}` {item['category']}: {item['detail']}")
            for path in item["items"][:10]:
                lines.append(f"  - `{path}`")
            if len(item["items"]) > 10:
                lines.append(f"  - ... {len(item['items']) - 10} more")

    contract = report.get("boundary_contract", {})
    lines.extend(["", "## Boundary Contract", ""])
    for name in ("active_code", "vendored_runtime", "legacy_reference", "local_data"):
        values = ", ".join(f"`{value}`" for value in contract.get(name, []))
        lines.append(f"- {name}: {values or '-'}")

    lines.extend(["", "## Local Data Roots", "", "| path | exists | files |", "| --- | --- | ---: |"])
    for name, info in report["local_data"].items():
        lines.append(f"| `{name}` | {json.dumps(info['exists'])} | {info['file_count']} |")

    lines.extend(["", "## Legacy References", "", "| path | exists | files |", "| --- | --- | ---: |"])
    for name, info in report["legacy_references"].items():
        lines.append(f"| `{name}` | {json.dumps(info['exists'])} | {info['file_count']} |")

    lines.append("")
    return "\n".join(lines)
