from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from src.core.paths import ROOT


DEFAULT_YOLOV5_ROOT = ROOT / "yolov5"

REQUIRED_RUNTIME_FILES = (
    "hubconf.py",
    "models/common.py",
    "models/experimental.py",
    "models/yolo.py",
    "utils/general.py",
    "utils/torch_utils.py",
)

ALLOWED_TOP_LEVEL_PY = {
    "benchmarks.py",
    "detect.py",
    "export.py",
    "hubconf.py",
    "train.py",
    "val.py",
}

LOCAL_ARTIFACT_PATTERNS = (
    ".DS_Store",
    "*.log",
    "*.pt",
    "*.pth",
    "runs",
)

PROJECT_SCRIPT_PATTERNS = (
    "20*.py",
    "*run_analysis*.py",
    "*splatoon*.py",
)


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def top_level_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((path for path in root.iterdir() if path.is_file()), key=lambda path: path.name)


def missing_runtime_files(root: Path) -> list[str]:
    return [path for path in REQUIRED_RUNTIME_FILES if not (root / path).is_file()]


def project_script_files(root: Path) -> list[str]:
    matches: list[str] = []
    for path in top_level_files(root):
        if path.suffix != ".py":
            continue
        if path.name not in ALLOWED_TOP_LEVEL_PY:
            matches.append(relative_path(root, path))
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in PROJECT_SCRIPT_PATTERNS):
            matches.append(relative_path(root, path))
    return matches


def local_artifacts(root: Path) -> list[str]:
    artifacts: list[str] = []
    if not root.exists():
        return artifacts
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in LOCAL_ARTIFACT_PATTERNS):
            artifacts.append(relative_path(root, path))
    return artifacts


def build_vendor_report(root: Path = DEFAULT_YOLOV5_ROOT) -> dict[str, Any]:
    root = root.expanduser().resolve()
    exists = root.is_dir()
    missing = missing_runtime_files(root)
    project_scripts = project_script_files(root)
    artifacts = local_artifacts(root)

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not exists:
        blockers.append(
            {
                "category": "missing_vendor_root",
                "detail": "YOLOv5 vendor directory is missing",
                "items": [root.as_posix()],
            }
        )
    if missing:
        blockers.append(
            {
                "category": "missing_runtime_files",
                "detail": "required local torch.hub runtime files are missing",
                "items": missing,
            }
        )
    if project_scripts:
        blockers.append(
            {
                "category": "project_scripts_in_vendor",
                "detail": "project-owned scripts should live under src/, scripts/, or legacy/",
                "items": project_scripts,
            }
        )
    if artifacts:
        warnings.append(
            {
                "category": "local_vendor_artifacts",
                "detail": "local/generated files are present under the vendor root",
                "items": artifacts,
            }
        )

    return {
        "status": "failed" if blockers else "passed",
        "root": root.as_posix(),
        "exists": exists,
        "required_runtime_files": list(REQUIRED_RUNTIME_FILES),
        "allowed_top_level_py": sorted(ALLOWED_TOP_LEVEL_PY),
        "missing_runtime_files": missing,
        "project_script_files": project_scripts,
        "local_artifacts": artifacts,
        "blockers": blockers,
        "warnings": warnings,
    }


def ensure_vendor_ready(root: Path = DEFAULT_YOLOV5_ROOT) -> Path:
    report = build_vendor_report(root)
    if report["status"] != "passed":
        details = "; ".join(
            f"{item['category']}: {', '.join(item['items'])}" for item in report["blockers"]
        )
        raise FileNotFoundError(f"YOLOv5 vendor runtime is not ready: {details}")
    return Path(report["root"])


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# YOLOv5 Vendor Boundary Report",
        "",
        f"- status: `{report['status']}`",
        f"- root: `{report['root']}`",
        f"- required_runtime_files: {len(report['required_runtime_files'])}",
        f"- local_artifacts: {len(report['local_artifacts'])}",
        "",
        "## Contract",
        "",
        "- `yolov5/` is a vendored runtime dependency used by `torch.hub.load(..., source=\"local\")`.",
        "- Project-owned analysis, training, and reporting code should live under `src/`, `scripts/`, or `legacy/`.",
        "- Local weights, logs, runs, and generated files under `yolov5/` are tolerated for local work but should not become supported runtime paths.",
        "",
        "## Blockers",
        "",
    ]
    if not report["blockers"]:
        lines.append("- No vendor-boundary blockers found.")
    else:
        for item in report["blockers"]:
            lines.append(f"- `{item['category']}`: {item['detail']}")
            for path in item["items"]:
                lines.append(f"  - `{path}`")

    lines.extend(["", "## Warnings", ""])
    if not report["warnings"]:
        lines.append("- No local vendor artifacts found.")
    else:
        for item in report["warnings"]:
            lines.append(f"- `{item['category']}`: {item['detail']}")
            for path in item["items"][:20]:
                lines.append(f"  - `{path}`")
            if len(item["items"]) > 20:
                lines.append(f"  - ... {len(item['items']) - 20} more")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
