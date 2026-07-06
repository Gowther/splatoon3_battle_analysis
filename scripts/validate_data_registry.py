from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, display_path, load_registry, resolve_project_path


HEATMAP_REQUIRED_ARTIFACTS = (
    "config",
    "output_dir",
    "report",
    "color_report",
    "player_tracks",
    "player_track_gaps",
    "player_routes_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local Splatoon 3 data registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, help="Optional JSON summary output.")
    parser.add_argument("--report", type=Path, help="Optional Markdown report output.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when required paths are missing.")
    return parser.parse_args()


def csv_row_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def path_summary(path_value: str | None, required: bool = True) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    exists = bool(path and path.exists())
    summary: dict[str, Any] = {
        "path": display_path(path),
        "required": required,
        "exists": exists,
    }
    if exists and path is not None:
        if path.is_file():
            summary["size_bytes"] = path.stat().st_size
            row_count = csv_row_count(path)
            if row_count is not None:
                summary["csv_rows"] = row_count
        elif path.is_dir():
            summary["file_count"] = sum(1 for child in path.iterdir() if child.is_file())
    return summary


def validate_match(match: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {
        "video": path_summary(match.get("video")),
    }
    for window in match.get("analysis_windows", []):
        artifacts[f"analysis_window:{window['id']}"] = {
            "path": "",
            "required": False,
            "exists": True,
            "start_seconds": window.get("start_seconds"),
            "stop_seconds": window.get("stop_seconds"),
            "sample_fps": window.get("sample_fps"),
            "device": window.get("device"),
        }

    heatmap = match.get("heatmap")
    if isinstance(heatmap, dict):
        for key in HEATMAP_REQUIRED_ARTIFACTS:
            artifacts[f"heatmap:{key}"] = path_summary(heatmap.get(key))

    missing = [name for name, artifact in artifacts.items() if artifact.get("required") and not artifact.get("exists")]
    return {
        "id": match["id"],
        "purpose": match.get("purpose", []),
        "status": "failed" if missing else "passed",
        "missing": missing,
        "artifacts": artifacts,
    }


def write_report(path: Path, registry_path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Data Registry Report",
        "",
        f"- registry: {display_path(registry_path)}",
        f"- matches: {len(results)}",
        "",
        "| match | status | purpose | missing |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| {id} | {status} | {purpose} | {missing} |".format(
                id=result["id"],
                status=result["status"],
                purpose=", ".join(result.get("purpose", [])),
                missing=", ".join(result.get("missing", [])) or "none",
            )
        )
    lines.extend(["", "## Artifacts", ""])
    for result in results:
        lines.extend([f"### {result['id']}", ""])
        for name, artifact in result["artifacts"].items():
            details = []
            if "csv_rows" in artifact:
                details.append(f"rows={artifact['csv_rows']}")
            if "file_count" in artifact:
                details.append(f"files={artifact['file_count']}")
            if "size_bytes" in artifact:
                details.append(f"bytes={artifact['size_bytes']}")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(f"- {name}: `{artifact.get('path', '')}` - {artifact['exists']}{suffix}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    registry_path = resolve_project_path(args.registry) or args.registry.expanduser()
    registry = load_registry(registry_path)
    results = [validate_match(match) for match in registry.get("matches", [])]
    summary = {
        "registry": display_path(registry_path),
        "status": "failed" if any(result["status"] == "failed" for result in results) else "passed",
        "matches": results,
    }

    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote registry summary: {output}")

    if args.report:
        report = args.report.expanduser()
        write_report(report, registry_path, results)
        print(f"wrote registry report: {report}")

    print(f"registry status: {summary['status']}")
    for result in results:
        print(f"- {result['id']}: {result['status']}")
    if args.strict and summary["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
