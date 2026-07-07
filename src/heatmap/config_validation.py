from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.data_registry import iter_heatmap_matches, resolve_project_path
from src.heatmap.config_loader import load_config


REQUIRED_SECTIONS = (
    "match",
    "video",
    "sampling",
    "map_view",
    "frame_quality",
    "teams",
    "marker_detection",
    "point_cleaning",
    "rendering",
    "state_join",
    "outputs",
)

REQUIRED_OUTPUT_KEYS = (
    "frames_dir",
    "valid_frames_csv",
    "clean_points_csv",
    "tracks_csv",
    "rendered_dir",
    "player_tracks_csv",
    "report_md",
)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def config_paths_from_registry(registry: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for _match, heatmap in iter_heatmap_matches(registry):
        config_path = resolve_project_path(heatmap.get("config"))
        if config_path and config_path not in seen:
            paths.append(config_path)
            seen.add(config_path)
    return paths


def validate_heatmap_config(path: Path) -> dict[str, Any]:
    problems: list[str] = []
    try:
        config = load_config(path)
    except Exception as exc:  # pragma: no cover - exact parser errors are not important here.
        return {"path": str(path), "status": "failed", "problems": [str(exc)]}

    for section in REQUIRED_SECTIONS:
        if section not in config:
            problems.append(f"missing section: {section}")

    for key in REQUIRED_OUTPUT_KEYS:
        if not config.get("outputs", {}).get(key):
            problems.append(f"missing outputs.{key}")

    if not config.get("match", {}).get("id"):
        problems.append("missing match.id")
    if not config.get("match", {}).get("input_video"):
        problems.append("missing match.input_video")

    unresolved = sorted({item for item in _strings(config) if "{" in item or "}" in item})
    if unresolved:
        problems.extend(f"unresolved placeholder: {item}" for item in unresolved)

    return {
        "path": str(path),
        "match_id": config.get("match", {}).get("id", ""),
        "output_dir": config.get("match", {}).get("output_dir", ""),
        "status": "failed" if problems else "passed",
        "problems": problems,
    }


def validate_heatmap_configs(paths: Iterable[Path]) -> dict[str, Any]:
    configs = [validate_heatmap_config(path) for path in paths]
    status = "passed" if all(item["status"] == "passed" for item in configs) else "failed"
    return {"status": status, "configs": configs}
