from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.data_registry import DEFAULT_REGISTRY, load_registry


DEFAULT_BASE_CONFIG = "src/heatmap/config_overhead_default.yaml"


def registry_match(registry: dict[str, Any], match_id: str) -> dict[str, Any]:
    for item in registry.get("matches", []):
        if item.get("id") == match_id:
            return item
    raise KeyError(f"Unknown registry match id: {match_id}")


def default_invalid_ranges(start_seconds: float, stop_seconds: float | None, duration_seconds: float | None) -> list[list[float]]:
    ranges: list[list[float]] = []
    if start_seconds > 0:
        ranges.append([0.0, round(max(0.0, start_seconds - 0.1), 1)])
    if stop_seconds is not None and duration_seconds is not None and stop_seconds < duration_seconds:
        tail_start = min(duration_seconds, stop_seconds + 10.0)
        ranges.append([round(tail_start, 1), round(duration_seconds, 1)])
    return ranges


def build_heatmap_config_override(
    registry: dict[str, Any],
    match_id: str,
    *,
    base_config: str = DEFAULT_BASE_CONFIG,
    output_dir: str | None = None,
    start_seconds: float | None = None,
    stop_seconds: float | None = None,
    sample_fps: float | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    match = registry_match(registry, match_id)
    heatmap = match.get("heatmap", {})
    selected_start = float(start_seconds if start_seconds is not None else heatmap.get("start_seconds", 20.0))
    selected_stop = stop_seconds if stop_seconds is not None else heatmap.get("stop_seconds")
    selected_sample_fps = float(sample_fps if sample_fps is not None else heatmap.get("sample_fps", 1.0))
    selected_output_dir = output_dir or heatmap.get("output_dir") or f"outputs/heatmap_{match_id}"

    config: dict[str, Any] = {
        "base_config": base_config,
        "match": {
            "id": match_id,
            "input_video": match.get("video", ""),
            "output_dir": selected_output_dir,
        },
        "sampling": {
            "start_seconds": selected_start,
            "sample_fps": selected_sample_fps,
        },
    }
    if selected_stop is not None:
        config["sampling"]["stop_seconds"] = float(selected_stop)
    if duration_seconds is not None:
        config["video"] = {"duration_seconds": float(duration_seconds)}

    invalid_ranges = default_invalid_ranges(selected_start, float(selected_stop) if selected_stop is not None else None, duration_seconds)
    if invalid_ranges:
        config["frame_quality"] = {"invalid_ranges_seconds": invalid_ranges}
    return config


def render_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_yaml(config), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_default_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return load_registry(path)
