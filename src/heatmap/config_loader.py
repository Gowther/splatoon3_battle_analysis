from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from src.core.paths import project_path as resolve_path

def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _format_value(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError:
            return value
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, context) for key, item in value.items()}
    return value


def _format_placeholders(config: Mapping[str, Any]) -> dict[str, Any]:
    match = dict(config.get("match", {}))
    match_context = {
        "match_id": match.get("id", ""),
        "input_video": match.get("input_video", ""),
    }
    formatted_match = _format_value(match, match_context)
    context = {
        **match_context,
        "output_dir": formatted_match.get("output_dir", ""),
    }
    return _format_value(dict(config), context)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in heatmap config: {path}")
    return loaded


def _load_config_raw(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = read_yaml(config_path)
    base_config = config.pop("base_config", None)
    if base_config:
        base = _load_config_raw(resolve_path(base_config))
        config = deep_merge(base, config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load_config_raw(path)
    return _format_placeholders(config)
