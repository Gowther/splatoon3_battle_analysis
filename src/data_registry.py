from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.core.paths import ROOT, display_path, resolve_project_path

DEFAULT_REGISTRY = ROOT / "config" / "data_registry.json"


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = resolve_project_path(path, ROOT) if path else DEFAULT_REGISTRY
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY
    with registry_path.open(encoding="utf-8") as f:
        return json.load(f)


def matches_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {match["id"]: match for match in registry.get("matches", [])}


def heatmap_entry(match: dict[str, Any]) -> dict[str, Any] | None:
    heatmap = match.get("heatmap")
    return heatmap if isinstance(heatmap, dict) else None


def iter_heatmap_matches(registry: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for match in registry.get("matches", []):
        heatmap = heatmap_entry(match)
        if heatmap:
            yield match, heatmap


def get_match(registry: dict[str, Any], match_id: str) -> dict[str, Any] | None:
    return matches_by_id(registry).get(match_id)
