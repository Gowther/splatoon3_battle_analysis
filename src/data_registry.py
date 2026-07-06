from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "data_registry.json"


def resolve_project_path(value: str | Path | None, root: Path = ROOT) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def display_path(path: Path | None, root: Path = ROOT) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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
