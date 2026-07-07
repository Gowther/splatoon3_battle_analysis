from __future__ import annotations

from typing import Any


def validation_analysis_ids(registry: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for match in registry.get("matches", []):
        match_id = str(match.get("id", ""))
        if not match_id.startswith("n_match_"):
            continue
        best_prefix = f"{match_id}_best_"
        best = [str(window.get("id")) for window in match.get("analysis_windows", []) if str(window.get("id", "")).startswith(best_prefix)]
        ids.extend(best[-1:])
    return ids


def validation_heatmap_ids(registry: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for match in registry.get("matches", []):
        match_id = str(match.get("id", ""))
        heatmap = match.get("heatmap")
        if match_id.startswith("f_match_") and isinstance(heatmap, dict) and heatmap.get("id"):
            ids.append(str(heatmap["id"]))
    return ids


def validation_ids(registry: dict[str, Any]) -> list[str]:
    return validation_analysis_ids(registry) + validation_heatmap_ids(registry)
