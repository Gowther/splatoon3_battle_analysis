from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.core.paths import ROOT, project_path
from src.data_registry import display_path
from src.heatmap.stage_coordinates import (
    DEFAULT_CONTROL_POINT_ASSET_DIR,
    StageBox,
    homography_from_control_points,
    load_control_point_asset,
    normalize_point_homography,
    parse_control_point,
    stage_box_from_config,
)


DEFAULT_REGISTRY_PATH = ROOT / "config" / "stage_registry.json"
# Two independent labelings of the same stage should agree closely. A quarter of
# the stage width is not a labeling wobble, it is a different landmark.
DEFAULT_MAX_DISAGREEMENT = 0.05
AGREEMENT_GRID = 9


def load_stage_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    resolved = project_path(path)
    if not resolved.is_file():
        return {"schema_version": 1, "stages": []}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stage registry must be an object: {display_path(resolved)}")
    payload.setdefault("stages", [])
    return payload


def write_stage_registry(registry: Mapping[str, Any], path: Path | str = DEFAULT_REGISTRY_PATH) -> Path:
    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def stage_entry(registry: Mapping[str, Any], stage_id: str) -> dict[str, Any] | None:
    for stage in registry.get("stages", []):
        if isinstance(stage, Mapping) and stage.get("stage_id") == stage_id:
            return dict(stage)
    return None


def stage_for_match(registry: Mapping[str, Any], match_id: str) -> dict[str, Any] | None:
    for stage in registry.get("stages", []):
        if isinstance(stage, Mapping) and match_id in stage.get("matches", []):
            return dict(stage)
    return None


def register_match(
    registry: Mapping[str, Any],
    stage_id: str,
    match_id: str,
    *,
    asset_path: str = "",
) -> dict[str, Any]:
    """Attach a match to a stage, keeping one canonical asset per stage."""
    output = {**registry, "stages": [dict(stage) for stage in registry.get("stages", [])]}
    for stage in output["stages"]:
        if stage.get("stage_id") != stage_id and match_id in stage.get("matches", []):
            stage["matches"] = [item for item in stage["matches"] if item != match_id]

    for stage in output["stages"]:
        if stage.get("stage_id") == stage_id:
            matches = list(stage.get("matches", []))
            if match_id not in matches:
                matches.append(match_id)
            stage["matches"] = sorted(matches)
            if asset_path:
                stage["control_point_asset"] = asset_path
            return output

    output["stages"].append(
        {
            "stage_id": stage_id,
            "matches": [match_id],
            "control_point_asset": asset_path,
        }
    )
    output["stages"].sort(key=lambda stage: stage.get("stage_id", ""))
    return output


def resolve_stage_asset(
    registry: Mapping[str, Any],
    match_id: str,
    *,
    asset_dir: Path | str = DEFAULT_CONTROL_POINT_ASSET_DIR,
) -> dict[str, Any] | None:
    """Find the control-point asset a match inherits from its stage."""
    stage = stage_for_match(registry, match_id)
    if stage is None:
        return None
    candidates: list[Path] = []
    declared = stage.get("control_point_asset", "")
    if declared:
        candidates.append(project_path(declared))
    stage_id = stage.get("stage_id", "")
    if stage_id:
        candidates.append(project_path(asset_dir) / f"{stage_id}.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            asset = load_control_point_asset(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if asset.get("template"):
            continue
        asset["stage_id"] = asset.get("stage_id") or stage_id
        asset["inherited_from_stage"] = stage_id
        asset["shared_with_matches"] = [item for item in stage.get("matches", []) if item != match_id]
        return asset
    return None


def sample_grid(source_box: StageBox, divisions: int = AGREEMENT_GRID) -> list[tuple[float, float]]:
    steps = max(2, divisions)
    return [
        (
            source_box.x1 + source_box.width * column / (steps - 1),
            source_box.y1 + source_box.height * row / (steps - 1),
        )
        for column in range(steps)
        for row in range(steps)
    ]


def compare_control_points(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    source_box: StageBox,
    *,
    max_disagreement: float = DEFAULT_MAX_DISAGREEMENT,
    divisions: int = AGREEMENT_GRID,
) -> dict[str, Any]:
    """Map a grid of ROI points through both homographies and measure disagreement.

    Two labelings of the same stage should send the same pixel to the same stage
    coordinate. A mislabeled landmark passes every single-asset check but shows up
    here as a large disagreement.
    """
    try:
        left_matrix = homography_from_control_points([parse_control_point(point) for point in left])
        right_matrix = homography_from_control_points([parse_control_point(point) for point in right])
    except ValueError as exc:
        return {"status": "invalid", "error": str(exc), "max_disagreement": None, "limit": max_disagreement}

    worst = 0.0
    worst_point: tuple[float, float] | None = None
    total = 0.0
    samples = sample_grid(source_box, divisions)
    for x, y in samples:
        try:
            left_stage = normalize_point_homography(x, y, left_matrix)
            right_stage = normalize_point_homography(x, y, right_matrix)
        except ValueError:
            return {"status": "invalid", "error": "homography produced an invalid point", "max_disagreement": None, "limit": max_disagreement}
        distance = (
            (left_stage["stage_x"] - right_stage["stage_x"]) ** 2
            + (left_stage["stage_y"] - right_stage["stage_y"]) ** 2
        ) ** 0.5
        total += distance
        if distance > worst:
            worst = distance
            worst_point = (x, y)
    return {
        "status": "ready" if worst <= max_disagreement else "disagrees",
        "max_disagreement": worst,
        "mean_disagreement": total / len(samples),
        "limit": max_disagreement,
        "worst_source_point": list(worst_point) if worst_point else [],
        "sampled_points": len(samples),
    }


def build_stage_registry_report(
    registry: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    asset_dir: Path | str = DEFAULT_CONTROL_POINT_ASSET_DIR,
    max_disagreement: float = DEFAULT_MAX_DISAGREEMENT,
) -> dict[str, Any]:
    """Report stage reuse coverage and cross-validate every candidate asset per stage."""
    stages: list[dict[str, Any]] = []
    for stage in registry.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_id = str(stage.get("stage_id", ""))
        matches = [str(item) for item in stage.get("matches", [])]
        canonical = resolve_stage_asset(registry, matches[0], asset_dir=asset_dir) if matches else None

        candidates: list[dict[str, Any]] = []
        for match_id in matches:
            candidate_path = project_path(asset_dir) / f"{match_id}.json"
            if not candidate_path.is_file():
                continue
            try:
                candidate = load_control_point_asset(candidate_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if candidate.get("template"):
                continue
            candidates.append({"match_id": match_id, "asset": candidate})

        comparisons: list[dict[str, Any]] = []
        if canonical is not None:
            for candidate in candidates:
                config = configs.get(candidate["match_id"])
                if config is None:
                    continue
                try:
                    source_box = stage_box_from_config(config)
                except (KeyError, TypeError, ValueError):
                    continue
                result = compare_control_points(
                    canonical.get("control_points", []),
                    candidate["asset"].get("control_points", []),
                    source_box,
                    max_disagreement=max_disagreement,
                )
                comparisons.append({"match_id": candidate["match_id"], **result})

        disagreeing = [item["match_id"] for item in comparisons if item.get("status") != "ready"]
        stages.append(
            {
                "stage_id": stage_id,
                "matches": matches,
                "match_count": len(matches),
                "canonical_asset": canonical.get("path", "") if canonical else "",
                "has_asset": canonical is not None,
                "per_match_assets": len(candidates),
                "comparisons": comparisons,
                "disagreeing_matches": disagreeing,
                "status": (
                    "needs_asset"
                    if canonical is None
                    else ("disagrees" if disagreeing else "ready")
                ),
            }
        )

    registered_matches = sorted({match for stage in stages for match in stage["matches"]})
    unregistered = sorted(set(configs) - set(registered_matches))
    failing = [stage["stage_id"] for stage in stages if stage["status"] != "ready"]
    return {
        "schema_version": 1,
        "status": "ready" if stages and not failing else ("needs_review" if stages else "empty"),
        "stage_count": len(stages),
        "registered_matches": registered_matches,
        "unregistered_matches": unregistered,
        "failing_stages": failing,
        "stages": stages,
    }


def render_registry_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage Registry",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- stages: {report.get('stage_count', 0)}",
        f"- registered matches: {len(report.get('registered_matches', []))}",
        f"- unregistered matches: {', '.join(report.get('unregistered_matches', [])) or 'none'}",
        f"- failing stages: {', '.join(report.get('failing_stages', [])) or 'none'}",
        "",
        "## Stages",
        "",
        "| stage | matches | asset | per-match assets | status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage in report.get("stages", []):
        lines.append(
            f"| {stage['stage_id']} | {stage['match_count']} | "
            f"`{stage['canonical_asset'] or '(none)'}` | {stage['per_match_assets']} | `{stage['status']}` |"
        )
    comparisons = [
        (stage["stage_id"], item)
        for stage in report.get("stages", [])
        for item in stage.get("comparisons", [])
    ]
    if comparisons:
        lines.extend(
            [
                "",
                "## Cross Validation",
                "",
                "A mislabeled landmark passes every single-asset check. It shows up here",
                "as two labelings of the same stage sending the same pixel to different",
                "stage coordinates.",
                "",
                "| stage | match | max disagreement | limit | status |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for stage_id, item in comparisons:
            value = item.get("max_disagreement")
            rendered = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
            lines.append(
                f"| {stage_id} | {item['match_id']} | {rendered} | {item.get('limit', 0):.4f} | `{item.get('status', '')}` |"
            )
    lines.append("")
    return "\n".join(lines)
