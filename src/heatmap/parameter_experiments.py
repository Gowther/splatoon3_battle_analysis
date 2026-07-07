from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from src.data_registry import display_path, get_match, load_registry, resolve_project_path
from src.heatmap.annotation_eval import evaluate_annotations
from src.heatmap.config_loader import load_config
from src.heatmap.trajectory_quality import compute_quality


DEFAULT_CANDIDATES = [
    {
        "id": "marker_recall_soft",
        "description": "Lower marker threshold and allow a wider label search radius.",
        "overrides": {
            "marker_detection.min_confidence": 0.40,
            "marker_detection.label_proximity_px": 65,
        },
    },
    {
        "id": "marker_precision_strict",
        "description": "Raise marker and cleaning confidence to reduce false positives.",
        "overrides": {
            "marker_detection.min_confidence": 0.50,
            "point_cleaning.min_confidence": 0.55,
        },
    },
    {
        "id": "track_step_lenient",
        "description": "Allow longer per-frame motion before classifying a jump reset.",
        "overrides": {
            "point_cleaning.max_track_step_px": 500,
            "identity_tracking.route_max_draw_step_px": 160,
        },
    },
    {
        "id": "merge_wider",
        "description": "Merge nearby color components more aggressively.",
        "overrides": {
            "marker_detection.merge_distance_px": 30,
            "point_cleaning.merge_distance_px": 24,
        },
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def annotation_match_ids(annotation_csv: Path) -> list[str]:
    if not annotation_csv.exists():
        return []
    return sorted({row.get("match_id", "") for row in read_csv_rows(annotation_csv) if row.get("match_id")})


def annotation_has_labels(annotation_csv: Path) -> bool:
    if not annotation_csv.exists():
        return False
    for row in read_csv_rows(annotation_csv):
        if row.get("x", "").strip() and row.get("y", "").strip():
            return True
    return False


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def path_under_output(value: str, old_output_dir: str, new_output_dir: str) -> str:
    if value == old_output_dir:
        return new_output_dir
    prefix = old_output_dir.rstrip("/") + "/"
    if value.startswith(prefix):
        return new_output_dir.rstrip("/") + "/" + value[len(prefix) :]
    return value


def rewrite_output_paths(config: dict[str, Any], new_output_dir: str) -> dict[str, Any]:
    output = copy.deepcopy(config)
    old_output_dir = str(output.get("match", {}).get("output_dir", ""))
    output.setdefault("match", {})["output_dir"] = new_output_dir
    for section_name in ("outputs", "state_join", "event_join", "color_calibration"):
        section = output.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in list(section.items()):
            if isinstance(value, str):
                section[key] = path_under_output(value, old_output_dir, new_output_dir)
    return output


def build_variant_config(base_config: dict[str, Any], candidate: dict[str, Any], output_dir: str) -> dict[str, Any]:
    config = rewrite_output_paths(base_config, output_dir)
    for dotted_key, value in candidate.get("overrides", {}).items():
        set_nested(config, dotted_key, value)
    return config


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def selected_registry(registry: dict[str, Any], match_ids: list[str], candidate_id: str, output_root: Path) -> dict[str, Any]:
    selected_matches: list[dict[str, Any]] = []
    for match_id in match_ids:
        match = get_match(registry, match_id)
        if not match:
            continue
        item = copy.deepcopy(match)
        heatmap = item.setdefault("heatmap", {})
        candidate_dir = output_root / match_id / candidate_id
        heatmap["player_tracks"] = str(candidate_dir / "player_tracks.csv")
        heatmap["player_track_gaps"] = str(candidate_dir / "player_track_gaps.csv")
        heatmap["player_routes_dir"] = str(candidate_dir / "player_routes")
        selected_matches.append(item)
    return {"schema_version": registry.get("schema_version", 1), "matches": selected_matches}


def candidate_outputs_ready(registry_payload: dict[str, Any]) -> bool:
    for match in registry_payload.get("matches", []):
        tracks = resolve_project_path(match.get("heatmap", {}).get("player_tracks"))
        if tracks is None or not tracks.exists():
            return False
    return bool(registry_payload.get("matches"))


def trajectory_metrics_for_registry(registry_payload: dict[str, Any]) -> dict[str, Any]:
    per_match: list[dict[str, Any]] = []
    for match in registry_payload.get("matches", []):
        heatmap = match.get("heatmap", {})
        metrics = compute_quality(
            resolve_project_path(heatmap.get("player_tracks")),
            resolve_project_path(heatmap.get("player_track_gaps")),
            resolve_project_path(heatmap.get("player_routes_dir")),
            expected_teams=list(heatmap.get("teams", [])),
        )
        per_match.append({"match_id": match.get("id", ""), "metrics": metrics})
    total_track_rows = sum(int(item["metrics"].get("track_rows") or 0) for item in per_match)
    total_gap_rows = sum(int(item["metrics"].get("gap_rows") or 0) for item in per_match)
    total_jump_reset_rows = sum(int(item["metrics"].get("jump_reset_rows") or 0) for item in per_match)
    return {
        "matches": per_match,
        "aggregate": {
            "track_rows": total_track_rows,
            "gap_rows": total_gap_rows,
            "gap_ratio": round(total_gap_rows / total_track_rows, 4) if total_track_rows else 0.0,
            "jump_reset_rows": total_jump_reset_rows,
            "jump_reset_ratio": round(total_jump_reset_rows / total_track_rows, 4) if total_track_rows else 0.0,
        },
    }


def build_parameter_experiment_plan(
    *,
    annotation_csv: Path,
    registry_path: Path,
    output_root: Path,
    write_configs: bool = False,
    threshold_px: float = 80.0,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    match_ids = annotation_match_ids(annotation_csv)
    has_labels = annotation_has_labels(annotation_csv)
    candidate_specs = candidates or DEFAULT_CANDIDATES
    runs: list[dict[str, Any]] = []

    for candidate in candidate_specs:
        candidate_id = str(candidate["id"])
        registry_payload = selected_registry(registry, match_ids, candidate_id, output_root)
        registry_path_out = output_root / candidate_id / "registry.json"
        config_paths: list[str] = []
        commands: list[str] = []
        for match_id in match_ids:
            match = get_match(registry, match_id)
            heatmap = match.get("heatmap", {}) if match else {}
            config_path_value = heatmap.get("config")
            if not config_path_value:
                continue
            base_config = load_config(config_path_value)
            variant_dir = output_root / match_id / candidate_id
            variant_config = build_variant_config(base_config, candidate, str(variant_dir))
            variant_config_path = output_root / candidate_id / f"{match_id}.yaml"
            config_paths.append(display_path(variant_config_path))
            commands.append(f"python -m src.heatmap.run_pipeline --config {display_path(variant_config_path)} --clean-output")
            if write_configs:
                write_yaml(variant_config_path, variant_config)

        if write_configs:
            registry_path_out.parent.mkdir(parents=True, exist_ok=True)
            registry_path_out.write_text(json.dumps(registry_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        evaluation_command = (
            f"python scripts/evaluate_heatmap_annotations.py {display_path(annotation_csv)} "
            f"--registry {display_path(registry_path_out)} "
            f"--output {display_path(output_root / candidate_id / 'annotation_eval.json')} "
            f"--report {display_path(output_root / candidate_id / 'annotation_eval.md')}"
        )
        commands.append(evaluation_command)
        outputs_ready = candidate_outputs_ready(registry_payload)
        metrics = {}
        trajectory = {}
        run_status = "needs_labels"
        if has_labels and outputs_ready:
            metrics = evaluate_annotations(annotation_csv, registry_payload, threshold_px=threshold_px)
            trajectory = trajectory_metrics_for_registry(registry_payload)
            run_status = "evaluated"
        elif has_labels:
            run_status = "planned"

        runs.append(
            {
                "id": candidate_id,
                "description": candidate.get("description", ""),
                "status": run_status,
                "overrides": candidate.get("overrides", {}),
                "config_paths": config_paths,
                "registry": display_path(registry_path_out),
                "commands": commands,
                "metrics": metrics,
                "trajectory": trajectory,
            }
        )

    status = "needs_labels" if not has_labels else ("ready" if any(run["status"] == "evaluated" for run in runs) else "planned")
    return {
        "schema_version": 1,
        "status": status,
        "annotation_csv": display_path(annotation_csv),
        "output_root": display_path(output_root),
        "match_ids": match_ids,
        "has_labels": has_labels,
        "runs": runs,
        "summary": {
            "candidate_count": len(runs),
            "evaluated": sum(1 for run in runs if run["status"] == "evaluated"),
            "planned": sum(1 for run in runs if run["status"] == "planned"),
            "needs_labels": sum(1 for run in runs if run["status"] == "needs_labels"),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Heatmap Parameter Experiments",
        "",
        f"- status: `{plan.get('status')}`",
        f"- annotation_csv: `{plan.get('annotation_csv', '')}`",
        f"- output_root: `{plan.get('output_root', '')}`",
        f"- matches: {', '.join(plan.get('match_ids', []))}",
        f"- has_labels: {plan.get('has_labels')}",
        "",
        "| status | id | recall | mean error | gap ratio | jump reset ratio | configs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in plan.get("runs", []):
        metrics = run.get("metrics", {})
        aggregate = run.get("trajectory", {}).get("aggregate", {})
        lines.append(
            f"| {run.get('status')} | {run.get('id')} | {metrics.get('recall', '')} | "
            f"{metrics.get('mean_error_px', '')} | {aggregate.get('gap_ratio', '')} | "
            f"{aggregate.get('jump_reset_ratio', '')} | {len(run.get('config_paths', []))} |"
        )
    for run in plan.get("runs", []):
        lines.extend(["", f"## {run['id']}", "", f"- status: `{run['status']}`", ""])
        lines.append(f"- overrides: `{json.dumps(run.get('overrides', {}), ensure_ascii=False)}`")
        if run.get("trajectory", {}).get("aggregate"):
            lines.append(f"- trajectory: `{json.dumps(run['trajectory']['aggregate'], ensure_ascii=False)}`")
        lines.append("")
        lines.extend(f"- `{command}`" for command in run.get("commands", []))
    lines.append("")
    return "\n".join(lines)
