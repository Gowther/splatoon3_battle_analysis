from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path
from src.data_registry import DEFAULT_REGISTRY, display_path, load_registry, resolve_project_path
from src.match_intake import DEFAULT_EVALUATION_CONFIG
from src.weapon_training import summarize_dataset


DEFAULT_EVALUATION_RESULTS = ROOT / "outputs" / "evaluation" / "evaluation_results.json"
DEFAULT_ASSET_GROUPS = {
    "models": ROOT / "models",
    "footages": ROOT / "footages",
    "weapon_icons": ROOT / "main_icons",
    "weapon_dataset": ROOT / "main_training_dataset",
    "outputs": ROOT / "outputs",
}


def load_json_if_exists(path: Path) -> Any | None:
    resolved = project_path(path)
    if not resolved.exists():
        return None
    with resolved.open(encoding="utf-8") as f:
        return json.load(f)


def file_count_and_bytes(path: Path) -> tuple[int, int, Counter[str]]:
    if not path.exists():
        return 0, 0, Counter()
    files = [item for item in path.rglob("*") if item.is_file()]
    suffixes = Counter(item.suffix.lower() or "<none>" for item in files)
    return len(files), sum(item.stat().st_size for item in files), suffixes


def asset_summary(groups: dict[str, Path] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in (groups or DEFAULT_ASSET_GROUPS).items():
        count, total_bytes, suffixes = file_count_and_bytes(path)
        result[name] = {
            "root": display_path(path),
            "exists": path.exists(),
            "file_count": count,
            "total_bytes": total_bytes,
            "top_extensions": dict(suffixes.most_common(8)),
        }
    return result


def registry_summary(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = project_path(path)
    registry = load_registry(registry_path)
    matches = registry.get("matches", [])
    missing_videos: list[dict[str, str]] = []
    purpose_counts: Counter[str] = Counter()
    analysis_windows = 0
    heatmap_matches = 0

    for match in matches:
        video = resolve_project_path(match.get("video"))
        if not video or not video.exists():
            missing_videos.append({"id": match.get("id", ""), "video": display_path(video)})
        purpose_counts.update(match.get("purpose", []))
        analysis_windows += len(match.get("analysis_windows", []))
        if isinstance(match.get("heatmap"), dict):
            heatmap_matches += 1

    return {
        "path": display_path(registry_path),
        "match_count": len(matches),
        "analysis_window_count": analysis_windows,
        "heatmap_match_count": heatmap_matches,
        "missing_videos": missing_videos,
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "status": "passed" if not missing_videos else "failed",
    }


def evaluation_config_summary(path: Path = DEFAULT_EVALUATION_CONFIG) -> dict[str, Any]:
    config_path = project_path(path)
    config = load_json_if_exists(config_path) or {}
    analysis_ids = [match.get("id", "") for match in config.get("analysis_matches", [])]
    heatmap_ids = [match.get("id", "") for match in config.get("heatmap_matches", [])]
    return {
        "path": display_path(config_path),
        "exists": config_path.exists(),
        "analysis_match_count": len(analysis_ids),
        "heatmap_match_count": len(heatmap_ids),
        "analysis_ids": analysis_ids,
        "heatmap_ids": heatmap_ids,
        "result_ids": [*analysis_ids, *heatmap_ids],
        "defaults": config.get("defaults", {}),
    }


def evaluation_results_summary(
    path: Path = DEFAULT_EVALUATION_RESULTS,
    configured_ids: list[str] | None = None,
) -> dict[str, Any]:
    results_path = project_path(path)
    expected_ids = set(configured_ids or [])
    results = load_json_if_exists(results_path)
    if results is None:
        return {
            "path": display_path(results_path),
            "exists": False,
            "status": "missing",
            "status_counts": {},
            "kind_counts": {},
            "result_ids": [],
            "missing_configured_results": sorted(expected_ids),
            "extra_results": [],
            "problems": [],
        }

    status_counts = Counter(str(item.get("status", "unknown")) for item in results)
    kind_counts = Counter(str(item.get("kind", "unknown")) for item in results)
    result_ids = {str(item.get("id", "")) for item in results if item.get("id")}
    missing_configured_results = sorted(expected_ids - result_ids)
    extra_results = sorted(result_ids - expected_ids) if expected_ids else []
    problems = [
        {
            "kind": item.get("kind", ""),
            "id": item.get("id", ""),
            "status": item.get("status", ""),
            "reason": item.get("reason", ""),
            "notes": item.get("notes", []),
        }
        for item in results
        if item.get("status") != "passed"
    ]
    status = "passed" if results and not problems and not missing_configured_results else "failed"
    return {
        "path": display_path(results_path),
        "exists": True,
        "status": status,
        "status_counts": dict(sorted(status_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "result_ids": sorted(result_ids),
        "missing_configured_results": missing_configured_results,
        "extra_results": extra_results,
        "problems": problems,
    }


def weapon_training_summary(dataset: Path, labels: Path, model: Path | None) -> dict[str, Any]:
    try:
        summary = summarize_dataset(project_path(dataset), project_path(labels), project_path(model) if model else None)
        payload = asdict(summary)
        payload["status"] = "passed" if summary.ok else "failed"
        return payload
    except Exception as exc:
        return {
            "dataset": display_path(project_path(dataset)),
            "labels": display_path(project_path(labels)),
            "model": display_path(project_path(model)) if model else "",
            "status": "failed",
            "error": str(exc),
        }


def recommendation_list(payload: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    registry = payload["registry"]
    evaluation = payload["evaluation_results"]
    weapon = payload["weapon_training"]

    if registry["missing_videos"]:
        recommendations.append("Fix missing registry videos before promoting new evaluation data.")
    if not evaluation["exists"]:
        recommendations.append("Run scripts/evaluate_matches.py and pass --evaluation-results to this report.")
    elif evaluation["missing_configured_results"]:
        recommendations.append("Regenerate the full fixed evaluation so every configured match has a result.")
    elif evaluation["problems"]:
        recommendations.append("Review failed or skipped evaluation items before changing models.")
    if weapon["status"] != "passed":
        recommendations.append("Run scripts/plan_weapon_training.py --strict and repair dataset/label/model mismatches.")
    if not recommendations:
        recommendations.append("Current baseline is ready for incremental data collection or targeted model experiments.")
    return recommendations


def overall_status(payload: dict[str, Any]) -> str:
    if payload["registry"]["status"] == "failed":
        return "failed"
    if payload["weapon_training"]["status"] == "failed":
        return "failed"
    evaluation_status = payload["evaluation_results"]["status"]
    if evaluation_status == "failed":
        return "failed"
    if evaluation_status == "missing":
        return "needs_evaluation"
    return "passed"


def build_quality_payload(
    registry: Path = DEFAULT_REGISTRY,
    evaluation_config: Path = DEFAULT_EVALUATION_CONFIG,
    evaluation_results: Path = DEFAULT_EVALUATION_RESULTS,
    dataset: Path = ROOT / "main_training_dataset",
    labels: Path = ROOT / "main_weapon_list.txt",
    weapon_model: Path | None = ROOT / "models" / "main_weapons_classification_weight.pth",
) -> dict[str, Any]:
    evaluation_config_payload = evaluation_config_summary(evaluation_config)
    payload = {
        "registry": registry_summary(registry),
        "evaluation_config": evaluation_config_payload,
        "evaluation_results": evaluation_results_summary(
            evaluation_results,
            configured_ids=evaluation_config_payload["result_ids"],
        ),
        "weapon_training": weapon_training_summary(dataset, labels, weapon_model),
        "assets": asset_summary(),
    }
    payload["overall_status"] = overall_status(payload)
    payload["recommendations"] = recommendation_list(payload)
    return payload


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def render_markdown(payload: dict[str, Any]) -> str:
    registry = payload["registry"]
    eval_config = payload["evaluation_config"]
    eval_results = payload["evaluation_results"]
    weapon = payload["weapon_training"]

    lines = [
        "# Splatoon 3 Quality Overview",
        "",
        f"- overall_status: `{payload['overall_status']}`",
        "",
        "## Data Registry",
        "",
        f"- registry: `{registry['path']}`",
        f"- matches: {registry['match_count']}",
        f"- analysis_windows: {registry['analysis_window_count']}",
        f"- heatmap_matches: {registry['heatmap_match_count']}",
        f"- missing_videos: {len(registry['missing_videos'])}",
        "",
        "## Evaluation",
        "",
        f"- config: `{eval_config['path']}`",
        f"- configured_analysis_matches: {eval_config['analysis_match_count']}",
        f"- configured_heatmap_matches: {eval_config['heatmap_match_count']}",
        f"- results: `{eval_results['path']}`",
        f"- results_status: `{eval_results['status']}`",
        f"- status_counts: {json.dumps(eval_results['status_counts'] or {}, ensure_ascii=False)}",
        f"- kind_counts: {json.dumps(eval_results['kind_counts'] or {}, ensure_ascii=False)}",
        f"- missing_configured_results: {len(eval_results['missing_configured_results'])}",
        "",
        "## Weapon Classifier",
        "",
        f"- status: `{weapon['status']}`",
        f"- images: {weapon.get('images', 0)}",
        f"- dataset_classes: {weapon.get('dataset_classes', 0)}",
        f"- label_classes: {weapon.get('label_classes', 0)}",
        f"- model_output_classes: {weapon.get('model_output_classes')}",
        "",
        "## Assets",
        "",
        "| group | exists | files | bytes |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, group in payload["assets"].items():
        lines.append(
            f"| {name} | {group['exists']} | {group['file_count']} | {format_bytes(group['total_bytes'])} |"
        )

    if eval_results["problems"]:
        lines.extend(["", "## Evaluation Problems", ""])
        for item in eval_results["problems"]:
            detail = item.get("reason") or "; ".join(item.get("notes", [])) or "no detail"
            lines.append(f"- {item['kind']} `{item['id']}`: {item['status']} ({detail})")

    if eval_results["missing_configured_results"]:
        lines.extend(["", "## Missing Evaluation Results", ""])
        for item in eval_results["missing_configured_results"]:
            lines.append(f"- `{item}`")

    if registry["missing_videos"]:
        lines.extend(["", "## Missing Registry Videos", ""])
        for item in registry["missing_videos"]:
            lines.append(f"- `{item['id']}`: `{item['video']}`")

    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload["recommendations"])
    return "\n".join(lines) + "\n"
