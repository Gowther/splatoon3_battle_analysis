from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.paths import ROOT, project_path
from src.data_registry import DEFAULT_REGISTRY, display_path


DEFAULT_EVALUATION_CONFIG = ROOT / "config" / "evaluation_matches.json"


class IntakeConflict(ValueError):
    """Raised when an intake update would overwrite existing registry data."""


@dataclass(frozen=True)
class IntakePaths:
    registry: Path = DEFAULT_REGISTRY
    evaluation_config: Path = DEFAULT_EVALUATION_CONFIG


def load_json(path: Path) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def display_project_path(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    return display_path(project_path(path))


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def seconds_slug(value: float | None, fallback: str) -> str:
    if value is None:
        return fallback
    text = f"{value:g}".replace("-", "m")
    return text.replace(".", "p")


def default_analysis_id(match_id: str, start_seconds: float | None, stop_seconds: float | None) -> str:
    if start_seconds is None and stop_seconds is None:
        return f"{match_id}_analysis"
    return f"{match_id}_{seconds_slug(start_seconds, 'start')}_{seconds_slug(stop_seconds, 'end')}"


def numeric_field(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def build_analysis_window(
    analysis_id: str,
    start_seconds: float | None,
    stop_seconds: float | None,
    sample_fps: float | None,
    device: str | None,
) -> dict[str, Any]:
    window: dict[str, Any] = {"id": analysis_id}
    for key, value in (
        ("start_seconds", numeric_field(start_seconds)),
        ("stop_seconds", numeric_field(stop_seconds)),
        ("sample_fps", numeric_field(sample_fps)),
    ):
        if value is not None:
            window[key] = value
    if device:
        window["device"] = device
    return window


def build_registry_match(
    match_id: str,
    video: str | Path,
    purpose: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    stage: str | None = None,
    analysis_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": match_id,
        "video": display_project_path(video),
        "purpose": unique_strings(purpose or ["analysis_candidate"]),
    }
    if notes:
        entry["notes"] = notes
    if mode:
        entry["mode"] = mode
    if stage:
        entry["stage"] = stage
    if analysis_window:
        entry["analysis_windows"] = [analysis_window]
    return entry


def build_evaluation_match(
    analysis_id: str,
    video: str | Path,
    start_seconds: float | None,
    stop_seconds: float | None,
    sample_fps: float | None,
    device: str | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": analysis_id,
        "input": display_project_path(video),
    }
    for key, value in (
        ("start_seconds", numeric_field(start_seconds)),
        ("stop_seconds", numeric_field(stop_seconds)),
        ("sample_fps", numeric_field(sample_fps)),
    ):
        if value is not None:
            entry[key] = value
    if device:
        entry["device"] = device
    return entry


def probe_video(path: str | Path) -> dict[str, Any]:
    video_path = project_path(path)
    result: dict[str, Any] = {
        "path": display_path(video_path),
        "exists": video_path.exists(),
        "readable": False,
    }
    if not video_path.exists():
        return result

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime extras
        result["error"] = f"cv2 import failed: {exc}"
        return result

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            result["error"] = "cv2 could not open the video"
            return result

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        result.update(
            {
                "readable": True,
                "fps": fps if fps > 0 else None,
                "frame_count": frame_count if frame_count > 0 else None,
                "duration_seconds": round(frame_count / fps, 3) if fps > 0 and frame_count > 0 else None,
                "width": width if width > 0 else None,
                "height": height if height > 0 else None,
            }
        )
    finally:
        capture.release()
    return result


def find_index_by_id(items: list[dict[str, Any]], entry_id: str) -> int | None:
    for index, item in enumerate(items):
        if item.get("id") == entry_id:
            return index
    return None


def upsert_entry(
    items: list[dict[str, Any]],
    entry: dict[str, Any],
    collection_name: str,
    replace: bool = False,
) -> str:
    index = find_index_by_id(items, entry["id"])
    if index is None:
        items.append(entry)
        return "added"
    if items[index] == entry:
        return "unchanged"
    if replace:
        items[index] = entry
        return "replaced"
    raise IntakeConflict(f"{collection_name} already has id {entry['id']}; pass --replace to overwrite it")


def merge_analysis_windows(
    existing_windows: list[dict[str, Any]],
    proposed_windows: list[dict[str, Any]],
    replace: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    merged = copy.deepcopy(existing_windows)
    statuses: list[str] = []
    for window in proposed_windows:
        statuses.append(upsert_entry(merged, window, "analysis_windows", replace=replace))
    if statuses == ["unchanged"]:
        return merged, "unchanged"
    if "replaced" in statuses:
        return merged, "replaced"
    return merged, "updated"


def merge_registry_match(
    existing: dict[str, Any],
    proposed: dict[str, Any],
    replace: bool = False,
) -> tuple[dict[str, Any], str]:
    if replace:
        merged = copy.deepcopy(proposed)
        if "heatmap" in existing and "heatmap" not in merged:
            merged["heatmap"] = copy.deepcopy(existing["heatmap"])
        return merged, "replaced"

    merged = copy.deepcopy(existing)
    for key in ("video", "notes", "mode", "stage"):
        proposed_value = proposed.get(key)
        if proposed_value in (None, ""):
            continue
        existing_value = merged.get(key)
        if existing_value in (None, ""):
            merged[key] = proposed_value
        elif existing_value != proposed_value:
            raise IntakeConflict(
                f"registry match {proposed['id']} has different {key}: "
                f"{existing_value!r} vs {proposed_value!r}; pass --replace to overwrite it"
            )

    merged["purpose"] = unique_strings([*merged.get("purpose", []), *proposed.get("purpose", [])])
    if proposed.get("analysis_windows"):
        windows, window_status = merge_analysis_windows(
            merged.get("analysis_windows", []),
            proposed["analysis_windows"],
            replace=False,
        )
        merged["analysis_windows"] = windows
        if window_status != "unchanged":
            return merged, window_status
    return merged, "unchanged" if merged == existing else "updated"


def upsert_registry_match(
    registry: dict[str, Any],
    entry: dict[str, Any],
    replace: bool = False,
) -> str:
    matches = registry.setdefault("matches", [])
    index = find_index_by_id(matches, entry["id"])
    if index is None:
        matches.append(entry)
        return "added"
    merged, status = merge_registry_match(matches[index], entry, replace=replace)
    matches[index] = merged
    return status


def build_intake_plan(
    match_id: str,
    video: str | Path,
    purpose: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    stage: str | None = None,
    analysis_id: str | None = None,
    start_seconds: float | None = None,
    stop_seconds: float | None = None,
    sample_fps: float | None = None,
    device: str | None = None,
    include_analysis_window: bool = True,
    include_evaluation_match: bool = True,
) -> dict[str, Any]:
    resolved_analysis_id = analysis_id or default_analysis_id(match_id, start_seconds, stop_seconds)
    analysis_window = (
        build_analysis_window(resolved_analysis_id, start_seconds, stop_seconds, sample_fps, device)
        if include_analysis_window
        else None
    )
    evaluation_match = (
        build_evaluation_match(resolved_analysis_id, video, start_seconds, stop_seconds, sample_fps, device)
        if include_evaluation_match
        else None
    )
    return {
        "match_id": match_id,
        "analysis_id": resolved_analysis_id,
        "video": display_project_path(video),
        "video_probe": probe_video(video),
        "registry_entry": build_registry_match(
            match_id,
            video,
            purpose=purpose,
            notes=notes,
            mode=mode,
            stage=stage,
            analysis_window=analysis_window,
        ),
        "evaluation_entry": evaluation_match,
    }


def apply_intake_plan(
    plan: dict[str, Any],
    paths: IntakePaths = IntakePaths(),
    replace: bool = False,
) -> dict[str, Any]:
    registry = load_json(paths.registry)
    evaluation = load_json(paths.evaluation_config)

    registry_status = upsert_registry_match(registry, plan["registry_entry"], replace=replace)
    evaluation_status = "skipped"
    if plan.get("evaluation_entry"):
        evaluation_status = upsert_entry(
            evaluation.setdefault("analysis_matches", []),
            plan["evaluation_entry"],
            "analysis_matches",
            replace=replace,
        )

    write_json(paths.registry, registry)
    write_json(paths.evaluation_config, evaluation)
    return {
        "registry": display_path(project_path(paths.registry)),
        "evaluation_config": display_path(project_path(paths.evaluation_config)),
        "registry_status": registry_status,
        "evaluation_status": evaluation_status,
    }


def render_intake_report(plan: dict[str, Any], write_result: dict[str, Any] | None = None) -> str:
    probe = plan["video_probe"]
    lines = [
        "# Match Intake Report",
        "",
        f"- match_id: `{plan['match_id']}`",
        f"- analysis_id: `{plan['analysis_id']}`",
        f"- video: `{plan['video']}`",
        f"- exists: {probe['exists']}",
        f"- readable: {probe['readable']}",
    ]
    if probe.get("duration_seconds") is not None:
        lines.append(f"- duration_seconds: {probe['duration_seconds']}")
    if probe.get("fps") is not None:
        lines.append(f"- fps: {probe['fps']}")
    if probe.get("frame_count") is not None:
        lines.append(f"- frame_count: {probe['frame_count']}")
    if probe.get("width") and probe.get("height"):
        lines.append(f"- resolution: {probe['width']}x{probe['height']}")
    if probe.get("error"):
        lines.append(f"- probe_error: {probe['error']}")

    if write_result:
        lines.extend(
            [
                "",
                "## Write Result",
                "",
                f"- registry: `{write_result['registry']}` ({write_result['registry_status']})",
                f"- evaluation_config: `{write_result['evaluation_config']}` ({write_result['evaluation_status']})",
            ]
        )

    lines.extend(
        [
            "",
            "## Registry Entry",
            "",
            "```json",
            json.dumps(plan["registry_entry"], indent=2, ensure_ascii=False),
            "```",
        ]
    )
    if plan.get("evaluation_entry"):
        lines.extend(
            [
                "",
                "## Evaluation Entry",
                "",
                "```json",
                json.dumps(plan["evaluation_entry"], indent=2, ensure_ascii=False),
                "```",
                "",
                "## Next Commands",
                "",
                "```bash",
                "python scripts/validate_data_registry.py --strict",
                f"python scripts/evaluate_matches.py --only {plan['analysis_id']} --run-analysis --strict",
                "```",
            ]
        )
    return "\n".join(lines) + "\n"
