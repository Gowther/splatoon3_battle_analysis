from __future__ import annotations

import csv
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.active_learning_workbench import display_path, safe_project_file, utc_now
from src.core.paths import ROOT, project_path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TIME_FIELDS = ("elapsed_time", "time", "event_time", "nearest_point_time", "clip_start")
DEFAULT_REVIEW_LOG_PATH = ROOT / "outputs" / "data_review_workbench" / "reviews.jsonl"
DEFAULT_SOURCE_ROOTS = (ROOT / "outputs",)
DEFAULT_VIDEO_ROOTS = (ROOT / "footages", ROOT / "sample")
MAX_SNAPSHOT_ROWS = 80
MATCH_TOKEN_RE = re.compile(r"(?:^|[_/.-])((?:n|f)_match_\d+|match_?\d+)(?=$|[_/.-])", re.IGNORECASE)


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def display_project_path(path: str | Path) -> str:
    return display_path(path)


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def match_tokens_for_text(value: str) -> list[str]:
    tokens: list[str] = []
    normalized = value.replace("\\", "/").lower()
    for match in MATCH_TOKEN_RE.finditer(normalized):
        token = match.group(1)
        if token.startswith("match") and "_" not in token:
            token = f"match_{token.removeprefix('match')}"
        if token not in tokens:
            tokens.append(token)
    return tokens


def first_time_field(fieldnames: list[str]) -> str:
    for field in TIME_FIELDS:
        if field in fieldnames:
            return field
    return ""


def classify_source(path: Path, fieldnames: list[str]) -> str:
    fields = set(fieldnames)
    if "elapsed_time" in fields and any(field.startswith("player_state_") for field in fieldnames):
        return "analysis_csv"
    if "event_id" in fields and ({"victim", "killer"} & fields or "cause_weapon" in fields):
        return "death_events"
    if {"x", "y", "team"} <= fields and ("track_slot" in fields or "player_id" in fields):
        return "heatmap_tracks"
    if {"event_id", "nearest_point_time", "nearest_point_x", "nearest_point_y"} <= fields:
        return "heatmap_event_join"
    if "candidate_id" in fields and first_time_field(fieldnames):
        return "review_candidates"
    if first_time_field(fieldnames):
        return "time_csv"
    return "csv"


def source_label(kind: str, path: Path) -> str:
    names = {
        "analysis_csv": "分析 CSV",
        "death_events": "死亡事件",
        "heatmap_tracks": "热力图轨迹",
        "heatmap_event_join": "热力图事件关联",
        "review_candidates": "复查候选",
        "time_csv": "时间序列 CSV",
        "csv": "CSV",
    }
    return f"{names.get(kind, kind)} · {display_project_path(path)}"


def csv_cache_key(path: str | Path) -> tuple[str, int, int]:
    resolved = project_path(path).resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=128)
def _read_csv_cached(path_text: str, mtime_ns: int, size: int) -> tuple[tuple[str, ...], tuple[tuple[tuple[str, str], ...], ...]]:
    del mtime_ns, size
    with Path(path_text).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = tuple(tuple((key, value) for key, value in row.items()) for row in reader)
    return fieldnames, rows


def read_csv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    resolved = project_path(path).resolve()
    if not resolved.exists() or resolved.is_dir():
        return [], []
    fieldnames, rows = _read_csv_cached(*csv_cache_key(resolved))
    return list(fieldnames), [dict(row) for row in rows]


def describe_video(path: Path) -> dict[str, Any]:
    resolved = project_path(path).resolve()
    return {
        "path": display_project_path(resolved),
        "label": resolved.stem,
        "match_id": resolved.stem,
        "match_tokens": match_tokens_for_text(resolved.stem),
        "size_bytes": resolved.stat().st_size,
    }


def discover_video_files(roots: tuple[Path, ...] = DEFAULT_VIDEO_ROOTS, limit: int = 200) -> list[dict[str, Any]]:
    videos: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and is_video_path(path):
                videos.append(describe_video(path))
                if len(videos) >= limit:
                    return videos
    return videos


def describe_data_source(path: Path) -> dict[str, Any] | None:
    fieldnames, rows = read_csv_rows(path)
    if not fieldnames:
        return None
    time_field = first_time_field(fieldnames)
    if not time_field:
        return None
    kind = classify_source(path, fieldnames)
    times = [value for value in (numeric_value(row.get(time_field)) for row in rows) if value is not None]
    match_tokens = match_tokens_for_text(display_project_path(path))
    for row in rows[:25]:
        match_id = str(row.get("match_id", "")).strip()
        if match_id:
            for token in match_tokens_for_text(match_id):
                if token not in match_tokens:
                    match_tokens.append(token)
    return {
        "path": display_project_path(path),
        "label": source_label(kind, path),
        "kind": kind,
        "time_field": time_field,
        "fieldnames": fieldnames,
        "row_count": len(rows),
        "min_time": min(times) if times else None,
        "max_time": max(times) if times else None,
        "match_tokens": match_tokens,
    }


def discover_data_sources(roots: tuple[Path, ...] = DEFAULT_SOURCE_ROOTS, limit: int = 600) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            source = describe_data_source(path)
            if source:
                sources.append(source)
                if len(sources) >= limit:
                    return sort_sources(sources)
    return sort_sources(sources)


def source_priority(source: dict[str, Any]) -> tuple[int, str]:
    kind_order = {
        "analysis_csv": 0,
        "death_events": 1,
        "heatmap_tracks": 2,
        "heatmap_event_join": 3,
        "review_candidates": 4,
        "time_csv": 5,
    }
    path = str(source.get("path", ""))
    smoothed_bonus = -1 if path.endswith("smoothed.csv") else 0
    return (kind_order.get(str(source.get("kind")), 9) + smoothed_bonus, path)


def sort_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(sources, key=source_priority)


def matching_sources(video_path: str, sources: list[dict[str, Any]], limit: int = 8) -> list[str]:
    video_tokens = set(match_tokens_for_text(Path(video_path).stem) or [Path(video_path).stem.lower()])
    matches = [
        source
        for source in sources
        if video_tokens and video_tokens.intersection(set(source.get("match_tokens", [])))
    ]
    return [str(source["path"]) for source in sort_sources(matches)[:limit]]


def build_review_summary(review_path: str | Path = DEFAULT_REVIEW_LOG_PATH) -> dict[str, Any]:
    resolved = project_path(review_path)
    if not resolved.exists():
        return {"path": display_project_path(resolved), "count": 0, "by_decision": {}, "last_reviewed_at": ""}
    decisions: Counter[str] = Counter()
    last_reviewed_at = ""
    count = 0
    with resolved.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            decisions[str(row.get("decision", "unknown"))] += 1
            last_reviewed_at = str(row.get("created_at", last_reviewed_at))
    return {
        "path": display_project_path(resolved),
        "count": count,
        "by_decision": dict(sorted(decisions.items())),
        "last_reviewed_at": last_reviewed_at,
    }


def build_data_review_state() -> dict[str, Any]:
    videos = discover_video_files()
    sources = discover_data_sources()
    for video in videos:
        video["suggested_sources"] = matching_sources(str(video["path"]), sources)
    status = "ready" if videos and sources else "needs_data"
    return {
        "schema_version": 1,
        "status": status,
        "videos": videos,
        "sources": sources,
        "review_summary": build_review_summary(),
    }


def row_time(row: dict[str, Any], time_field: str) -> float | None:
    return numeric_value(row.get(time_field))


def nearest_rows(
    rows: list[dict[str, str]],
    time_field: str,
    time_value: float,
    *,
    window: float,
    include_window: bool,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    timed: list[tuple[float, float, dict[str, str]]] = []
    for row in rows:
        value = row_time(row, time_field)
        if value is None:
            continue
        timed.append((abs(value - time_value), value, row))
    if not timed:
        return [], None, None
    timed.sort(key=lambda item: (item[0], item[1]))
    if include_window:
        selected = [item for item in timed if item[0] <= window]
        if not selected:
            selected = timed[:1]
    else:
        selected = timed[:1]
    selected = selected[:MAX_SNAPSHOT_ROWS]
    return [
        {
            "time": item_time,
            "delta": round(delta, 4),
            "values": row,
        }
        for delta, item_time, row in selected
    ], selected[0][1], round(selected[0][0], 4)


def snapshot_for_source(source_path: str | Path, time_value: float, window: float) -> dict[str, Any]:
    resolved = safe_project_file(source_path)
    source = describe_data_source(resolved)
    if not source:
        return {
            "path": display_project_path(resolved),
            "status": "missing_or_not_time_indexed",
            "rows": [],
        }
    fieldnames, rows = read_csv_rows(resolved)
    include_window = str(source["kind"]) in {"death_events", "heatmap_tracks", "heatmap_event_join", "review_candidates"}
    selected_rows, selected_time, delta = nearest_rows(
        rows,
        str(source["time_field"]),
        time_value,
        window=window,
        include_window=include_window,
    )
    return {
        **source,
        "status": "ready" if selected_rows else "empty",
        "selected_time": selected_time,
        "delta": delta,
        "rows": selected_rows,
        "display_row_count": len(selected_rows),
        "total_row_count": len(rows),
        "fieldnames": fieldnames,
    }


def build_time_snapshot(
    video_path: str,
    time_value: float,
    source_paths: list[str] | None = None,
    *,
    window: float = 0.35,
) -> dict[str, Any]:
    video = safe_project_file(video_path)
    if not is_video_path(video):
        raise ValueError("unsupported video type")
    state_sources = discover_data_sources()
    selected_paths = source_paths or matching_sources(display_project_path(video), state_sources)
    snapshots = [snapshot_for_source(path, time_value, window) for path in selected_paths]
    return {
        "schema_version": 1,
        "status": "ready" if snapshots else "needs_data",
        "video": describe_video(video),
        "time": round(float(time_value), 4),
        "window": window,
        "source_paths": selected_paths,
        "sources": snapshots,
    }


def record_data_review(payload: dict[str, Any], review_path: str | Path = DEFAULT_REVIEW_LOG_PATH) -> dict[str, Any]:
    decision = str(payload.get("decision", "")).strip()
    if decision not in {"accurate", "incorrect", "needs_review", "skipped"}:
        raise ValueError("decision must be accurate, incorrect, needs_review, or skipped")
    video_path = str(payload.get("video_path", "")).strip()
    if not video_path:
        raise ValueError("video_path is required")
    video = safe_project_file(video_path)
    if not is_video_path(video):
        raise ValueError("unsupported video type")
    time_value = numeric_value(payload.get("time"))
    if time_value is None:
        raise ValueError("time is required")
    source_paths = [display_project_path(safe_project_file(path)) for path in payload.get("source_paths", []) if str(path).strip()]
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "video_path": display_project_path(video),
        "time": round(time_value, 4),
        "source_paths": source_paths,
        "decision": decision,
        "incorrect_fields": list(payload.get("incorrect_fields", [])),
        "note": str(payload.get("note", "")).strip(),
        "snapshot": payload.get("snapshot", {}),
    }
    resolved = project_path(review_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "saved", "review": record, "summary": build_review_summary(resolved)}
