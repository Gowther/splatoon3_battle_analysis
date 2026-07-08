from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TIMELINE_FIELDS = [
    "timeline_id",
    "unified_time",
    "event_type",
    "event_ids",
    "sources",
    "source_count",
    "local_times",
    "match_id",
    "killer",
    "victims",
    "cause_weapons",
    "clip_ids",
    "clip_paths",
    "confidence",
    "notes",
]


@dataclass(frozen=True)
class EventSource:
    source_id: str
    rows: Sequence[Mapping[str, Any]]
    time_offset: float = 0.0


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def event_local_time(row: Mapping[str, Any]) -> float | None:
    return float_or_none(row.get("time")) or float_or_none(row.get("event_time")) or float_or_none(row.get("elapsed_time"))


def confidence(row: Mapping[str, Any]) -> float:
    return (
        float_or_none(row.get("attribution_confidence"))
        or float_or_none(row.get("confidence"))
        or float_or_none(row.get("score"))
        or 0.0
    )


def semantic_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("event") or row.get("event_type") or row.get("clip_type") or "event"),
        str(row.get("killer", "")),
        str(row.get("victim", "") or row.get("victims", "")),
        str(row.get("cause_weapon", "") or row.get("cause_weapons", "")),
    )


def normalize_source_events(sources: Sequence[EventSource]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in sources:
        for row in source.rows:
            local_time = event_local_time(row)
            if local_time is None:
                continue
            unified_time = local_time + source.time_offset
            event_id = str(row.get("event_id") or row.get("clip_id") or "")
            output.append(
                {
                    "source_id": source.source_id,
                    "time_offset": source.time_offset,
                    "local_time": local_time,
                    "unified_time": unified_time,
                    "event_id": event_id,
                    "event_type": str(row.get("event") or row.get("event_type") or row.get("clip_type") or "event"),
                    "match_id": row.get("match_id", ""),
                    "killer": row.get("killer", ""),
                    "victim": row.get("victim", "") or row.get("victims", ""),
                    "cause_weapon": row.get("cause_weapon", "") or row.get("cause_weapons", ""),
                    "clip_id": row.get("clip_id", ""),
                    "clip_path": row.get("clip_path", ""),
                    "confidence": confidence(row),
                    "notes": row.get("notes", "") or row.get("attribution_evidence", ""),
                    "_raw": dict(row),
                }
            )
    return sorted(output, key=lambda row: (float(row["unified_time"]), row["source_id"], row["event_id"]))


def should_merge_event(group: Sequence[Mapping[str, Any]], event: Mapping[str, Any], merge_window_seconds: float) -> bool:
    if not group:
        return False
    event_id = str(event.get("event_id", ""))
    if event_id and any(str(item.get("event_id", "")) == event_id for item in group):
        return True
    representative = group[0]
    if semantic_key(representative) != semantic_key(event):
        return False
    group_time = sum(float(item["unified_time"]) for item in group) / len(group)
    return abs(float(event["unified_time"]) - group_time) <= merge_window_seconds


def group_aligned_events(events: Sequence[Mapping[str, Any]], merge_window_seconds: float) -> list[list[Mapping[str, Any]]]:
    groups: list[list[Mapping[str, Any]]] = []
    for event in events:
        matched = False
        for group in groups:
            if should_merge_event(group, event, merge_window_seconds):
                group.append(event)
                matched = True
                break
        if not matched:
            groups.append([event])
    return groups


def unique_join(values: Iterable[Any]) -> str:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def timeline_row(group: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    unified = sum(float(item["unified_time"]) for item in group) / len(group)
    event_type = unique_join(item.get("event_type", "") for item in group) or "event"
    return {
        "timeline_id": f"timeline_{index:04d}",
        "unified_time": format_seconds(unified),
        "event_type": event_type,
        "event_ids": unique_join(item.get("event_id", "") for item in group),
        "sources": unique_join(item.get("source_id", "") for item in group),
        "source_count": len({str(item.get("source_id", "")) for item in group if item.get("source_id")}),
        "local_times": ";".join(f"{item.get('source_id', '')}:{format_seconds(float(item['local_time']))}" for item in group),
        "match_id": unique_join(item.get("match_id", "") for item in group),
        "killer": unique_join(item.get("killer", "") for item in group),
        "victims": unique_join(item.get("victim", "") for item in group),
        "cause_weapons": unique_join(item.get("cause_weapon", "") for item in group),
        "clip_ids": unique_join(item.get("clip_id", "") for item in group),
        "clip_paths": unique_join(item.get("clip_path", "") for item in group),
        "confidence": format_seconds(max(float(item.get("confidence", 0.0)) for item in group)),
        "notes": unique_join(item.get("notes", "") for item in group),
    }


def clips_by_event_id(clip_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {}
    for clip in clip_rows:
        for event_id in str(clip.get("event_ids", "")).split(";"):
            event_id = event_id.strip()
            if event_id:
                output.setdefault(event_id, []).append(clip)
    return output


def attach_clip_rows(events: Sequence[dict[str, Any]], clip_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_event = clips_by_event_id(clip_rows)
    output: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        clips = by_event.get(str(row.get("event_id", "")), [])
        if clips:
            row["clip_id"] = unique_join(clip.get("clip_id", "") for clip in clips)
            row["clip_path"] = unique_join(clip.get("clip_path", "") for clip in clips)
        output.append(row)
    return output


def build_replay_timeline(
    sources: Sequence[EventSource],
    clip_rows: Sequence[Mapping[str, Any]] | None = None,
    merge_window_seconds: float = 1.0,
) -> dict[str, Any]:
    normalized = normalize_source_events(sources)
    normalized = attach_clip_rows(normalized, list(clip_rows or []))
    groups = group_aligned_events(normalized, merge_window_seconds)
    rows = [timeline_row(group, index) for index, group in enumerate(groups, start=1)]
    return {
        "schema_version": 1,
        "status": "ready" if rows else "empty",
        "source_count": len(sources),
        "raw_event_count": len(normalized),
        "timeline_event_count": len(rows),
        "merged_event_count": sum(1 for row in rows if int(row["source_count"]) > 1),
        "merge_window_seconds": merge_window_seconds,
        "timeline": rows,
    }
