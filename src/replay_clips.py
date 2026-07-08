from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data_registry import display_path
from src.death_event_assets import write_clip


REPLAY_CLIP_FIELDS = [
    "clip_id",
    "clip_type",
    "match_id",
    "event_ids",
    "start_time",
    "end_time",
    "duration",
    "killer",
    "victims",
    "cause_weapons",
    "score",
    "source_video",
    "clip_path",
    "status",
    "notes",
]


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


def safe_name(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)[:120] or "clip"


def event_time(row: Mapping[str, Any]) -> float | None:
    return float_or_none(row.get("time")) or float_or_none(row.get("elapsed_time"))


def event_window(row: Mapping[str, Any], default_before: float, default_after: float) -> tuple[float, float] | None:
    time_value = event_time(row)
    if time_value is None:
        return None
    start = float_or_none(row.get("clip_start"))
    stop = float_or_none(row.get("clip_end"))
    if start is None:
        start = max(0.0, time_value - default_before)
    if stop is None:
        stop = time_value + default_after
    return max(0.0, start), max(start, stop)


def attribution_confidence(row: Mapping[str, Any]) -> float:
    return float_or_none(row.get("attribution_confidence")) or float_or_none(row.get("confidence")) or 0.0


def event_score(row: Mapping[str, Any]) -> float:
    score = 50.0
    if row.get("attribution_status") == "attributed":
        score += 20.0
    if row.get("killer"):
        score += 10.0
    if row.get("cause_weapon"):
        score += 8.0
    score += min(10.0, attribution_confidence(row) * 10.0)
    return round(score, 3)


def build_event_clip(
    row: Mapping[str, Any],
    index: int,
    source_video: str,
    output_dir: Path,
    default_before: float,
    default_after: float,
    write_clips: bool,
) -> dict[str, Any] | None:
    window = event_window(row, default_before, default_after)
    if window is None:
        return None
    start, stop = window
    event_id = str(row.get("event_id") or f"event_{index:04d}")
    clip_id = f"death_{index:04d}_{safe_name(event_id)}"
    clip_path = ""
    status = "planned"
    if write_clips and source_video:
        target = output_dir / "clips" / f"{clip_id}.mp4"
        if write_clip(Path(source_video), start, stop, target):
            clip_path = display_path(target)
            status = "ready"
        else:
            status = "clip_failed"
    return {
        "clip_id": clip_id,
        "clip_type": "death_event",
        "match_id": row.get("match_id", ""),
        "event_ids": event_id,
        "start_time": format_seconds(start),
        "end_time": format_seconds(stop),
        "duration": format_seconds(stop - start),
        "killer": row.get("killer", ""),
        "victims": row.get("victim", ""),
        "cause_weapons": row.get("cause_weapon", ""),
        "score": event_score(row),
        "source_video": display_path(Path(source_video)) if source_video else "",
        "clip_path": clip_path or row.get("clip_path", ""),
        "status": status,
        "notes": row.get("attribution_evidence", "") or row.get("notes", ""),
    }


def cluster_highlight_clips(
    events: Sequence[Mapping[str, Any]],
    source_video: str,
    output_dir: Path,
    gap_seconds: float,
    write_clips: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in events:
        killer = str(row.get("killer", "")).strip()
        if not killer:
            continue
        grouped.setdefault(killer, []).append(row)

    clips: list[dict[str, Any]] = []
    for killer, rows in grouped.items():
        rows = sorted(rows, key=lambda row: event_time(row) or 0.0)
        current: list[Mapping[str, Any]] = []
        for row in rows:
            if not current:
                current = [row]
                continue
            previous_time = event_time(current[-1])
            current_time = event_time(row)
            if previous_time is not None and current_time is not None and current_time - previous_time <= gap_seconds:
                current.append(row)
            else:
                if len(current) >= 2:
                    clips.append(build_cluster_clip(killer, current, source_video, output_dir, len(clips) + 1, write_clips))
                current = [row]
        if len(current) >= 2:
            clips.append(build_cluster_clip(killer, current, source_video, output_dir, len(clips) + 1, write_clips))
    return clips


def build_cluster_clip(
    killer: str,
    rows: Sequence[Mapping[str, Any]],
    source_video: str,
    output_dir: Path,
    index: int,
    write_clips: bool,
) -> dict[str, Any]:
    windows = [event_window(row, 6.0, 3.0) for row in rows]
    windows = [window for window in windows if window is not None]
    start = min(window[0] for window in windows)
    stop = max(window[1] for window in windows)
    event_ids = [str(row.get("event_id", "")) for row in rows if row.get("event_id")]
    clip_id = f"highlight_{index:04d}_{safe_name(killer)}"
    clip_path = ""
    status = "planned"
    if write_clips and source_video:
        target = output_dir / "clips" / f"{clip_id}.mp4"
        if write_clip(Path(source_video), start, stop, target):
            clip_path = display_path(target)
            status = "ready"
        else:
            status = "clip_failed"
    return {
        "clip_id": clip_id,
        "clip_type": "multi_kill_candidate",
        "match_id": rows[0].get("match_id", ""),
        "event_ids": ";".join(event_ids),
        "start_time": format_seconds(start),
        "end_time": format_seconds(stop),
        "duration": format_seconds(stop - start),
        "killer": killer,
        "victims": ";".join(str(row.get("victim", "")) for row in rows if row.get("victim")),
        "cause_weapons": ";".join(sorted({str(row.get("cause_weapon", "")) for row in rows if row.get("cause_weapon")})),
        "score": round(90.0 + len(rows) * 10.0, 3),
        "source_video": display_path(Path(source_video)) if source_video else "",
        "clip_path": clip_path,
        "status": status,
        "notes": f"{len(rows)} deaths by same killer within {format_seconds(stop - start)}s",
    }


def build_replay_clip_plan(
    events: Sequence[Mapping[str, Any]],
    output_dir: Path,
    source_video: str = "",
    default_before: float = 6.0,
    default_after: float = 3.0,
    highlight_gap_seconds: float = 6.0,
    write_clips: bool = False,
) -> dict[str, Any]:
    event_clips = [
        clip
        for index, row in enumerate(events, start=1)
        if (clip := build_event_clip(row, index, source_video, output_dir, default_before, default_after, write_clips)) is not None
    ]
    highlight_clips = cluster_highlight_clips(events, source_video, output_dir, highlight_gap_seconds, write_clips)
    clips = sorted(event_clips + highlight_clips, key=lambda row: (-float(row.get("score", 0)), row.get("start_time", "")))
    return {
        "schema_version": 1,
        "status": "ready" if clips else "empty",
        "event_count": len(events),
        "clip_count": len(clips),
        "highlight_count": len(highlight_clips),
        "write_clips": write_clips,
        "output_dir": display_path(output_dir),
        "source_video": display_path(Path(source_video)) if source_video else "",
        "clips": clips,
    }
