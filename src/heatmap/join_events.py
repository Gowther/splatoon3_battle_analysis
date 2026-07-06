from __future__ import annotations

import argparse
import bisect
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.heatmap.extract_frames import load_config, resolve_path


Row = Dict[str, str]
Event = Dict[str, object]
Point = Dict[str, object]


EVENT_TEMPLATE_FIELDS = ["time", "event", "team", "player", "killer", "victim", "clip_path", "segment_id", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join external kill/death events to heatmap points.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--events", help="Override event CSV path.")
    parser.add_argument("--window-seconds", type=float, help="Override event-to-point time window.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Row]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_event_template(path: Path) -> None:
    if path.exists():
        return
    write_csv(path, EVENT_TEMPLATE_FIELDS, [])


def parse_time(row: Dict[str, object], field: str) -> Optional[float]:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def parse_points(rows: Sequence[Row]) -> List[Point]:
    points: List[Point] = []
    for row in rows:
        time_value = parse_time(row, "time")
        if time_value is None:
            continue
        point: Point = dict(row)
        point["_time"] = time_value
        points.append(point)
    points.sort(key=lambda item: float(item["_time"]))
    return points


def parse_events(rows: Sequence[Row]) -> List[Event]:
    events: List[Event] = []
    for index, row in enumerate(rows, start=1):
        time_value = parse_time(row, "time")
        if time_value is None:
            continue
        event: Event = dict(row)
        event["_time"] = time_value
        event["_event_id"] = str(index)
        events.append(event)
    events.sort(key=lambda item: float(item["_time"]))
    return events


def point_matches_event_team(point: Point, event: Event) -> bool:
    event_team = str(event.get("team", "")).strip()
    if not event_team:
        return True
    return str(point.get("team", "")).strip() == event_team


def events_near_time(
    time_value: float,
    events: Sequence[Event],
    event_times: Sequence[float],
    window_seconds: float,
) -> List[Event]:
    start = bisect.bisect_left(event_times, time_value - window_seconds)
    stop = bisect.bisect_right(event_times, time_value + window_seconds)
    return list(events[start:stop])


def nearest_point_for_event(event: Event, points: Sequence[Point], point_times: Sequence[float], window_seconds: float) -> Dict[str, object]:
    event_time = float(event["_time"])
    start = bisect.bisect_left(point_times, event_time - window_seconds)
    stop = bisect.bisect_right(point_times, event_time + window_seconds)
    candidates = [point for point in points[start:stop] if point_matches_event_team(point, event)]
    output: Dict[str, object] = {
        "event_id": event["_event_id"],
        "event_time": f"{event_time:.3f}",
        "event": event.get("event", ""),
        "team": event.get("team", ""),
        "player": event.get("player", ""),
        "killer": event.get("killer", ""),
        "victim": event.get("victim", ""),
        "clip_path": event.get("clip_path", ""),
        "near_point_count": len(candidates),
        "nearest_point_time": "",
        "nearest_point_delta": "",
        "nearest_point_team": "",
        "nearest_point_x": "",
        "nearest_point_y": "",
    }
    if not candidates:
        return output
    nearest = min(candidates, key=lambda point: abs(float(point["_time"]) - event_time))
    output.update(
        {
            "nearest_point_time": nearest.get("time", ""),
            "nearest_point_delta": round(abs(float(nearest["_time"]) - event_time), 3),
            "nearest_point_team": nearest.get("team", ""),
            "nearest_point_x": nearest.get("x", ""),
            "nearest_point_y": nearest.get("y", ""),
        }
    )
    return output


def join_point_events(points: Sequence[Point], events: Sequence[Event], window_seconds: float) -> List[Dict[str, object]]:
    event_times = [float(event["_time"]) for event in events]
    output_rows: List[Dict[str, object]] = []
    for point in points:
        nearby = [
            event
            for event in events_near_time(float(point["_time"]), events, event_times, window_seconds)
            if point_matches_event_team(point, event)
        ]
        row = {key: value for key, value in point.items() if not key.startswith("_")}
        row["event_count_nearby"] = len(nearby)
        row["event_types_nearby"] = ";".join(sorted({str(event.get("event", "")) for event in nearby if event.get("event")}))
        if nearby:
            nearest = min(nearby, key=lambda event: abs(float(event["_time"]) - float(point["_time"])))
            row["nearest_event_time"] = f"{float(nearest['_time']):.3f}"
            row["nearest_event_delta"] = round(abs(float(nearest["_time"]) - float(point["_time"])), 3)
            row["nearest_event_type"] = nearest.get("event", "")
            row["nearest_event_team"] = nearest.get("team", "")
            row["nearest_event_player"] = nearest.get("player", "")
            row["nearest_event_killer"] = nearest.get("killer", "")
            row["nearest_event_victim"] = nearest.get("victim", "")
            row["nearest_event_clip_path"] = nearest.get("clip_path", "")
        else:
            row["nearest_event_time"] = ""
            row["nearest_event_delta"] = ""
            row["nearest_event_type"] = ""
            row["nearest_event_team"] = ""
            row["nearest_event_player"] = ""
            row["nearest_event_killer"] = ""
            row["nearest_event_victim"] = ""
            row["nearest_event_clip_path"] = ""
        output_rows.append(row)
    return output_rows


def event_segments(events: Sequence[Event], gap_seconds: float) -> List[Dict[str, object]]:
    if not events:
        return []
    segments: List[List[Event]] = []
    current: List[Event] = []
    for event in events:
        if not current or float(event["_time"]) - float(current[-1]["_time"]) <= gap_seconds:
            current.append(event)
        else:
            segments.append(current)
            current = [event]
    if current:
        segments.append(current)

    rows: List[Dict[str, object]] = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment[0]["_time"])
        stop = float(segment[-1]["_time"])
        rows.append(
            {
                "segment_id": index,
                "start_time": f"{start:.3f}",
                "stop_time": f"{stop:.3f}",
                "duration": round(stop - start, 3),
                "event_count": len(segment),
                "event_types": ";".join(sorted({str(event.get("event", "")) for event in segment if event.get("event")})),
                "clip_paths": ";".join(str(event.get("clip_path", "")) for event in segment if event.get("clip_path")),
            }
        )
    return rows


def output_fieldnames(base_rows: Sequence[Row]) -> List[str]:
    base = list(base_rows[0].keys()) if base_rows else []
    additions = [
        "event_count_nearby",
        "event_types_nearby",
        "nearest_event_time",
        "nearest_event_delta",
        "nearest_event_type",
        "nearest_event_team",
        "nearest_event_player",
        "nearest_event_killer",
        "nearest_event_victim",
        "nearest_event_clip_path",
    ]
    return base + [field for field in additions if field not in base]


def report_rows(points: Sequence[Point], events: Sequence[Event], point_event_rows: Sequence[Dict[str, object]], event_near_rows: Sequence[Dict[str, object]], segments: Sequence[Dict[str, object]], window_seconds: float) -> List[Dict[str, object]]:
    event_counts = Counter(str(event.get("event", "")) for event in events if event.get("event"))
    points_with_events = sum(1 for row in point_event_rows if int(row["event_count_nearby"]) > 0)
    matched_events = sum(1 for row in event_near_rows if int(row["near_point_count"]) > 0)
    rows: List[Dict[str, object]] = [
        {"metric": "point_rows", "value": len(points)},
        {"metric": "event_rows", "value": len(events)},
        {"metric": "points_with_nearby_events", "value": points_with_events},
        {"metric": "events_with_nearby_points", "value": matched_events},
        {"metric": "segments", "value": len(segments)},
        {"metric": "time_window_seconds", "value": window_seconds},
    ]
    for event_type, count in sorted(event_counts.items()):
        rows.append({"metric": f"event_type_{event_type}", "value": count})
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    event_path = resolve_path(args.events) if args.events else resolve_path(config["event_join"]["event_csv"])
    ensure_event_template(resolve_path(config["outputs"]["event_template_csv"]))
    window_seconds = float(args.window_seconds if args.window_seconds is not None else config["event_join"]["time_window_seconds"])
    gap_seconds = float(config["event_join"]["segment_gap_seconds"])

    point_rows = read_csv(resolve_path(config["outputs"]["enriched_points_csv"]))
    event_rows = read_csv(event_path)
    points = parse_points(point_rows)
    events = parse_events(event_rows)
    point_times = [float(point["_time"]) for point in points]
    point_event_rows = join_point_events(points, events, window_seconds)
    event_near_rows = [nearest_point_for_event(event, points, point_times, window_seconds) for event in events]
    segments = event_segments(events, gap_seconds)

    write_csv(resolve_path(config["outputs"]["points_with_events_csv"]), output_fieldnames(point_rows), point_event_rows)
    write_csv(
        resolve_path(config["outputs"]["events_near_points_csv"]),
        [
            "event_id",
            "event_time",
            "event",
            "team",
            "player",
            "killer",
            "victim",
            "clip_path",
            "near_point_count",
            "nearest_point_time",
            "nearest_point_delta",
            "nearest_point_team",
            "nearest_point_x",
            "nearest_point_y",
        ],
        event_near_rows,
    )
    write_csv(
        resolve_path(config["outputs"]["event_segments_csv"]),
        ["segment_id", "start_time", "stop_time", "duration", "event_count", "event_types", "clip_paths"],
        segments,
    )
    write_csv(
        resolve_path(config["outputs"]["event_join_report_csv"]),
        ["metric", "value"],
        report_rows(points, events, point_event_rows, event_near_rows, segments, window_seconds),
    )

    print(f"event csv: {event_path}")
    print(f"points: {len(points)}")
    print(f"events: {len(events)}")
    print(f"points with events csv: {resolve_path(config['outputs']['points_with_events_csv'])}")
    print(f"events near points csv: {resolve_path(config['outputs']['events_near_points_csv'])}")
    print(f"event join report: {resolve_path(config['outputs']['event_join_report_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
