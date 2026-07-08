from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.replay_timeline import (
    TIMELINE_FIELDS,
    EventSource,
    build_replay_timeline,
    read_csv_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align event CSVs from multiple sources into one replay timeline.")
    parser.add_argument("--events-csv", action="append", required=True, help="Event CSV path. May be repeated.")
    parser.add_argument("--source-id", action="append", help="Source id for each --events-csv. Defaults to source_1...")
    parser.add_argument("--time-offset", action="append", type=float, help="Seconds added to each source local time.")
    parser.add_argument("--clips-csv", help="Optional replay_clips.csv to attach clip ids/paths.")
    parser.add_argument("--merge-window-seconds", type=float, default=1.0)
    parser.add_argument("--output-csv", default="outputs/replay_timeline/replay_timeline.csv")
    parser.add_argument("--output-json", default="outputs/replay_timeline/replay_timeline.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_ids = args.source_id or []
    offsets = args.time_offset or []
    sources: list[EventSource] = []
    for index, path_value in enumerate(args.events_csv, start=1):
        source_id = source_ids[index - 1] if index - 1 < len(source_ids) else f"source_{index}"
        offset = offsets[index - 1] if index - 1 < len(offsets) else 0.0
        path = project_path(path_value)
        sources.append(EventSource(source_id=source_id, rows=read_csv_rows(Path(path)), time_offset=offset))

    clips = read_csv_rows(project_path(args.clips_csv)) if args.clips_csv else []
    report = build_replay_timeline(sources, clips, merge_window_seconds=args.merge_window_seconds)
    output_csv = project_path(args.output_csv)
    output_json = project_path(args.output_json)
    write_csv(Path(output_csv), TIMELINE_FIELDS, report["timeline"])
    write_json(Path(output_json), report)

    print(f"sources: {report['source_count']}")
    print(f"raw events: {report['raw_event_count']}")
    print(f"timeline events: {report['timeline_event_count']} merged={report['merged_event_count']}")
    print(f"timeline csv: {output_csv}")
    print(f"timeline json: {output_json}")
    return 0 if report["timeline_event_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
