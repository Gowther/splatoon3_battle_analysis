from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.death_event_assets import (
    DEATH_ASSET_FIELDS,
    DEFAULT_FRAME_OFFSETS,
    enrich_events_with_assets,
    export_death_event_assets,
    read_csv_rows,
    write_csv,
    write_json,
)
from src.death_events import DEATH_EVENT_FIELDS


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export review frames and optional short clips for death events.")
    parser.add_argument("--events-csv", required=True, help="Death event CSV from scripts/extract_death_events.py.")
    parser.add_argument("--video", required=True, help="Source gameplay video or image.")
    parser.add_argument("--output-dir", default="outputs/death_events/assets", help="Directory for per-event assets.")
    parser.add_argument("--manifest-csv", help="Asset manifest CSV path.")
    parser.add_argument("--manifest-json", help="Asset report JSON path.")
    parser.add_argument("--updated-events-csv", help="Optional event CSV with clip_path/asset notes filled.")
    parser.add_argument(
        "--frame-offsets",
        default=",".join(str(value) for value in DEFAULT_FRAME_OFFSETS),
        help="Comma-separated offsets from event time for review frames.",
    )
    parser.add_argument("--clip-before", type=float, default=8.0, help="Fallback seconds before event when clip_start is empty.")
    parser.add_argument("--clip-after", type=float, default=4.0, help="Fallback seconds after event when clip_end is empty.")
    parser.add_argument("--write-clips", action="store_true", help="Also write per-event MP4 clips.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_csv = project_path(args.events_csv)
    video_path = project_path(args.video)
    output_dir = project_path(args.output_dir)
    manifest_csv = project_path(args.manifest_csv) if args.manifest_csv else output_dir / "death_event_assets.csv"
    manifest_json = project_path(args.manifest_json) if args.manifest_json else output_dir / "death_event_assets.json"
    updated_events_csv = project_path(args.updated_events_csv) if args.updated_events_csv else None
    frame_offsets = parse_float_list(args.frame_offsets)

    events = read_csv_rows(Path(events_csv))
    report = export_death_event_assets(
        events,
        video_path=Path(video_path),
        output_dir=Path(output_dir),
        frame_offsets=frame_offsets,
        default_before=args.clip_before,
        default_after=args.clip_after,
        write_clips=args.write_clips,
    )
    write_csv(Path(manifest_csv), DEATH_ASSET_FIELDS, report["assets"])
    write_json(Path(manifest_json), report)
    if updated_events_csv is not None:
        enriched = enrich_events_with_assets(events, report["assets"])
        write_csv(Path(updated_events_csv), DEATH_EVENT_FIELDS, enriched)

    print(f"events csv: {events_csv}")
    print(f"source video: {video_path}")
    print(f"assets: {report['asset_count']} ready={report['ready_count']} partial={report['partial_count']} failed={report['failed_count']}")
    print(f"asset manifest csv: {manifest_csv}")
    print(f"asset manifest json: {manifest_json}")
    if updated_events_csv is not None:
        print(f"updated events csv: {updated_events_csv}")
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
