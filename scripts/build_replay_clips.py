from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.replay_clips import REPLAY_CLIP_FIELDS, build_replay_clip_plan, read_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replay clip and highlight plans from death events.")
    parser.add_argument("--events-csv", required=True, help="Attributed death events CSV.")
    parser.add_argument("--video", help="Source video. Required only when --write-clips is used.")
    parser.add_argument("--output-dir", default="outputs/replay_clips", help="Clip plan and optional clips output directory.")
    parser.add_argument("--manifest-csv", help="Replay clip manifest CSV path.")
    parser.add_argument("--manifest-json", help="Replay clip plan JSON path.")
    parser.add_argument("--clip-before", type=float, default=6.0)
    parser.add_argument("--clip-after", type=float, default=3.0)
    parser.add_argument("--highlight-gap-seconds", type=float, default=6.0)
    parser.add_argument("--write-clips", action="store_true", help="Write MP4 clips in addition to the plan.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_clips and not args.video:
        raise SystemExit("--video is required when --write-clips is set")
    events_csv = project_path(args.events_csv)
    source_video = str(project_path(args.video)) if args.video else ""
    output_dir = project_path(args.output_dir)
    manifest_csv = project_path(args.manifest_csv) if args.manifest_csv else output_dir / "replay_clips.csv"
    manifest_json = project_path(args.manifest_json) if args.manifest_json else output_dir / "replay_clips.json"

    report = build_replay_clip_plan(
        read_csv_rows(Path(events_csv)),
        output_dir=Path(output_dir),
        source_video=source_video,
        default_before=args.clip_before,
        default_after=args.clip_after,
        highlight_gap_seconds=args.highlight_gap_seconds,
        write_clips=args.write_clips,
    )
    write_csv(Path(manifest_csv), REPLAY_CLIP_FIELDS, report["clips"])
    write_json(Path(manifest_json), report)

    print(f"events csv: {events_csv}")
    print(f"clips: {report['clip_count']} highlights={report['highlight_count']}")
    print(f"manifest csv: {manifest_csv}")
    print(f"manifest json: {manifest_json}")
    return 0 if report["clip_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
