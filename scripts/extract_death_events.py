from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.death_events import (
    DEFAULT_ALIVE_STATE_IDS,
    DEFAULT_DEAD_STATE_IDS,
    build_death_event_report,
    extract_death_events,
    read_csv_rows,
    write_event_csv,
    write_event_json,
)


def parse_id_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract player death events from analysis CSV player_state transitions.")
    parser.add_argument("--analysis-csv", required=True, help="Analysis CSV produced by scripts/run_analysis.py.")
    parser.add_argument("--match-id", help="Match id written into event rows. Defaults to the analysis CSV stem.")
    parser.add_argument("--output-dir", default="outputs/death_events", help="Default directory for generated event files.")
    parser.add_argument("--output-csv", help="Death event CSV path.")
    parser.add_argument("--output-json", help="Death event report JSON path.")
    parser.add_argument(
        "--dead-state-ids",
        default=",".join(DEFAULT_DEAD_STATE_IDS),
        help="Comma-separated detector class ids that mean dead. Default: 1.",
    )
    parser.add_argument(
        "--alive-state-ids",
        default=",".join(DEFAULT_ALIVE_STATE_IDS),
        help="Comma-separated detector class ids that mean non-dead player lamp states. Default: 0,14.",
    )
    parser.add_argument("--clip-before", type=float, default=8.0, help="Seconds before death to reserve for later clips.")
    parser.add_argument("--clip-after", type=float, default=4.0, help="Seconds after death to reserve for later clips.")
    parser.add_argument("--min-dead-frames", type=int, default=1, help="Minimum consecutive dead rows required.")
    parser.add_argument(
        "--include-initial-dead",
        action="store_true",
        help="Emit an event when the first observed state for a slot is already dead.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_csv = project_path(args.analysis_csv)
    match_id = args.match_id or analysis_csv.stem
    output_dir = project_path(args.output_dir)
    output_csv = project_path(args.output_csv) if args.output_csv else output_dir / f"{match_id}_death_events.csv"
    output_json = project_path(args.output_json) if args.output_json else output_dir / f"{match_id}_death_events.json"
    dead_state_ids = parse_id_list(args.dead_state_ids)
    alive_state_ids = parse_id_list(args.alive_state_ids)

    rows = read_csv_rows(Path(analysis_csv))
    events = extract_death_events(
        rows,
        match_id=match_id,
        dead_state_ids=dead_state_ids,
        alive_state_ids=alive_state_ids,
        clip_before=args.clip_before,
        clip_after=args.clip_after,
        min_dead_frames=args.min_dead_frames,
        include_initial_dead=args.include_initial_dead,
    )
    report = build_death_event_report(
        rows,
        match_id=match_id,
        events=events,
        dead_state_ids=dead_state_ids,
        alive_state_ids=alive_state_ids,
    )

    write_event_csv(Path(output_csv), events)
    write_event_json(Path(output_json), report)

    print(f"analysis csv: {analysis_csv}")
    print(f"match id: {match_id}")
    print(f"death events: {len(events)}")
    print(f"event csv: {output_csv}")
    print(f"event report: {output_json}")
    if not events:
        print(f"blocking reason: {report['blocking_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
