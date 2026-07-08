from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.death_attribution import (
    ATTRIBUTION_FIELDS,
    attribute_death_events,
    read_csv_rows,
    write_csv,
    write_json,
)
from src.weapons import load_weapon_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute death events using OCR text and weapon snapshots.")
    parser.add_argument("--events-csv", required=True, help="Death events CSV.")
    parser.add_argument("--ocr-candidates-csv", help="Death OCR candidate CSV with ocr_text/corrected_text fields.")
    parser.add_argument("--analysis-csv", help="Analysis CSV used for nearest weapon snapshot.")
    parser.add_argument("--weapon-list", default="main_weapon_list.txt", help="Weapon names used to match OCR text.")
    parser.add_argument("--output-csv", default="outputs/death_events/attributed_death_events.csv")
    parser.add_argument("--report-json", default="outputs/death_events/death_attribution_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_csv = project_path(args.events_csv)
    ocr_csv = project_path(args.ocr_candidates_csv) if args.ocr_candidates_csv else None
    analysis_csv = project_path(args.analysis_csv) if args.analysis_csv else None
    weapon_list = project_path(args.weapon_list)
    output_csv = project_path(args.output_csv)
    report_json = project_path(args.report_json)

    events = read_csv_rows(Path(events_csv))
    ocr_rows = read_csv_rows(Path(ocr_csv)) if ocr_csv else []
    analysis_rows = read_csv_rows(Path(analysis_csv)) if analysis_csv else []
    weapon_names = load_weapon_names(Path(weapon_list))
    report = attribute_death_events(events, ocr_rows, analysis_rows, weapon_names)
    write_csv(Path(output_csv), ATTRIBUTION_FIELDS, report["events"])
    write_json(Path(report_json), report)

    print(f"events csv: {events_csv}")
    if ocr_csv:
        print(f"ocr candidates csv: {ocr_csv}")
    if analysis_csv:
        print(f"analysis csv: {analysis_csv}")
    print(f"events: {report['event_count']} attributed={report['attributed_count']}")
    print(f"by status: {report['by_status']}")
    print(f"output csv: {output_csv}")
    print(f"report json: {report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
