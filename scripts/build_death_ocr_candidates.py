from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.death_ocr_candidates import (
    DEATH_OCR_CANDIDATE_FIELDS,
    build_death_ocr_candidates,
    read_csv_rows,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build death-screen and kill-log OCR review candidates from death event assets.")
    parser.add_argument("--asset-manifest", required=True, help="CSV from scripts/export_death_event_windows.py.")
    parser.add_argument("--output-dir", default="outputs/death_events/ocr_candidates", help="Output directory for crops and reports.")
    parser.add_argument("--candidates-csv", help="Candidate CSV path.")
    parser.add_argument("--report-json", help="Candidate report JSON path.")
    parser.add_argument("--max-frames-per-event", type=int, default=2, help="How many exported frames to crop per death event.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_manifest = project_path(args.asset_manifest)
    output_dir = project_path(args.output_dir)
    candidates_csv = project_path(args.candidates_csv) if args.candidates_csv else output_dir / "death_ocr_candidates.csv"
    report_json = project_path(args.report_json) if args.report_json else output_dir / "death_ocr_candidates.json"

    assets = read_csv_rows(Path(asset_manifest))
    report = build_death_ocr_candidates(
        assets,
        output_dir=Path(output_dir),
        max_frames_per_event=args.max_frames_per_event,
    )
    write_csv(Path(candidates_csv), DEATH_OCR_CANDIDATE_FIELDS, report["candidates"])
    write_json(Path(report_json), report)

    print(f"asset manifest: {asset_manifest}")
    print(f"ocr candidates: {report['candidate_count']}")
    print(f"crop failures: {report['failure_count']}")
    print(f"candidates csv: {candidates_csv}")
    print(f"report json: {report_json}")
    return 0 if report["candidate_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
