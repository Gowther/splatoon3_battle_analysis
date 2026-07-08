from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.death_annotation_store import build_death_annotation_report, read_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report death-event OCR annotation coverage.")
    parser.add_argument(
        "--labels-csv",
        default="outputs/active_learning_workbench/death_event_ocr_labels.csv",
        help="Applied death OCR labels CSV.",
    )
    parser.add_argument("--candidates-csv", help="Optional death OCR candidate CSV.")
    parser.add_argument("--attribution-csv", help="Optional attributed death events CSV.")
    parser.add_argument("--output", default="outputs/death_events/death_annotation_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_csv = project_path(args.labels_csv)
    candidates_csv = project_path(args.candidates_csv) if args.candidates_csv else None
    attribution_csv = project_path(args.attribution_csv) if args.attribution_csv else None
    output = project_path(args.output)

    report = build_death_annotation_report(
        read_csv_rows(Path(labels_csv)),
        read_csv_rows(Path(candidates_csv)) if candidates_csv else [],
        read_csv_rows(Path(attribution_csv)) if attribution_csv else [],
    )
    write_json(Path(output), report)

    print(f"labels csv: {labels_csv}")
    if candidates_csv:
        print(f"candidates csv: {candidates_csv}")
    print(f"label count: {report['label_count']} coverage={report['coverage_ratio']}")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
