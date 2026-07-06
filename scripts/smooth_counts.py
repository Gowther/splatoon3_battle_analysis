from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.count_smoothing import CountSmoothingConfig, correction_summary, read_csv, smooth_rows, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Smooth isolated count/penalty OCR jumps in an analysis CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional JSON correction report.")
    parser.add_argument("--max-jump", type=int, default=20)
    parser.add_argument("--neighbor-tolerance", type=int, default=3)
    parser.add_argument("--lookahead", type=int, default=3)
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser()
    output = args.output.expanduser()
    report = args.report.expanduser() if args.report else None
    config = CountSmoothingConfig(
        max_jump=args.max_jump,
        neighbor_tolerance=args.neighbor_tolerance,
        lookahead=args.lookahead,
    )

    fieldnames, rows = read_csv(csv_path)
    smoothed_rows, corrections = smooth_rows(rows, config=config)
    write_csv(output, fieldnames, smoothed_rows)
    summary = correction_summary(corrections)

    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote smoothed csv: {output}")
    print(f"count corrections: {summary['total_corrections']}")
    for field, count in summary["by_field"].items():
        print(f"- {field}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
