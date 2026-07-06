from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_csv import populated, quality_warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a lightweight Markdown report for an analysis CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", type=Path, help="Markdown output path. Defaults next to the CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    output = args.output or csv_path.with_suffix(".report.md")
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    eight_state_rows = sum(1 for row in rows if all(populated(row, f"player_state_{i}") for i in range(1, 9)))
    weapon_rows = sum(1 for row in rows if populated(row, "weapon_1"))
    count_rows = sum(1 for row in rows if populated(row, "count_left") or populated(row, "count_right"))
    message_rows = sum(1 for row in rows if populated(row, "message"))
    warnings = quality_warnings(rows, fieldnames)

    lines = [
        f"# CSV Report: {csv_path.name}",
        "",
        f"- rows: {len(rows)}",
        f"- elapsed: {rows[0].get('elapsed_time') if rows else ''} -> {rows[-1].get('elapsed_time') if rows else ''}",
        f"- 8-player state rows: {eight_state_rows}",
        f"- weapon rows: {weapon_rows}",
        f"- count rows: {count_rows}",
        f"- message rows: {message_rows}",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
