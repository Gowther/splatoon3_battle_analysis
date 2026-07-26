from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.outputs_retention import (
    DEFAULT_OUTPUTS_DIR,
    REGENERABLE_DIR_NAMES,
    apply_retention_plan,
    build_retention_plan,
    format_size,
    render_markdown,
)
from src.report_io import write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report and optionally reclaim regenerable space under outputs/.")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--kind", action="append", dest="kinds", help=f"Regenerable dir name to target. Defaults to {', '.join(REGENERABLE_DIR_NAMES)}.")
    parser.add_argument("--min-size-mb", type=float, default=0.0, help="Ignore candidates smaller than this.")
    parser.add_argument("--output", type=Path, help="Optional Markdown report.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report.")
    parser.add_argument("--apply", action="store_true", help="Actually delete the planned directories.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_retention_plan(
        args.outputs_dir,
        dir_names=args.kinds or REGENERABLE_DIR_NAMES,
        min_size_bytes=int(args.min_size_mb * 1024 * 1024),
    )
    result = apply_retention_plan(plan, dry_run=not args.apply)

    if args.output:
        write_text_report(args.output.expanduser(), render_markdown(plan, result))
    if args.json_output:
        write_json_report(args.json_output.expanduser(), {"plan": plan, "result": result})

    print(f"outputs total: {plan['total_display']}")
    print(f"reclaimable: {plan['reclaimable_display']} ({plan['reclaimable_percent']}%) across {len(plan['candidates'])} dirs")
    if args.apply:
        print(f"removed {result['removed_count']} dirs, freed {format_size(result['freed_bytes'])}")
    else:
        print("dry run; pass --apply to delete")
    for error in result["errors"]:
        print(f"- {error}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
