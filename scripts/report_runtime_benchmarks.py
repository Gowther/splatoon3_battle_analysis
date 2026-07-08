from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime_benchmarks import (
    build_runtime_benchmark_report,
    default_runtime_reports,
    parse_runtime_report_arg,
    render_markdown,
)
from src.report_io import write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runtime report JSON files as benchmark baselines.")
    parser.add_argument("--runtime-report", action="append", default=[], help="label=path or path. May be repeated.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "runtime" / "runtime_benchmarks.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "runtime" / "runtime_benchmarks.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [parse_runtime_report_arg(item) for item in args.runtime_report] or default_runtime_reports()
    report = build_runtime_benchmark_report(reports)
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"runtime benchmark status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
