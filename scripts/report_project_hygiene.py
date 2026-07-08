from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment
from src.project_hygiene import build_hygiene_report, render_markdown
from src.report_io import emit_markdown_or_stdout, strict_exit_code, write_json_report


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report project layout, output, and legacy-boundary hygiene.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Markdown report output. Prints to stdout when omitted.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report output.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless hygiene status is passed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_hygiene_report(args.root)
    markdown = render_markdown(report)

    emit_markdown_or_stdout(args.output, markdown)

    if args.json_output:
        write_json_report(args.json_output, report)

    print(f"project hygiene status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"passed"})


if __name__ == "__main__":
    raise SystemExit(main())
