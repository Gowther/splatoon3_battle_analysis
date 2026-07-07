from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, project_path
from src.project_hygiene import build_hygiene_report, render_markdown


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report project layout, output, and legacy-boundary hygiene.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Markdown report output. Prints to stdout when omitted.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report output.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless hygiene status is passed.")
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote: {target}")


def main() -> int:
    args = parse_args()
    report = build_hygiene_report(args.root)
    markdown = render_markdown(report)

    if args.output:
        write_text(args.output, markdown)
    else:
        print(markdown, end="")

    if args.json_output:
        write_text(args.json_output, json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"project hygiene status: {report['status']}")
    return 1 if args.strict and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
