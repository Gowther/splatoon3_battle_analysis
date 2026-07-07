from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.change_package import build_change_package, render_markdown, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the current worktree as a review/commit package.")
    parser.add_argument("--verification", action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "change_package.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "change_package.json")
    return parser.parse_args()


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout


def main() -> int:
    args = parse_args()
    report = build_change_package(git_status(), verification=args.verification)
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(args.json_output.expanduser(), report)
    print(f"change package paths: {report['change_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
