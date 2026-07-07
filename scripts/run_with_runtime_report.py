from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment
from src.runtime_report import build_runtime_report, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command and write a runtime report.")
    parser.add_argument("--name", default="command")
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("missing command after --")
    env = configure_environment()
    started = time.perf_counter()
    result = subprocess.run(command, cwd=args.cwd.expanduser(), env=env)
    duration = round(time.perf_counter() - started, 4)
    report = build_runtime_report(
        args.name,
        [{"label": args.name, "command": " ".join(command), "duration_seconds": duration, "returncode": result.returncode}],
        metadata={"cwd": str(args.cwd.expanduser()), "returncode": result.returncode},
    )
    write_json(args.output.expanduser(), report)
    if args.markdown_output:
        write_markdown(args.markdown_output.expanduser(), report)
    print(f"runtime report: {args.output.expanduser()}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
