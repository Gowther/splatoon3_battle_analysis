from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_manifest import build_experiment_manifest, parse_labeled_path, render_markdown, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest for data/model experiments.")
    parser.add_argument("--experiment-id", default="local_refactor_baseline")
    parser.add_argument("--source", action="append", default=[], help="label=path or path. Defaults to key config files when omitted.")
    parser.add_argument("--artifact", action="append", default=[], help="label=path or path. May be repeated.")
    parser.add_argument("--verification", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--include-git-status", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "experiment_manifest.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "experiment_manifest.json")
    return parser.parse_args()


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout


def main() -> int:
    args = parse_args()
    sources = [parse_labeled_path(item) for item in args.source] if args.source else None
    artifacts = [parse_labeled_path(item) for item in args.artifact]
    manifest = build_experiment_manifest(
        experiment_id=args.experiment_id,
        sources=sources,
        artifacts=artifacts,
        verification=args.verification,
        notes=args.note,
        git_status=git_status() if args.include_git_status else None,
    )
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(render_markdown(manifest), encoding="utf-8")
    write_json(args.json_output.expanduser(), manifest)
    print(f"experiment manifest status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
