from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_training_plan import (
    DEFAULT_TRAINING_TARGETS,
    build_model_training_plan,
    load_training_targets,
    render_markdown,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan detector/OCR model training data requirements.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_TARGETS)
    parser.add_argument("--target", action="append", default=[], help="Training target id. May be repeated.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_training_plan.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_training_plan.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless selected targets are ready.")
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote: {path}")


def main() -> int:
    args = parse_args()
    report = build_model_training_plan(load_training_targets(args.config), target_ids=args.target or None)
    write_text(args.output.expanduser(), render_markdown(report))
    write_json(args.json_output.expanduser(), report)
    print(f"model training plan status: {report['status']}")
    return 1 if args.strict and report["status"] != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
