from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_training_plan import DEFAULT_TRAINING_TARGETS, build_model_training_plan, load_training_targets, render_markdown
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run detector/OCR training dataset layouts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_TARGETS)
    parser.add_argument("--target", action="append", default=[], help="Training target id. May be repeated.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_training_datasets.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_training_datasets.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless selected dataset targets are ready.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_model_training_plan(load_training_targets(args.config), target_ids=args.target or None)
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"model training dataset status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"ready"})


if __name__ == "__main__":
    raise SystemExit(main())
