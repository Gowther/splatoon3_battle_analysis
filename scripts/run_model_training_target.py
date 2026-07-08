from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_training_plan import DEFAULT_TRAINING_TARGETS, load_training_targets
from src.model_training_runner import build_training_launch_plan, execute_training_launch_plan, render_markdown
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute a configured detector/OCR training target.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_TARGETS)
    parser.add_argument("--target", required=True, help="Training target id from config/model_training_targets.json.")
    parser.add_argument("--execute", action="store_true", help="Actually run the configured candidate_command.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_training_launch.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_training_launch.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the launch plan is ready/completed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_training_launch_plan(load_training_targets(args.config), target_id=args.target)
    report = plan
    if args.execute:
        if plan["status"] == "ready":
            report = execute_training_launch_plan(plan)
        else:
            report = dict(plan)
            report["execution"] = {"status": "not_run", "returncode": None, "error": "launch plan is not ready"}
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"model training launch status: {report['status']}")
    return strict_exit_code(report["status"], args.strict or args.execute, passing_statuses={"ready", "completed"})


if __name__ == "__main__":
    raise SystemExit(main())
