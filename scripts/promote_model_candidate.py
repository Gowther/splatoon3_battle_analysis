from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_promotion import (
    DEFAULT_PROMOTION_BACKUP_DIR,
    apply_model_promotion,
    build_model_promotion_plan,
    render_markdown,
)
from src.model_registry import DEFAULT_MODEL_REGISTRY, load_model_registry
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or apply promotion of a candidate model into the registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--model-id", required=True, help="Registered model id to promote.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate model file.")
    parser.add_argument("--validation-report", type=Path, help="JSON report from the promotion gate.")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_PROMOTION_BACKUP_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_promotion_plan.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_promotion_plan.json")
    parser.add_argument("--apply", action="store_true", help="Copy the candidate and update the registry.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the plan is ready/promoted.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_model_promotion_plan(
        load_model_registry(args.registry),
        model_id=args.model_id,
        candidate_path=args.candidate,
        validation_report=args.validation_report,
        backup_dir=args.backup_dir,
    )
    report = plan
    if args.apply:
        if plan["status"] == "ready":
            report = apply_model_promotion(args.registry, plan, backup_dir=args.backup_dir)
        else:
            report = dict(plan)
            report["apply_error"] = "promotion was not applied because the plan is not ready"
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"model promotion status: {report['status']}")
    return strict_exit_code(report["status"], args.strict or args.apply, passing_statuses={"ready", "promoted"})


if __name__ == "__main__":
    raise SystemExit(main())
