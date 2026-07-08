from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, project_path
from src.model_experiments import (
    DEFAULT_EXPERIMENT_CONFIG,
    build_experiment_plan,
    load_json,
    load_optional_json,
    render_markdown,
)
from src.report_io import write_json_report, write_text_report


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan model replacement experiments from current quality signals.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--model-errors", type=Path, help="Optional model_errors.json from scripts/report_model_errors.py.")
    parser.add_argument("--heatmap-comparison", type=Path, help="Optional heatmap_comparison.json.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_experiment_plan.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_experiment_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    model_errors = load_optional_json(args.model_errors)
    heatmap_comparison = load_optional_json(args.heatmap_comparison)
    plan = build_experiment_plan(config, model_errors=model_errors, heatmap_comparison=heatmap_comparison)

    output = write_text_report(args.output, render_markdown(plan))
    write_json_report(args.json_output, plan)

    print(f"wrote model experiment plan: {output}")
    print(f"wrote model experiment plan json: {project_path(args.json_output)}")
    print(
        "model experiment priorities: high={high_priority} medium={medium_priority} baseline={baseline_priority}".format(
            **plan["summary"]
        )
    )
    for experiment in plan["experiments"]:
        print(f"- {experiment['priority']}: {experiment['id']} ({experiment['candidate']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
