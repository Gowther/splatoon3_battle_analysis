from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment
from src.data_registry import DEFAULT_REGISTRY
from src.match_intake import DEFAULT_EVALUATION_CONFIG
from src.model_quality import (
    DEFAULT_EVALUATION_RESULTS,
    build_quality_payload,
    render_markdown,
)
from src.report_io import emit_markdown_or_stdout, strict_exit_code, write_json_report


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a project-wide model/data quality overview.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--evaluation-results", type=Path, default=DEFAULT_EVALUATION_RESULTS)
    parser.add_argument("--dataset", type=Path, default=Path("main_training_dataset"))
    parser.add_argument("--labels", type=Path, default=Path("main_weapon_list.txt"))
    parser.add_argument("--weapon-model", type=Path, default=Path("models/main_weapons_classification_weight.pth"))
    parser.add_argument("--output", type=Path, help="Markdown report output. Prints to stdout when omitted.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON payload output.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the overview status is passed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_quality_payload(
        registry=args.registry,
        evaluation_config=args.evaluation_config,
        evaluation_results=args.evaluation_results,
        dataset=args.dataset,
        labels=args.labels,
        weapon_model=args.weapon_model,
    )
    markdown = render_markdown(payload)

    emit_markdown_or_stdout(args.output, markdown)

    if args.json_output:
        write_json_report(args.json_output, payload)

    print(f"quality overview status: {payload['overall_status']}")
    return strict_exit_code(payload["overall_status"], args.strict, passing_statuses={"passed"})


if __name__ == "__main__":
    raise SystemExit(main())
