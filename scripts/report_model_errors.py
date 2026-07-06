from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, project_path
from src.model_error_report import (
    ErrorThresholds,
    build_error_report,
    paths_from_evaluation_results,
    render_markdown,
    thresholds_as_json,
)


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report likely model/OCR error signals from analysis CSVs.")
    parser.add_argument("--csv", action="append", default=[], help="Analysis CSV path. May be repeated.")
    parser.add_argument("--evaluation-results", type=Path, help="evaluation_results.json from scripts/evaluate_matches.py.")
    parser.add_argument("--smoothed", action="store_true", help="Use smoothed_csv entries from evaluation results.")
    parser.add_argument("--output", type=Path, help="Markdown report output. Prints to stdout when omitted.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report output.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when high or warning issues are found.")
    parser.add_argument("--player-state-warning-ratio", type=float, default=0.2)
    parser.add_argument("--player-state-high-ratio", type=float, default=0.5)
    parser.add_argument("--weapon-missing-warning-ratio", type=float, default=0.1)
    parser.add_argument("--count-row-warning-ratio", type=float, default=0.2)
    parser.add_argument("--objective-row-info-ratio", type=float, default=0.2)
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote: {target}")


def main() -> int:
    args = parse_args()
    csv_paths = [Path(path) for path in args.csv]
    if args.evaluation_results:
        csv_paths.extend(paths_from_evaluation_results(args.evaluation_results, use_smoothed=args.smoothed))
    if not csv_paths:
        raise SystemExit("pass at least one --csv or --evaluation-results")

    thresholds = ErrorThresholds(
        player_state_missing_warning_ratio=args.player_state_warning_ratio,
        player_state_missing_high_ratio=args.player_state_high_ratio,
        weapon_missing_after_first_warning_ratio=args.weapon_missing_warning_ratio,
        count_row_warning_ratio=args.count_row_warning_ratio,
        objective_row_info_ratio=args.objective_row_info_ratio,
    )
    report = build_error_report(csv_paths, thresholds=thresholds)
    report["thresholds"] = thresholds_as_json(thresholds)
    markdown = render_markdown(report)

    if args.output:
        write_text(args.output, markdown)
    else:
        print(markdown, end="")
    if args.json_output:
        write_text(args.json_output, json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"model error report status: {report['status']}")
    return 1 if args.strict and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
