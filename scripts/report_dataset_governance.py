from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment
from src.data_registry import DEFAULT_REGISTRY
from src.dataset_governance import build_dataset_governance_report, render_markdown
from src.report_io import strict_exit_code, write_json_report, write_text_report


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report dataset, label, and registry metadata governance status.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", type=Path, default=ROOT / "main_training_dataset")
    parser.add_argument("--labels", type=Path, default=ROOT / "main_weapon_list.txt")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "main_weapons_classification_weight.pth")
    parser.add_argument("--min-images-per-class", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "dataset_governance.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "dataset_governance.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the governance report is passed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_dataset_governance_report(
        registry_path=args.registry,
        dataset=args.dataset,
        labels=args.labels,
        model=args.model,
        min_images_per_class=args.min_images_per_class,
    )
    write_text_report(args.output, render_markdown(report))
    write_json_report(args.json_output, report)
    print(f"dataset governance status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"passed"})


if __name__ == "__main__":
    raise SystemExit(main())
