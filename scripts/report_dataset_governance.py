from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, project_path
from src.data_registry import DEFAULT_REGISTRY
from src.dataset_governance import build_dataset_governance_report, render_markdown, write_json


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


def write_text(path: Path, content: str) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote: {target}")


def main() -> int:
    args = parse_args()
    report = build_dataset_governance_report(
        registry_path=args.registry,
        dataset=args.dataset,
        labels=args.labels,
        model=args.model,
        min_images_per_class=args.min_images_per_class,
    )
    write_text(args.output, render_markdown(report))
    write_json(args.json_output, report)
    print(f"dataset governance status: {report['status']}")
    return 1 if args.strict and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
