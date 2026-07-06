from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, model_path, project_path
from src.weapon_training import dataset_class_counts, format_summary, summarize_dataset, summary_as_json, write_labels


configure_environment()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect weapon-classifier training data and print a training plan.")
    parser.add_argument("--dataset", default="main_training_dataset", help="Weapon image dataset root.")
    parser.add_argument("--labels", default="main_weapon_list.txt", help="Weapon label list.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output", default="models/main_weapons_classification_weight.pth")
    parser.add_argument("--model", default="models/main_weapons_classification_weight.pth", help="Existing model used to validate output class count.")
    parser.add_argument("--write-labels", action="store_true", help="Replace labels with dataset class order.")
    parser.add_argument("--json", action="store_true", help="Print the dataset summary as JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when labels/data/model are inconsistent.")
    args = parser.parse_args()

    dataset = project_path(args.dataset)
    labels_path = project_path(args.labels)
    existing_model = project_path(args.model) if args.model else model_path("main_weapons_classification_weight.pth")

    if args.write_labels:
        labels = sorted(dataset_class_counts(dataset))
        write_labels(labels_path, labels)
        print(f"wrote labels: {labels_path}")

    summary = summarize_dataset(dataset, labels_path, existing_model)
    if args.json:
        print(summary_as_json(summary), end="")
    else:
        print(format_summary(summary))
        print(f"epochs: {args.epochs}")
        print(f"batch size: {args.batch_size}")
        print(f"device: {args.device}")
        print(f"planned output: {args.output}")
        print("status: planning only; use scripts/train_weapon_classifier.py to train")
    return 1 if args.strict and not summary.ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
