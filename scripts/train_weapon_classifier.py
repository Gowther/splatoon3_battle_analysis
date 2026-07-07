from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment, model_path, project_path
from src.weapon_training import (
    build_dataloaders,
    load_labels,
    load_initial_classifier,
    summarize_dataset,
    summary_as_json,
    train_classifier,
    write_labels,
)


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Splatoon 3 weapon classifier.")
    parser.add_argument("--dataset", default="main_training_dataset", help="ImageFolder-style dataset root.")
    parser.add_argument("--labels", default="main_weapon_list.txt", help="Class label list written beside the model.")
    parser.add_argument("--output", default="models/main_weapons_classification_weight.pth", help="Output .pth model.")
    parser.add_argument("--metrics", default="outputs/weapon_training_metrics.json", help="Training metrics JSON.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ResNet18 ImageNet weights.")
    parser.add_argument(
        "--initial-model",
        help="Fine-tune from an existing full .pth model. The output class count must match the dataset classes.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-class", type=int, help="Limit each class for smoke tests.")
    parser.add_argument("--write-labels", action="store_true", help="Write labels from dataset class order before training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print the plan without training.")
    parser.add_argument("--json", action="store_true", help="Print the dataset summary as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = project_path(args.dataset)
    labels_path = project_path(args.labels)
    output_path = project_path(args.output)
    metrics_path = project_path(args.metrics)
    initial_model_path = project_path(args.initial_model) if args.initial_model else None
    summary = summarize_dataset(dataset, labels_path, model_path("main_weapons_classification_weight.pth"))

    if args.json:
        print(summary_as_json(summary), end="")
    else:
        print(f"dataset: {summary.dataset}")
        print(f"images: {summary.images}")
        print(f"dataset classes: {summary.dataset_classes}")
        print(f"labels: {summary.label_classes} from {summary.labels}")
        if summary.model_output_classes is not None:
            print(f"model output classes: {summary.model_output_classes}")
        print(f"epochs: {args.epochs}")
        print(f"batch size: {args.batch_size}")
        print(f"device: {args.device}")
        if initial_model_path:
            print(f"initial model: {initial_model_path}")
        print(f"output: {output_path}")
        print(f"metrics: {metrics_path}")

    loaders, sizes, class_names, _ = build_dataloaders(
        dataset,
        args.batch_size,
        args.val_split,
        args.test_split,
        args.num_workers,
        args.seed,
        args.max_samples_per_class,
    )
    print(f"splits: {json.dumps(sizes, sort_keys=True)}")

    if args.write_labels:
        write_labels(labels_path, class_names)
        print(f"wrote labels: {labels_path}")
    elif load_labels(labels_path) != class_names:
        print("error: label list does not match dataset class order; rerun with --write-labels after reviewing.")
        return 1

    if initial_model_path:
        try:
            load_initial_classifier(initial_model_path, len(class_names))
        except ValueError as exc:
            print(f"error: {exc}")
            return 1

    if args.dry_run:
        print("status: dry run only; no training executed")
        return 0

    metrics = train_classifier(
        dataset,
        output_path,
        metrics_path,
        labels_path,
        args.epochs,
        args.batch_size,
        args.device,
        args.pretrained,
        args.learning_rate,
        args.momentum,
        args.step_size,
        args.gamma,
        args.val_split,
        args.test_split,
        args.num_workers,
        args.seed,
        args.max_samples_per_class,
        initial_model_path,
    )
    print(f"wrote model: {output_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"best val accuracy: {metrics['best_val_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
