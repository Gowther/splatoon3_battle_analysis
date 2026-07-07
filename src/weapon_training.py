from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from src.core.paths import ROOT, project_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class WeaponDatasetSummary:
    dataset: str
    labels: str
    images: int
    dataset_classes: int
    label_classes: int
    model_output_classes: int | None
    missing_dataset_classes: list[str]
    missing_label_classes: list[str]
    duplicate_labels: list[str]
    class_counts: dict[str, int]

    @property
    def ok(self) -> bool:
        if self.missing_dataset_classes or self.missing_label_classes or self.duplicate_labels:
            return False
        if self.model_output_classes is not None and self.model_output_classes != self.label_classes:
            return False
        return self.dataset_classes == self.label_classes


def resolve_project_path(path: str | Path) -> Path:
    return project_path(path)


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def label_from_path(path: Path, dataset_root: Path) -> str:
    relative = path.relative_to(dataset_root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return path.stem.split("_")[0]


def load_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_labels(path: Path, labels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{label}\n" for label in labels), encoding="utf-8")


def dataset_class_counts(dataset_root: Path) -> Counter[str]:
    return Counter(label_from_path(path, dataset_root) for path in image_files(dataset_root))


def duplicate_labels(labels: Sequence[str]) -> list[str]:
    counts = Counter(labels)
    return sorted(label for label, count in counts.items() if count > 1)


def load_torch_model(path: Path, device: str = "cpu"):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def model_output_count(model: nn.Module) -> int | None:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    return None


def summarize_dataset(dataset: Path, labels_path: Path, model_path: Path | None = None) -> WeaponDatasetSummary:
    counts = dataset_class_counts(dataset)
    dataset_classes = sorted(counts)
    labels = load_labels(labels_path)
    output_classes = None
    if model_path and model_path.exists():
        output_classes = model_output_count(load_torch_model(model_path, "cpu"))
    return WeaponDatasetSummary(
        dataset=str(dataset),
        labels=str(labels_path),
        images=sum(counts.values()),
        dataset_classes=len(dataset_classes),
        label_classes=len(labels),
        model_output_classes=output_classes,
        missing_dataset_classes=[label for label in labels if label not in counts],
        missing_label_classes=[label for label in dataset_classes if label not in labels],
        duplicate_labels=duplicate_labels(labels),
        class_counts=dict(sorted(counts.items())),
    )


def format_summary(summary: WeaponDatasetSummary, top_n: int = 20) -> str:
    lines = [
        f"dataset: {summary.dataset}",
        f"images: {summary.images}",
        f"dataset classes: {summary.dataset_classes}",
        f"labels: {summary.label_classes} from {summary.labels}",
    ]
    if summary.model_output_classes is not None:
        lines.append(f"model output classes: {summary.model_output_classes}")
    if summary.class_counts:
        lines.append("top labels:")
        for label, count in Counter(summary.class_counts).most_common(top_n):
            lines.append(f"- {label}: {count}")
    else:
        lines.append("warning: no training images found")
    if summary.missing_dataset_classes:
        lines.append(f"warning: {len(summary.missing_dataset_classes)} labels have no images in this dataset")
        lines.extend(f"- missing dataset class: {label}" for label in summary.missing_dataset_classes[:20])
    if summary.missing_label_classes:
        lines.append(f"warning: {len(summary.missing_label_classes)} dataset classes are missing from labels")
        lines.extend(f"- missing label: {label}" for label in summary.missing_label_classes[:20])
    if summary.duplicate_labels:
        lines.append(f"warning: {len(summary.duplicate_labels)} duplicate labels")
        lines.extend(f"- duplicate label: {label}" for label in summary.duplicate_labels[:20])
    if summary.model_output_classes is not None and summary.model_output_classes != summary.label_classes:
        lines.append("warning: model output class count does not match label count")
    lines.append("status: ready" if summary.ok else "status: needs attention")
    return "\n".join(lines)


def build_resnet18_classifier(num_classes: int, pretrained: bool = False, dropout: float = 0.5) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def load_initial_classifier(path: Path, expected_classes: int) -> nn.Module:
    model = load_torch_model(path, "cpu")
    output_classes = model_output_count(model)
    if output_classes != expected_classes:
        raise ValueError(
            f"Initial model output classes ({output_classes}) do not match dataset classes ({expected_classes})."
        )
    return model


def training_transform(augment: bool = True) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((64, 64))]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(steps)


def choose_training_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def limited_indices(samples: Sequence[tuple[str, int]], max_samples_per_class: int | None, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, target) in enumerate(samples):
        by_class[int(target)].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:max_samples_per_class] if max_samples_per_class else indices)
    return sorted(selected)


def stratified_split_indices(
    samples: Sequence[tuple[str, int]],
    indices: Sequence[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        _, target = samples[index]
        by_class[int(target)].append(index)

    split = {"train": [], "val": [], "test": []}
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        total = len(class_indices)
        test_count = int(round(total * test_split)) if total > 2 else 0
        val_count = int(round(total * val_split)) if total - test_count > 1 else 0
        if total - test_count - val_count < 1:
            val_count = max(0, total - test_count - 1)
        split["test"].extend(class_indices[:test_count])
        split["val"].extend(class_indices[test_count : test_count + val_count])
        split["train"].extend(class_indices[test_count + val_count :])
    for values in split.values():
        values.sort()
    return split


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    val_split: float,
    test_split: float,
    num_workers: int,
    seed: int,
    max_samples_per_class: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, int], list[str], dict[str, list[int]]]:
    train_source = datasets.ImageFolder(str(dataset_root), transform=training_transform(True))
    eval_source = datasets.ImageFolder(str(dataset_root), transform=training_transform(False))
    indices = limited_indices(train_source.samples, max_samples_per_class, seed)
    splits = stratified_split_indices(train_source.samples, indices, val_split, test_split, seed)
    dataset_by_phase = {
        "train": Subset(train_source, splits["train"]),
        "val": Subset(eval_source, splits["val"]),
        "test": Subset(eval_source, splits["test"]),
    }
    loaders = {
        phase: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=phase == "train",
            num_workers=num_workers,
        )
        for phase, dataset in dataset_by_phase.items()
        if len(dataset) > 0
    }
    sizes = {phase: len(dataset) for phase, dataset in dataset_by_phase.items()}
    return loaders, sizes, list(train_source.classes), splits


def run_phase(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    running_correct = 0
    sample_count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if optimizer:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if optimizer:
                loss.backward()
                optimizer.step()
        batch_size = int(inputs.size(0))
        running_loss += float(loss.item()) * batch_size
        running_correct += int(torch.sum(preds == labels.data).item())
        sample_count += batch_size
    if sample_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": running_loss / sample_count, "accuracy": running_correct / sample_count}


def train_classifier(
    dataset_root: Path,
    output_path: Path,
    metrics_path: Path,
    labels_path: Path,
    epochs: int,
    batch_size: int,
    device_name: str,
    pretrained: bool,
    learning_rate: float,
    momentum: float,
    step_size: int,
    gamma: float,
    val_split: float,
    test_split: float,
    num_workers: int,
    seed: int,
    max_samples_per_class: int | None = None,
    initial_model_path: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    loaders, sizes, class_names, splits = build_dataloaders(
        dataset_root,
        batch_size,
        val_split,
        test_split,
        num_workers,
        seed,
        max_samples_per_class,
    )
    if not loaders.get("train"):
        raise RuntimeError("No training images found.")

    device = choose_training_device(device_name)
    if initial_model_path:
        model = load_initial_classifier(initial_model_path, len(class_names)).to(device)
        initialization = "initial_model"
    else:
        model = build_resnet18_classifier(len(class_names), pretrained=pretrained).to(device)
        initialization = "pretrained_resnet18" if pretrained else "resnet18"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epoch_metrics: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(epochs):
        train_metrics = run_phase(model, loaders["train"], criterion, optimizer, device)
        val_metrics = run_phase(model, loaders["val"], criterion, None, device) if "val" in loaders else {}
        scheduler.step()
        val_acc = float(val_metrics.get("accuracy", train_metrics["accuracy"]))
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        epoch_metrics.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_acc:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    test_metrics = run_phase(model, loaders["test"], criterion, None, device) if "test" in loaders else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output_path)

    metrics = {
        "dataset": str(dataset_root),
        "output": str(output_path),
        "labels": str(labels_path),
        "initialization": initialization,
        "initial_model": str(initial_model_path) if initial_model_path else "",
        "classes": class_names,
        "class_count": len(class_names),
        "sizes": sizes,
        "splits": {phase: len(indices) for phase, indices in splits.items()},
        "epochs": epoch_metrics,
        "best_val_accuracy": best_val_acc,
        "test": test_metrics,
        "elapsed_seconds": time.time() - started,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def icon_labels(icon_dir: Path) -> list[str]:
    return sorted(path.stem for path in icon_dir.glob("*.png") if not path.name.startswith("."))


def background_images(background_dir: Path) -> list[Path]:
    return image_files(background_dir)


def synthetic_weapon_image(
    icon_path: Path,
    backgrounds: Sequence[Path],
    rng: random.Random,
    size: tuple[int, int] = (200, 200),
) -> Image.Image:
    if backgrounds:
        base = Image.open(rng.choice(list(backgrounds))).convert("RGB")
        base = base.resize(size).filter(ImageFilter.BLUR)
    else:
        base = Image.new("RGB", size, color=(32, 32, 32))

    logo = Image.open(icon_path).convert("RGBA")
    logo_w, logo_h = logo.size
    scale = rng.uniform(0.5, 1.0)
    logo = logo.resize((max(1, int(logo_w * scale)), max(1, int(logo_h * scale))))
    logo = ImageEnhance.Color(logo).enhance(rng.uniform(0.2, 1.0))
    logo = ImageEnhance.Brightness(logo).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.2:
        logo = logo.rotate(rng.randint(-10, 10), expand=True)
    x = int(size[0] * 0.5 - logo.size[0] * 0.5 + rng.uniform(-0.1, 0.1) * size[0])
    y = int(size[1] * 0.5 - logo.size[1] * 0.5 + rng.uniform(-0.1, 0.1) * size[1])
    base.paste(logo, (x, y), logo)
    return base


def generate_synthetic_dataset(
    icon_dir: Path,
    background_dir: Path,
    output_dir: Path,
    images_per_class: int,
    seed: int,
    write_label_path: Path | None = None,
) -> dict[str, Any]:
    labels = icon_labels(icon_dir)
    backgrounds = background_images(background_dir)
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for label in labels:
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        icon_path = icon_dir / f"{label}.png"
        for index in range(images_per_class):
            image = synthetic_weapon_image(icon_path, backgrounds, rng)
            image.save(label_dir / f"{label}_{index:05d}.jpg", quality=95)
            generated += 1
    if write_label_path:
        write_labels(write_label_path, labels)
    return {
        "icons": str(icon_dir),
        "backgrounds": str(background_dir),
        "output": str(output_dir),
        "classes": len(labels),
        "background_count": len(backgrounds),
        "images_per_class": images_per_class,
        "generated": generated,
        "labels": str(write_label_path) if write_label_path else None,
    }


def summary_as_json(summary: WeaponDatasetSummary) -> str:
    return json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n"
