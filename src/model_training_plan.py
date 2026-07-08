from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.core.paths import ROOT, project_path


DEFAULT_TRAINING_TARGETS = ROOT / "config" / "model_training_targets.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_SUFFIXES = {".txt"}


def load_training_targets(path: Path = DEFAULT_TRAINING_TARGETS) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as f:
        return json.load(f)


def count_files(path: Path, suffixes: set[str]) -> int:
    target = project_path(path)
    if not target.exists() or not target.is_dir():
        return 0
    return sum(1 for item in target.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def read_yaml(path: Path) -> dict[str, Any]:
    target = project_path(path)
    if not target.exists():
        return {}
    with target.open(encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    return value if isinstance(value, dict) else {}


def split_dataset_report(split_name: str, split: dict[str, Any]) -> dict[str, Any]:
    image_dir = str(split.get("images", ""))
    label_dir = str(split.get("labels", ""))
    image_path = project_path(image_dir)
    label_path = project_path(label_dir)
    image_count = count_files(image_path, IMAGE_SUFFIXES)
    label_count = count_files(label_path, LABEL_SUFFIXES)
    missing_paths = [
        path
        for path, exists in (
            (image_dir, image_path.exists()),
            (label_dir, label_path.exists()),
        )
        if not exists
    ]
    status = "ready"
    if missing_paths or image_count == 0 or label_count == 0:
        status = "needs_data"
    elif image_count != label_count:
        status = "needs_review"
    return {
        "name": split_name,
        "status": status,
        "images": image_dir,
        "labels": label_dir,
        "image_count": image_count,
        "label_count": label_count,
        "missing_paths": missing_paths,
    }


def dataset_spec_report(target: dict[str, Any]) -> dict[str, Any]:
    spec = target.get("dataset_spec") or {}
    if not isinstance(spec, dict) or not spec:
        return {"status": "not_configured", "format": "", "splits": [], "warnings": [], "blockers": []}

    data_yaml = str(spec.get("data_yaml", ""))
    data_yaml_path = project_path(data_yaml)
    yaml_payload = read_yaml(data_yaml_path)
    class_names = list(spec.get("class_names", []))
    splits_config = spec.get("splits", {}) if isinstance(spec.get("splits"), dict) else {}
    splits = [split_dataset_report(name, split) for name, split in sorted(splits_config.items())]
    blockers: list[str] = []
    warnings: list[str] = []
    if not data_yaml_path.exists():
        blockers.append(f"missing data yaml: {data_yaml}")
    if not splits:
        blockers.append("dataset_spec.splits is empty")
    for split in splits:
        if split["status"] == "needs_data":
            blockers.append(f"{split['name']} split needs data")
        elif split["status"] == "needs_review":
            warnings.append(f"{split['name']} image/label counts differ")

    yaml_names = yaml_payload.get("names", [])
    yaml_nc = yaml_payload.get("nc")
    if yaml_payload:
        if isinstance(yaml_names, list) and class_names and list(yaml_names) != class_names:
            warnings.append("data.yaml names differ from dataset_spec.class_names")
        if yaml_nc is not None and class_names and int(yaml_nc) != len(class_names):
            warnings.append("data.yaml nc differs from dataset_spec.class_names length")

    status = "ready"
    if blockers:
        status = "needs_data"
    elif warnings:
        status = "needs_review"
    return {
        "status": status,
        "format": spec.get("format", ""),
        "data_yaml": data_yaml,
        "data_yaml_exists": data_yaml_path.exists(),
        "class_count": len(class_names),
        "yaml_class_count": len(yaml_names) if isinstance(yaml_names, list) else None,
        "yaml_nc": yaml_nc,
        "splits": splits,
        "warnings": warnings,
        "blockers": blockers,
    }


def target_report(target: dict[str, Any]) -> dict[str, Any]:
    required = [str(path) for path in target.get("required_paths", [])]
    missing = [path for path in required if not project_path(path).exists()]
    dataset = dataset_spec_report(target)
    status = "ready" if not missing else "needs_data"
    if dataset["status"] == "needs_data":
        status = "needs_data"
    return {
        "id": target.get("id", ""),
        "area": target.get("area", ""),
        "model_id": target.get("model_id", ""),
        "status": status,
        "dataset_status": dataset["status"],
        "dataset_root": target.get("dataset_root", ""),
        "dataset_spec": dataset,
        "required_paths": required,
        "missing_paths": missing,
        "candidate_output_dir": target.get("candidate_output_dir", ""),
        "baseline_model": target.get("baseline_model", ""),
        "candidate_command": target.get("candidate_command", ""),
        "promotion_gate": target.get("promotion_gate", ""),
        "notes": target.get("notes", ""),
    }


def build_model_training_plan(config: dict[str, Any], *, target_ids: list[str] | None = None) -> dict[str, Any]:
    selected = set(target_ids or [])
    targets = [
        target_report(target)
        for target in config.get("targets", [])
        if not selected or target.get("id") in selected
    ]
    missing_ids = sorted(selected - {target["id"] for target in targets})
    status = "ready" if targets and all(target["status"] == "ready" for target in targets) else "needs_data"
    if missing_ids:
        status = "failed"
    return {
        "schema_version": 1,
        "status": status,
        "target_count": len(targets),
        "missing_target_ids": missing_ids,
        "targets": targets,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Training Plan",
        "",
        f"- status: `{report.get('status')}`",
        f"- target_count: {report.get('target_count', 0)}",
        f"- missing_target_ids: {', '.join(report.get('missing_target_ids', [])) or '-'}",
        "",
        "| id | area | status | dataset | missing paths |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for target in report.get("targets", []):
        lines.append(
            f"| `{target.get('id', '')}` | {target.get('area', '')} | `{target.get('status', '')}` | "
            f"`{target.get('dataset_status', '')}` | "
            f"{len(target.get('missing_paths', []))} |"
        )
    for target in report.get("targets", []):
        lines.extend(
            [
                "",
                f"## {target.get('id', '')}",
                "",
                f"- model_id: `{target.get('model_id', '')}`",
                f"- dataset_root: `{target.get('dataset_root', '')}`",
                f"- candidate_output_dir: `{target.get('candidate_output_dir', '')}`",
                f"- baseline_model: `{target.get('baseline_model', '')}`",
                f"- promotion_gate: `{target.get('promotion_gate', '')}`",
                f"- candidate_command: `{target.get('candidate_command', '')}`",
                f"- notes: {target.get('notes', '')}",
            ]
        )
        dataset = target.get("dataset_spec", {})
        if dataset:
            lines.extend(
                [
                    "",
                    "Dataset dry run:",
                    f"- format: `{dataset.get('format', '')}`",
                    f"- data_yaml: `{dataset.get('data_yaml', '')}`",
                    f"- status: `{dataset.get('status', '')}`",
                    f"- class_count: {dataset.get('class_count', '')}",
                    "",
                    "| split | status | images | labels |",
                    "| --- | --- | ---: | ---: |",
                ]
            )
            for split in dataset.get("splits", []):
                lines.append(
                    f"| {split.get('name', '')} | `{split.get('status', '')}` | "
                    f"{split.get('image_count', 0)} | {split.get('label_count', 0)} |"
                )
            if dataset.get("blockers"):
                lines.extend(["", "Dataset blockers:"])
                lines.extend(f"- {item}" for item in dataset["blockers"])
            if dataset.get("warnings"):
                lines.extend(["", "Dataset warnings:"])
                lines.extend(f"- {item}" for item in dataset["warnings"])
        if target.get("missing_paths"):
            lines.extend(["", "Missing paths:"])
            lines.extend(f"- `{path}`" for path in target["missing_paths"])
    lines.append("")
    return "\n".join(lines)
