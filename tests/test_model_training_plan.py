from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.model_training_plan import build_model_training_plan, dataset_spec_report, render_markdown


class ModelTrainingPlanTests(unittest.TestCase):
    def test_build_model_training_plan_marks_ready_when_required_paths_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required = root / "data.yaml"
            required.write_text("names: []\n", encoding="utf-8")

            report = build_model_training_plan(
                {"targets": [{"id": "t1", "area": "detector", "required_paths": [str(required)]}]}
            )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["targets"][0]["status"], "ready")

    def test_dataset_spec_report_counts_yolo_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "a.jpg").write_text("", encoding="utf-8")
            (label_dir / "a.txt").write_text("", encoding="utf-8")
            data_yaml = root / "data.yaml"
            data_yaml.write_text("nc: 1\nnames: ['a']\n", encoding="utf-8")

            report = dataset_spec_report(
                {
                    "dataset_spec": {
                        "format": "yolo_detection",
                        "data_yaml": str(data_yaml),
                        "splits": {"train": {"images": str(image_dir), "labels": str(label_dir)}},
                        "class_names": ["a"],
                    }
                }
            )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["splits"][0]["image_count"], 1)

    def test_build_model_training_plan_marks_missing_data(self) -> None:
        report = build_model_training_plan(
            {"targets": [{"id": "t1", "area": "detector", "required_paths": ["missing/path"]}]}
        )

        self.assertEqual(report["status"], "needs_data")
        self.assertEqual(report["targets"][0]["missing_paths"], ["missing/path"])
        self.assertEqual(report["targets"][0]["dataset_status"], "not_configured")

    def test_build_model_training_plan_flags_unknown_target(self) -> None:
        report = build_model_training_plan({"targets": [{"id": "t1"}]}, target_ids=["unknown"])

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["missing_target_ids"], ["unknown"])

    def test_render_markdown_includes_candidate_command(self) -> None:
        markdown = render_markdown(
            {
                "status": "needs_data",
                "target_count": 1,
                "missing_target_ids": [],
                "targets": [
                    {
                        "id": "t1",
                        "area": "detector",
                        "status": "needs_data",
                        "missing_paths": ["missing/path"],
                        "candidate_command": "python train.py",
                    }
                ],
            }
        )

        self.assertIn("# Model Training Plan", markdown)
        self.assertIn("python train.py", markdown)
        self.assertIn("missing/path", markdown)

    def test_render_markdown_includes_dataset_dry_run(self) -> None:
        markdown = render_markdown(
            {
                "status": "ready",
                "target_count": 1,
                "missing_target_ids": [],
                "targets": [
                    {
                        "id": "t1",
                        "area": "detector",
                        "status": "ready",
                        "dataset_status": "ready",
                        "missing_paths": [],
                        "dataset_spec": {
                            "format": "yolo_detection",
                            "data_yaml": "data.yaml",
                            "status": "ready",
                            "class_count": 1,
                            "splits": [{"name": "train", "status": "ready", "image_count": 1, "label_count": 1}],
                        },
                    }
                ],
            }
        )

        self.assertIn("Dataset dry run", markdown)
        self.assertIn("yolo_detection", markdown)


if __name__ == "__main__":
    unittest.main()
