from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.model_training_plan import build_model_training_plan, render_markdown


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

    def test_build_model_training_plan_marks_missing_data(self) -> None:
        report = build_model_training_plan(
            {"targets": [{"id": "t1", "area": "detector", "required_paths": ["missing/path"]}]}
        )

        self.assertEqual(report["status"], "needs_data")
        self.assertEqual(report["targets"][0]["missing_paths"], ["missing/path"])

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


if __name__ == "__main__":
    unittest.main()
