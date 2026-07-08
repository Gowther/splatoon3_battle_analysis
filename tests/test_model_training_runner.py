from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from src.model_training_runner import build_training_launch_plan, normalized_command_args, render_markdown


class ModelTrainingRunnerTests(unittest.TestCase):
    def test_normalized_command_args_uses_current_python(self) -> None:
        args = normalized_command_args("python yolov5/train.py --epochs 1")

        self.assertEqual(args[0], sys.executable)
        self.assertEqual(args[1], "yolov5/train.py")

    def test_build_launch_plan_ready_for_configured_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required = root / "data.yaml"
            required.write_text("names: []\n", encoding="utf-8")

            report = build_training_launch_plan(
                {
                    "targets": [
                        {
                            "id": "detector",
                            "area": "ui_detection",
                            "required_paths": [str(required)],
                            "candidate_command": "python yolov5/train.py --data data.yaml",
                        }
                    ]
                },
                target_id="detector",
            )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["blockers"], [])
        self.assertTrue(report["command_args"][0].endswith("python") or report["command_args"][0] == sys.executable)

    def test_build_launch_plan_blocks_missing_data(self) -> None:
        report = build_training_launch_plan(
            {
                "targets": [
                    {
                        "id": "detector",
                        "required_paths": ["missing/data.yaml"],
                        "candidate_command": "python yolov5/train.py",
                    }
                ]
            },
            target_id="detector",
        )

        self.assertEqual(report["status"], "needs_data")
        self.assertIn("missing/data.yaml", report["blockers"])

    def test_build_launch_plan_flags_unknown_target(self) -> None:
        report = build_training_launch_plan({"targets": [{"id": "detector"}]}, target_id="missing")

        self.assertEqual(report["status"], "failed")
        self.assertIn("unknown training target", report["blockers"][0])

    def test_render_markdown_includes_command_and_blockers(self) -> None:
        markdown = render_markdown(
            {
                "status": "needs_data",
                "target_id": "detector",
                "target": {"dataset_status": "needs_data"},
                "command": "python yolov5/train.py",
                "blockers": ["missing/data.yaml"],
                "warnings": [],
                "execution": {"status": "not_run", "returncode": None},
            }
        )

        self.assertIn("# Model Training Launch Plan", markdown)
        self.assertIn("python yolov5/train.py", markdown)
        self.assertIn("missing/data.yaml", markdown)


if __name__ == "__main__":
    unittest.main()
