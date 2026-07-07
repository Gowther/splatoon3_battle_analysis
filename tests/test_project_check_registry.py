from __future__ import annotations

import unittest
from pathlib import Path

from src.project_check_registry import experiment_delivery_steps, heatmap_annotation_steps, model_error_steps, tooling_smoke_steps


class ProjectCheckRegistryTests(unittest.TestCase):
    def test_heatmap_annotation_steps_are_named_and_ordered(self) -> None:
        steps = heatmap_annotation_steps(Path("python"), Path("/tmp/work"))

        self.assertEqual(steps[0].name, "heatmap annotation round helper")
        self.assertEqual(steps[-1].name, "heatmap parameter experiment helper")

    def test_tooling_smoke_steps_include_intake_and_quality_inputs(self) -> None:
        steps = tooling_smoke_steps(Path("python"), Path("/tmp/work"), "footages/match_1.mp4", "cpu")
        names = [step.name for step in steps]

        self.assertIn("match intake dry run", names)
        self.assertIn("heatmap quality loop helper", names)
        self.assertIn("model registry report helper", names)
        self.assertIn("model training plan helper", names)
        self.assertTrue(any("footages/match_1.mp4" in [str(part) for part in step.command] for step in steps))

    def test_model_error_steps_use_sample_csv(self) -> None:
        steps = model_error_steps(Path("python"), Path("/tmp/work"))

        self.assertEqual(steps[0].name, "model error report helper")
        self.assertIn(Path("/tmp/work") / "sample.csv", steps[0].command)

    def test_experiment_delivery_steps_include_manifest_and_productization(self) -> None:
        names = [step.name for step in experiment_delivery_steps(Path("python"), Path("/repo"), Path("/tmp/work"))]

        self.assertIn("experiment manifest helper", names)
        self.assertIn("heatmap productization helper", names)
        self.assertIn("model data readiness helper", names)
        self.assertIn("stage coordinate normalization helper", names)


if __name__ == "__main__":
    unittest.main()
