from __future__ import annotations

import unittest
from pathlib import Path

from src.project_check_registry import experiment_delivery_steps, heatmap_annotation_steps


class ProjectCheckRegistryTests(unittest.TestCase):
    def test_heatmap_annotation_steps_are_named_and_ordered(self) -> None:
        steps = heatmap_annotation_steps(Path("python"), Path("/tmp/work"))

        self.assertEqual(steps[0].name, "heatmap annotation round helper")
        self.assertEqual(steps[-1].name, "heatmap parameter experiment helper")

    def test_experiment_delivery_steps_include_manifest_and_productization(self) -> None:
        names = [step.name for step in experiment_delivery_steps(Path("python"), Path("/repo"), Path("/tmp/work"))]

        self.assertIn("experiment manifest helper", names)
        self.assertIn("heatmap productization helper", names)
        self.assertIn("stage coordinate normalization helper", names)


if __name__ == "__main__":
    unittest.main()
