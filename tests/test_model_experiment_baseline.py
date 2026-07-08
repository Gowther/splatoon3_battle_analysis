from __future__ import annotations

import unittest
from pathlib import Path

from src.model_experiment_baseline import baseline_paths, build_baseline_steps, summarize_steps


class ModelExperimentBaselineTests(unittest.TestCase):
    def test_baseline_steps_include_registry_dataset_readiness_and_manifest(self) -> None:
        steps = build_baseline_steps(python=Path("python"), output_dir=Path("outputs/baseline"))
        names = [step.name for step in steps]

        self.assertIn("model registry", names)
        self.assertIn("model training datasets", names)
        self.assertIn("model data readiness", names)
        self.assertEqual(names[-1], "experiment manifest")

    def test_validation_paths_switch_when_suite_runs(self) -> None:
        existing = baseline_paths(Path("outputs/baseline"), validation_suite_ran=False)
        generated = baseline_paths(Path("outputs/baseline"), validation_suite_ran=True)

        self.assertEqual(existing["evaluation_results_json"], Path("outputs/validation_suite/evaluation/evaluation_results.json"))
        self.assertEqual(generated["evaluation_results_json"], Path("outputs/baseline/validation_suite/evaluation/evaluation_results.json"))

    def test_summarize_steps_marks_failed(self) -> None:
        summary = summarize_steps([{"status": "passed"}, {"status": "failed"}])

        self.assertEqual(summary["status"], "failed")


if __name__ == "__main__":
    unittest.main()
