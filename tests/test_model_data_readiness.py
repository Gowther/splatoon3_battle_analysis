from __future__ import annotations

import unittest

from src.model_data_readiness import build_model_data_readiness_report, render_markdown


class ModelDataReadinessTests(unittest.TestCase):
    def test_missing_labels_block_model_data_experiments(self) -> None:
        report = build_model_data_readiness_report(
            annotation_round={"status": "needs_labels", "progress": {"labeled_rows": 0}},
            parameter_experiments={"status": "needs_labels"},
            validation_suite={"status": "passed"},
            runtime_benchmarks={"status": "ready"},
            dataset_governance={"status": "passed"},
            model_experiment_plan={"status": "planned", "summary": {"high_priority": 1}},
        )

        self.assertEqual(report["status"], "needs_data")
        self.assertTrue(any(item["area"] == "heatmap_labels" for item in report["blockers"]))

    def test_ready_when_labels_and_baselines_are_present(self) -> None:
        report = build_model_data_readiness_report(
            annotation_round={"status": "ready_for_evaluation", "progress": {"labeled_rows": 30}},
            parameter_experiments={"status": "planned"},
            validation_suite={"status": "passed"},
            runtime_benchmarks={"status": "ready"},
            dataset_governance={"status": "passed"},
            model_experiment_plan={"status": "planned", "summary": {}},
        )

        self.assertEqual(report["status"], "ready_for_model_data_experiments")
        self.assertEqual(report["blockers"], [])

    def test_render_markdown_lists_blockers(self) -> None:
        markdown = render_markdown(build_model_data_readiness_report(annotation_round={"progress": {"labeled_rows": 0}}))

        self.assertIn("Blockers", markdown)
        self.assertIn("heatmap_labels", markdown)


if __name__ == "__main__":
    unittest.main()
