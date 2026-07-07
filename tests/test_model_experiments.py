from __future__ import annotations

import unittest

from src.model_experiments import build_benchmark_plan, build_experiment_plan, render_benchmark_markdown


class ModelExperimentTests(unittest.TestCase):
    def test_build_experiment_plan_prioritizes_triggered_categories(self) -> None:
        config = {
            "experiments": [
                {
                    "id": "count_probe",
                    "area": "count_ocr",
                    "candidate": "thresholds",
                    "status": "planned",
                    "trigger_categories": ["count_ocr"],
                },
                {
                    "id": "message_probe",
                    "area": "message_ocr",
                    "candidate": "ocr",
                    "status": "planned",
                    "trigger_categories": ["message_ocr"],
                },
            ]
        }
        plan = build_experiment_plan(config, model_errors={"issue_counts": {"count_ocr": 23}})
        priorities = {experiment["id"]: experiment["priority"] for experiment in plan["experiments"]}
        self.assertEqual(priorities["count_probe"], "high")
        self.assertEqual(priorities["message_probe"], "baseline")

    def test_build_benchmark_plan_defaults_to_triggered_experiments(self) -> None:
        experiment_plan = {
            "experiments": [
                {
                    "id": "count_probe",
                    "area": "count_ocr",
                    "candidate": "thresholds",
                    "priority": "high",
                    "metrics": ["count_jump_warnings"],
                    "pass_criteria": ["no extra jumps"],
                    "baseline_commands": ["python baseline.py"],
                },
                {
                    "id": "message_probe",
                    "area": "message_ocr",
                    "candidate": "ocr",
                    "priority": "baseline",
                },
            ]
        }

        benchmark = build_benchmark_plan(
            experiment_plan,
            evaluation_results=[{"status": "passed"}, {"status": "failed"}],
            validation_ids=["n_match_1_best_10_40"],
        )

        self.assertEqual(benchmark["status"], "ready")
        self.assertEqual([run["id"] for run in benchmark["runs"]], ["count_probe"])
        self.assertEqual(benchmark["baseline_result_status_counts"], {"passed": 1, "failed": 1})
        self.assertEqual(benchmark["runs"][0]["result_template"]["status"], "not_run")

    def test_render_benchmark_markdown_includes_commands(self) -> None:
        benchmark = build_benchmark_plan(
            {
                "experiments": [
                    {
                        "id": "heatmap_probe",
                        "area": "heatmap",
                        "candidate": "tracker",
                        "priority": "high",
                    }
                ]
            },
            validation_ids=["heatmap_f_match_1"],
        )
        markdown = render_benchmark_markdown(benchmark)

        self.assertIn("# Model Benchmark Plan", markdown)
        self.assertIn("report_heatmap_quality_loop.py", markdown)


if __name__ == "__main__":
    unittest.main()
