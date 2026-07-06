from __future__ import annotations

import unittest

from src.model_experiments import build_experiment_plan


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


if __name__ == "__main__":
    unittest.main()
