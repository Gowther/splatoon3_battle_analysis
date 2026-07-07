from __future__ import annotations

import unittest

from src.validation_sample_report import build_report, render_markdown


class ValidationSampleReportTests(unittest.TestCase):
    def test_build_report_pairs_registry_and_evaluation_results(self) -> None:
        registry = {
            "matches": [
                {
                    "id": "n_match_1",
                    "video": "footages/n_match_1.mp4",
                    "analysis_windows": [{"id": "n_match_1_best_10_40", "start_seconds": 10.0, "stop_seconds": 40.0}],
                },
                {
                    "id": "f_match_1",
                    "video": "footages/f_match_1.mp4",
                    "heatmap": {"teams": ["blue", "yellow"]},
                },
            ]
        }
        evaluation = [
            {
                "id": "n_match_1_best_10_40",
                "status": "passed",
                "smoothed_csv": "outputs/evaluation/n_match_1_best_10_40/smoothed.csv",
                "smoothed_metrics": {"rows": 10, "count_rows": 10},
            }
        ]
        heatmap_comparison = {"status": "passed", "matches": [{"match_id": "f_match_1", "status": "passed", "metrics": {"track_rows": 100}, "anomalies": {"total": 3}}]}
        model_error = {"status": "passed", "files": [{"file": "outputs/evaluation/n_match_1_best_10_40/smoothed.csv", "status": "passed", "issues": []}]}

        report = build_report(registry, evaluation, heatmap_comparison, {}, model_error)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["normal_samples"][0]["evaluation_status"], "passed")
        self.assertEqual(report["heatmap_samples"][0]["metrics"]["track_rows"], 100)

    def test_build_report_includes_heatmap_quality_loop(self) -> None:
        report = build_report(
            {"matches": []},
            [],
            {"status": "passed", "matches": []},
            {},
            {"status": "passed", "files": []},
            {
                "status": "passed",
                "metrics": {
                    "labeled_rows": 8,
                    "matched_labels": 7,
                    "recall": 0.875,
                    "precision_on_complete_groups": 0.9,
                    "mean_error_px": 18.5,
                },
            },
        )
        markdown = render_markdown(report)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["heatmap_quality_loop_status"], "passed")
        self.assertIn("## Heatmap Quality Loop", markdown)
        self.assertIn("0.875", markdown)


if __name__ == "__main__":
    unittest.main()
