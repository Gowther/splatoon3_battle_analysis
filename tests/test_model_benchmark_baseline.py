from __future__ import annotations

import unittest

from src.model_benchmark_baseline import build_baseline_snapshot, render_markdown


class ModelBenchmarkBaselineTests(unittest.TestCase):
    def test_build_baseline_snapshot_collects_counts(self) -> None:
        report = build_baseline_snapshot(
            evaluation_results=[{"status": "passed"}, {"status": "failed"}],
            model_errors={"status": "needs_review", "issue_counts": {"count_ocr": 2}},
            heatmap_comparison={"status": "passed", "matches": [{}], "aggregate": {"anomaly_counts": {"jump_reset": 1}}},
            heatmap_quality_loop={"status": "needs_labels", "metrics": {"labeled_rows": 0}},
            benchmark_plan={"status": "ready", "summary": {"run_count": 1}},
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["evaluation"]["status_counts"], {"passed": 1, "failed": 1})
        self.assertEqual(report["model_errors"]["issue_counts"], {"count_ocr": 2})

    def test_render_markdown_includes_status(self) -> None:
        markdown = render_markdown(
            build_baseline_snapshot(
                evaluation_results=None,
                model_errors=None,
                heatmap_comparison=None,
                heatmap_quality_loop=None,
                benchmark_plan=None,
            )
        )

        self.assertIn("`needs_inputs`", markdown)


if __name__ == "__main__":
    unittest.main()
