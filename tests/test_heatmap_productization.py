from __future__ import annotations

import unittest

from src.heatmap.productization import build_productization_report, render_markdown


class HeatmapProductizationTests(unittest.TestCase):
    def test_productization_is_blocked_without_labels(self) -> None:
        report = build_productization_report(annotation_round={"progress": {"labeled_rows": 0}})

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["recommended_next_action"], "Fill manual heatmap labels.")

    def test_productization_moves_to_review_with_labels_and_anomalies(self) -> None:
        report = build_productization_report(
            annotation_round={"progress": {"labeled_rows": 5}},
            heatmap_comparison={"aggregate": {"anomaly_counts": {"jump_reset": 2}}},
        )

        self.assertEqual(report["status"], "needs_review")

    def test_render_markdown_includes_milestones(self) -> None:
        markdown = render_markdown(build_productization_report(annotation_round={"progress": {"labeled_rows": 0}}))

        self.assertIn("Milestones", markdown)
        self.assertIn("label_gate", markdown)


if __name__ == "__main__":
    unittest.main()
