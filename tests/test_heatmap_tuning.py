from __future__ import annotations

import unittest

from src.heatmap.tuning import recommendations_from_metrics


class HeatmapTuningTests(unittest.TestCase):
    def test_missing_labels_recommends_labeling_first(self) -> None:
        recommendations = recommendations_from_metrics({"status": "no_labels"}, {})

        self.assertEqual(recommendations[0]["area"], "labels")
        self.assertEqual(recommendations[0]["priority"], "high")

    def test_low_recall_recommends_marker_probe(self) -> None:
        recommendations = recommendations_from_metrics({"status": "evaluated", "recall": 0.5, "missed_labels": 3}, {})

        self.assertTrue(any(item["area"] == "marker_detection" for item in recommendations))


if __name__ == "__main__":
    unittest.main()
