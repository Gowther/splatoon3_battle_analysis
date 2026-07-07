from __future__ import annotations

import unittest

from src.validation_suite import validation_analysis_ids, validation_heatmap_ids, validation_ids


class ValidationSuiteTests(unittest.TestCase):
    def test_validation_ids_select_best_normal_and_f_heatmaps(self) -> None:
        registry = {
            "matches": [
                {"id": "n_match_1", "analysis_windows": [{"id": "n_match_1_20_40"}, {"id": "n_match_1_best_100_130"}]},
                {"id": "match_1", "analysis_windows": [{"id": "match_1_10_150"}]},
                {"id": "f_match_1", "heatmap": {"id": "heatmap_f_match_1"}},
                {"id": "match_9", "heatmap": {"id": "heatmap_match9"}},
            ]
        }

        self.assertEqual(validation_analysis_ids(registry), ["n_match_1_best_100_130"])
        self.assertEqual(validation_heatmap_ids(registry), ["heatmap_f_match_1"])
        self.assertEqual(validation_ids(registry), ["n_match_1_best_100_130", "heatmap_f_match_1"])


if __name__ == "__main__":
    unittest.main()
