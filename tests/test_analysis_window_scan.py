from __future__ import annotations

import unittest

from src.analysis_window_scan import analysis_metrics, candidate_windows, count_jump_warnings, rank_candidates


class AnalysisWindowScanTests(unittest.TestCase):
    def test_candidate_windows_respect_duration_margin(self) -> None:
        windows = candidate_windows(130.0, start_seconds=20.0, window_seconds=30.0, stride_seconds=40.0, stop_margin_seconds=20.0)

        self.assertEqual(
            windows,
            [
                {"start_seconds": 20.0, "stop_seconds": 50.0},
                {"start_seconds": 60.0, "stop_seconds": 90.0},
            ],
        )

    def test_analysis_metrics_counts_populated_columns(self) -> None:
        row = {f"player_state_{i}": "alive" for i in range(1, 9)}
        row.update(
            {
                "weapon_1": "Splattershot",
                "count_left": "42",
                "count_right": "",
                "penalty_left": "",
                "penalty_right": "",
                "asari_count": "0",
                "hoko_count": "0",
                "area_count": "1",
                "yagura_count": "0",
                "player_detected": "True",
                "message": "",
            }
        )

        metrics = analysis_metrics([row])

        self.assertEqual(metrics["rows"], 1)
        self.assertEqual(metrics["state_ratio"], 1.0)
        self.assertEqual(metrics["weapon_ratio"], 1.0)
        self.assertEqual(metrics["count_ratio"], 1.0)
        self.assertEqual(metrics["objective_ratio"], 1.0)
        self.assertEqual(metrics["player_ratio"], 1.0)

    def test_count_jump_warnings_detect_large_changes(self) -> None:
        rows = [
            {"elapsed_time": "1.0", "count_left": "90"},
            {"elapsed_time": "1.2", "count_left": "8"},
            {"elapsed_time": "1.4", "count_left": "89"},
        ]

        warnings = count_jump_warnings(rows)

        self.assertEqual(len(warnings), 2)

    def test_rank_candidates_prefers_count_coverage(self) -> None:
        ranked = rank_candidates(
            [
                {"id": "state_only", "start_seconds": 20.0, "metrics": {"rows": 10, "state_ratio": 1.0, "weapon_ratio": 1.0, "objective_ratio": 1.0, "player_ratio": 1.0, "count_ratio": 0.0}},
                {"id": "with_count", "start_seconds": 60.0, "metrics": {"rows": 10, "state_ratio": 0.8, "weapon_ratio": 1.0, "objective_ratio": 1.0, "player_ratio": 0.8, "count_ratio": 0.8}},
            ]
        )

        self.assertEqual(ranked[0]["id"], "with_count")


if __name__ == "__main__":
    unittest.main()
