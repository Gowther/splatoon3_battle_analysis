from __future__ import annotations

import unittest

from src.count_smoothing import CountSmoothingConfig, smooth_rows


class CountSmoothingTests(unittest.TestCase):
    def test_smooths_leading_single_frame_noise(self) -> None:
        rows = [
            {"elapsed_time": "1.0", "count_left": "2"},
            {"elapsed_time": "1.2", "count_left": "61"},
            {"elapsed_time": "1.4", "count_left": "62"},
        ]

        smoothed, corrections = smooth_rows(rows, config=CountSmoothingConfig(max_jump=20, neighbor_tolerance=3))

        self.assertEqual(smoothed[0]["count_left"], "62")
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0].reason, "dropped_digit_before_stable_neighbor")

    def test_keeps_real_leading_change_without_stable_neighbor(self) -> None:
        rows = [
            {"elapsed_time": "1.0", "count_left": "2"},
            {"elapsed_time": "1.2", "count_left": "61"},
            {"elapsed_time": "1.4", "count_left": "35"},
        ]

        smoothed, corrections = smooth_rows(rows, config=CountSmoothingConfig(max_jump=20, neighbor_tolerance=3))

        self.assertEqual(smoothed[0]["count_left"], "2")
        self.assertEqual(corrections, [])

    def test_smooths_dropped_tens_digit_sequence(self) -> None:
        rows = [
            {"elapsed_time": "100.0", "count_left": "6"},
            {"elapsed_time": "100.2", "count_left": "66"},
            {"elapsed_time": "100.4", "count_left": "5"},
            {"elapsed_time": "100.6", "count_left": "64"},
            {"elapsed_time": "100.8", "count_left": "53"},
            {"elapsed_time": "101.0", "count_left": "22"},
            {"elapsed_time": "101.2", "count_left": "2"},
            {"elapsed_time": "101.4", "count_left": "61"},
        ]

        smoothed, corrections = smooth_rows(rows, config=CountSmoothingConfig(max_jump=20, neighbor_tolerance=3))

        self.assertEqual([row["count_left"] for row in smoothed], ["66", "66", "65", "64", "53", "52", "52", "61"])
        self.assertTrue(any(correction.reason.startswith("dropped_digit") for correction in corrections))

    def test_smooths_high_spike_between_stable_neighbors(self) -> None:
        rows = [
            {"elapsed_time": "1.0", "count_left": "67"},
            {"elapsed_time": "1.2", "count_left": "87"},
            {"elapsed_time": "1.4", "count_left": "66"},
        ]

        smoothed, corrections = smooth_rows(rows, config=CountSmoothingConfig(max_jump=20, neighbor_tolerance=3))

        self.assertEqual([row["count_left"] for row in smoothed], ["67", "67", "66"])
        self.assertEqual(corrections[0].reason, "short_noise_run_between_stable_neighbors")


if __name__ == "__main__":
    unittest.main()
