from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.heatmap.clean_points import TrackState, assign_tracks_for_team
from src.heatmap.infer_player_tracks import clean_route_images


TRACKING_CONFIG = {
    "max_track_step_px": 420,
    "max_track_speed_px_per_second": 100,
    "max_track_gap_seconds": 2.0,
    "prediction_horizon_seconds": 1.0,
    "min_assignment_gate_px": 10,
    "assignment_confidence_weight_px": 0,
    "velocity_smoothing_alpha": 0.5,
}


def point(x: float, y: float, *, confidence: float = 0.9) -> dict:
    return {
        "match_id": "match",
        "time": "1.000",
        "frame_index": "60",
        "team": "yellow",
        "x": x,
        "y": y,
        "confidence": confidence,
        "frame_path": "frame.jpg",
    }


class HeatmapPlayerTracksTests(unittest.TestCase):
    def test_global_assignment_keeps_both_slots_when_local_nearest_conflicts(self) -> None:
        states = {
            1: TrackState(0, 0, 0),
            2: TrackState(10, 0, 0),
        }

        rows = assign_tracks_for_team(
            [point(6, 0), point(14, 0)],
            states,
            time_value=1.0,
            tracking_config=TRACKING_CONFIG,
        )

        self.assertEqual([(row["track_slot"], row["x"]) for row in rows], [(1, 6), (2, 14)])
        self.assertTrue(all(row["track_status"] == "matched" for row in rows))

    def test_empty_slot_is_started_instead_of_forcing_a_large_jump(self) -> None:
        states = {1: TrackState(0, 0, 0), 2: None}

        rows = assign_tracks_for_team(
            [point(500, 500)],
            states,
            time_value=3.0,
            tracking_config=TRACKING_CONFIG,
        )

        self.assertEqual(rows[0]["track_slot"], 2)
        self.assertEqual(rows[0]["track_status"], "new")
        self.assertLess(rows[0]["tracking_confidence"], rows[0]["confidence"])

    def test_stale_slot_is_reacquired_without_claiming_a_match(self) -> None:
        states = {1: TrackState(0, 0, 0)}

        rows = assign_tracks_for_team(
            [point(500, 500)],
            states,
            time_value=3.0,
            tracking_config=TRACKING_CONFIG,
        )

        self.assertEqual(rows[0]["track_slot"], 1)
        self.assertEqual(rows[0]["track_status"], "reacquired")
        self.assertEqual(rows[0]["time_delta"], 3.0)

    def test_prediction_fields_are_emitted_for_matched_track(self) -> None:
        states = {1: TrackState(10, 10, 0, vx=5, vy=0, observations=3)}

        rows = assign_tracks_for_team(
            [point(15, 10)],
            states,
            time_value=1.0,
            tracking_config=TRACKING_CONFIG,
        )

        self.assertEqual(rows[0]["prediction_error"], 0.0)
        self.assertEqual(rows[0]["time_delta"], 1.0)
        self.assertEqual(rows[0]["observation_count"], 4)
        self.assertGreater(rows[0]["tracking_confidence"], 0.8)

    def test_clean_route_images_removes_only_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale_png = output_dir / "old_route.png"
            keep_txt = output_dir / "README.txt"
            stale_png.write_bytes(b"png")
            keep_txt.write_text("keep", encoding="utf-8")

            removed = clean_route_images(output_dir)

            self.assertEqual(removed, 1)
            self.assertFalse(stale_png.exists())
            self.assertTrue(keep_txt.exists())


if __name__ == "__main__":
    unittest.main()
