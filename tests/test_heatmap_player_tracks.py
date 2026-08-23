from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.heatmap.clean_points import TrackState, assign_tracks_for_team
from src.heatmap.infer_player_tracks import clean_route_images, report_rows


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
        "track_slot_hint": "",
        "player_id": "",
    }


def identified_point(x: float, y: float, slot: int, player_id: str) -> dict:
    row = point(x, y)
    row.update({"track_slot_hint": slot, "player_id": player_id, "source": "player_name_template"})
    return row


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

    def test_name_template_hint_binds_observation_to_configured_slot(self) -> None:
        states = {1: TrackState(100, 100, 0), 2: TrackState(500, 500, 0)}
        rows = assign_tracks_for_team(
            [identified_point(505, 500, 2, "blue_player_2")],
            states,
            time_value=1.0,
            tracking_config=TRACKING_CONFIG,
        )
        self.assertEqual(rows[0]["track_slot"], 2)
        self.assertEqual(rows[0]["player_id"], "blue_player_2")
        self.assertEqual(rows[0]["track_status"], "matched")

    def test_large_hinted_step_is_explicit_jump_reset(self) -> None:
        states = {1: TrackState(100, 100, 0)}
        config = {**TRACKING_CONFIG, "max_matched_gap_seconds": 0.5, "max_matched_step_px": 120}
        rows = assign_tracks_for_team(
            [identified_point(250, 100, 1, "yellow_player_1")],
            states,
            time_value=0.2,
            tracking_config=config,
        )
        self.assertEqual(rows[0]["track_status"], "jump_reset")

    def test_identity_report_exposes_gap_and_step_quality_metrics(self) -> None:
        rows = [
            {
                "player_id": "yellow_player_1",
                "track_status": "matched",
                "step_distance": "55",
                "identity_note": "hud_slot_mapping_verified",
            },
            {
                "player_id": "yellow_player_1",
                "track_status": "reacquired",
                "step_distance": "200",
                "identity_note": "hud_slot_mapping_verified",
            },
        ]

        report = {row["metric"]: row["value"] for row in report_rows(rows, [{"time": "2"}], [Path("route.png")])}

        self.assertEqual(report["gap_ratio"], 0.5)
        self.assertEqual(report["large_step_rows"], 1)
        self.assertEqual(report["matched_large_step_rows"], 0)
        self.assertEqual(report["reacquired_large_step_rows"], 1)
        self.assertEqual(report["max_matched_step_px"], 55.0)
        self.assertEqual(report["max_reacquired_step_px"], 200.0)

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
