from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from src.heatmap.detect_markers import (
    PlayerTemplate,
    PlayerTemplateState,
    _template_search_box,
    match_marker_near,
    player_template_is_alive,
    reference_template_points,
    state_row_at_or_before,
)


def template() -> PlayerTemplate:
    marker = np.array(
        [
            [0, 255, 0],
            [255, 255, 255],
            [0, 255, 0],
        ],
        dtype=np.uint8,
    )
    return PlayerTemplate(
        player_id="yellow_player_1",
        team="yellow",
        track_slot=1,
        reference_box=(40, 40, 50, 46),
        reference_marker=(45, 50),
        marker_offset=(5, 10),
        label_edges=np.ones((6, 10), dtype=np.uint8),
        marker_edges=marker,
    )


class HeatmapDetectMarkerTests(unittest.TestCase):
    def test_match9_template_slots_match_result_screen_order(self) -> None:
        config = yaml.safe_load(Path("src/heatmap/config_match9.yaml").read_text(encoding="utf-8"))
        slots = {item["player_id"]: item["track_slot"] for item in config["marker_detection"]["player_templates"]}

        self.assertEqual(
            {key: slots[key] for key in ("yellow_player_1", "yellow_player_2", "yellow_player_3", "yellow_player_4")},
            {"yellow_player_1": 2, "yellow_player_2": 1, "yellow_player_3": 3, "yellow_player_4": 4},
        )
        self.assertEqual(
            {key: slots[key] for key in ("blue_player_1", "blue_player_2", "blue_player_3", "blue_player_4")},
            {"blue_player_1": 4, "blue_player_2": 3, "blue_player_3": 2, "blue_player_4": 1},
        )

    def test_marker_template_finds_shift_within_local_window(self) -> None:
        edges = np.zeros((80, 100), dtype=np.uint8)
        edges[34:37, 61:64] = template().marker_edges

        score, point = match_marker_near(edges, template().marker_edges, (60, 35), search_radius=5)

        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(point, (62, 35))

    def test_backward_tracking_uses_absolute_time_delta(self) -> None:
        config = {
            "map_view": {"roi": {"x1": 0, "y1": 0, "x2": 200, "y2": 120}},
            "marker_detection": {
                "template_max_gap_seconds": 2.0,
                "template_min_search_gate_px": 10,
                "template_max_speed_px_per_second": 20,
            },
        }

        search_box, tracked, gate = _template_search_box(
            template(), PlayerTemplateState(100, 60, 60.0), 59.0, (120, 200), config
        )

        self.assertTrue(tracked)
        self.assertEqual(gate, 20.0)
        self.assertNotEqual(search_box, (0, 0, 200, 120))

    def test_reference_seed_emits_stable_identity_and_coordinate(self) -> None:
        rows = reference_template_points(
            [template()],
            {"time": "60.000", "frame_index": "3600", "frame_path": "frame.jpg"},
            {"match": {"id": "match"}},
        )

        self.assertEqual(rows[0]["player_id"], "yellow_player_1")
        self.assertEqual(rows[0]["track_slot_hint"], 1)
        self.assertEqual((rows[0]["x"], rows[0]["y"]), (45.0, 50.0))
        self.assertEqual(rows[0]["source"], "reference_marker_seed")

    def test_dead_hud_slot_suppresses_that_player_template(self) -> None:
        config = {
            "teams": {"yellow": {}, "blue": {}},
            "death_events": {"dead_state_ids": [3]},
        }
        state = {"player_state_1": "3", "player_state_2": "0"}

        self.assertFalse(player_template_is_alive(template(), state, config))
        self.assertTrue(player_template_is_alive(template(), {"player_state_1": "0"}, config))

    def test_state_lookup_does_not_leak_future_death_into_previous_frame(self) -> None:
        rows = [
            {"elapsed_time": "49", "player_state_3": "0"},
            {"elapsed_time": "50", "player_state_3": "3"},
        ]

        self.assertEqual(state_row_at_or_before(rows, 49.8)["elapsed_time"], "49")
        self.assertEqual(state_row_at_or_before(rows, 50.0)["elapsed_time"], "50")
        self.assertIsNone(state_row_at_or_before(rows, 48.0))


if __name__ == "__main__":
    unittest.main()
