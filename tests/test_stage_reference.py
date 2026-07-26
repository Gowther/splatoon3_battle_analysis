from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.heatmap.stage_coordinates import StageBox, parse_control_point
from src.heatmap.stage_reference import (
    build_draft_asset,
    frame_filename,
    grid_lines,
    reference_times,
)


class StageReferenceTests(unittest.TestCase):
    def test_grid_lines_span_roi_with_normalized_labels(self) -> None:
        box = StageBox(0, 180, 1760, 980)

        lines = grid_lines(box, divisions=10)

        self.assertEqual(len(lines["vertical"]), 11)
        self.assertEqual(len(lines["horizontal"]), 11)
        self.assertEqual(lines["vertical"][0]["x"], 0.0)
        self.assertEqual(lines["vertical"][0]["label"], 0.0)
        self.assertEqual(lines["vertical"][-1]["x"], 1760.0)
        self.assertEqual(lines["vertical"][-1]["label"], 1.0)
        self.assertEqual(lines["horizontal"][5]["y"], 580.0)
        self.assertEqual(lines["horizontal"][5]["label"], 0.5)

    def test_grid_lines_rejects_zero_divisions(self) -> None:
        with self.assertRaises(ValueError):
            grid_lines(StageBox(0, 0, 10, 10), divisions=0)

    def test_reference_times_uses_config_reference_and_extra_probes(self) -> None:
        config = {
            "map_view": {"reference_time_seconds": 60.0},
            "sampling": {"start_seconds": 20.0},
        }

        times = reference_times(config, extra_times=[30.0, 90.0, 60.0])

        self.assertEqual(times, [30.0, 60.0, 90.0])

    def test_reference_times_falls_back_to_sampling_start(self) -> None:
        config = {"map_view": {}, "sampling": {"start_seconds": 20.0}}

        self.assertEqual(reference_times(config), [20.0])

    def test_frame_filename_is_sortable(self) -> None:
        self.assertEqual(frame_filename(30.0), "reference_00030.000s.jpg")
        self.assertLess(frame_filename(30.0), frame_filename(90.0))
        self.assertLess(frame_filename(90.0), frame_filename(120.0))

    def test_draft_asset_is_template_with_roi_corner_seeds(self) -> None:
        draft = build_draft_asset("match9_stage", StageBox(0, 180, 1760, 980))

        self.assertTrue(draft["template"])
        self.assertEqual(draft["stage_id"], "match9_stage")
        self.assertEqual(len(draft["control_points"]), 4)
        parsed = [parse_control_point(point) for point in draft["control_points"]]
        self.assertEqual(parsed[0]["source_x"], 0.0)
        self.assertEqual(parsed[0]["stage_x"], 0.0)
        self.assertEqual(parsed[2]["source_x"], 1760.0)
        self.assertEqual(parsed[2]["stage_y"], 1.0)

    def test_draft_asset_round_trips_through_json(self) -> None:
        draft = build_draft_asset("stage", StageBox(0, 0, 100, 100))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draft.json"
            path.write_text(json.dumps(draft), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["control_points"], draft["control_points"])
        self.assertTrue(loaded["template"])


if __name__ == "__main__":
    unittest.main()
