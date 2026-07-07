from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.heatmap.config_loader import load_config
from src.heatmap.config_template import build_heatmap_config_override, default_invalid_ranges, write_yaml


class HeatmapConfigTemplateTests(unittest.TestCase):
    def test_default_invalid_ranges_cover_intro_and_tail(self) -> None:
        ranges = default_invalid_ranges(20.0, 330.0, 368.6)

        self.assertEqual(ranges, [[0.0, 19.9], [340.0, 368.6]])

    def test_build_heatmap_config_override_uses_registry_defaults(self) -> None:
        config = build_heatmap_config_override(
            {
                "matches": [
                    {
                        "id": "m1",
                        "video": "footages/m1.mp4",
                        "heatmap": {"stop_seconds": 120.0, "sample_fps": 2.0},
                    }
                ]
            },
            "m1",
            duration_seconds=150.0,
        )

        self.assertEqual(config["match"]["input_video"], "footages/m1.mp4")
        self.assertEqual(config["match"]["output_dir"], "outputs/heatmap_m1")
        self.assertEqual(config["sampling"]["stop_seconds"], 120.0)
        self.assertEqual(config["sampling"]["sample_fps"], 2.0)
        self.assertEqual(config["frame_quality"]["invalid_ranges_seconds"], [[0.0, 19.9], [130.0, 150.0]])

    def test_generated_override_loads_with_base_config(self) -> None:
        config = build_heatmap_config_override(
            {"matches": [{"id": "m1", "video": "footages/m1.mp4"}]},
            "m1",
            stop_seconds=80.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_yaml(path, config)

            loaded = load_config(path)

        self.assertEqual(loaded["match"]["id"], "m1")
        self.assertEqual(loaded["match"]["output_dir"], "outputs/heatmap_m1")
        self.assertEqual(loaded["outputs"]["frames_dir"], "outputs/heatmap_m1/frames")
        self.assertEqual(loaded["sampling"]["stop_seconds"], 80.0)


if __name__ == "__main__":
    unittest.main()
