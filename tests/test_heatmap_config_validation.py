from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.heatmap.config_validation import (
    config_paths_from_registry,
    validate_heatmap_config,
    validate_heatmap_configs,
)


class HeatmapConfigValidationTests(unittest.TestCase):
    def test_registry_config_paths_are_unique(self) -> None:
        registry = {
            "matches": [
                {"id": "a", "heatmap": {"config": "src/heatmap/config_f_match_1.yaml"}},
                {"id": "b", "heatmap": {"config": "src/heatmap/config_f_match_1.yaml"}},
                {"id": "c", "heatmap": {"config": "src/heatmap/config_f_match_2.yaml"}},
            ]
        }

        paths = config_paths_from_registry(registry)

        self.assertEqual([path.name for path in paths], ["config_f_match_1.yaml", "config_f_match_2.yaml"])

    def test_real_f_match_config_passes(self) -> None:
        report = validate_heatmap_config(Path("src/heatmap/config_f_match_1.yaml"))

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["match_id"], "f_match_1")
        self.assertEqual(report["output_dir"], "outputs/heatmap_f_match_1")

    def test_unresolved_placeholder_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "bad.yaml"
            config.write_text(
                "\n".join(
                    [
                        "match:",
                        "  id: bad",
                        "  input_video: footages/bad.mp4",
                        "video: {}",
                        "sampling: {}",
                        "map_view: {}",
                        "frame_quality: {}",
                        "teams: {}",
                        "marker_detection: {}",
                        "point_cleaning: {}",
                        "rendering: {}",
                        "state_join: {}",
                        "outputs:",
                        "  frames_dir: \"{missing}/frames\"",
                        "  valid_frames_csv: valid.csv",
                        "  clean_points_csv: points.csv",
                        "  tracks_csv: tracks.csv",
                        "  rendered_dir: rendered",
                        "  player_tracks_csv: players.csv",
                        "  report_md: report.md",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_heatmap_configs([config])

        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["configs"][0]["problems"])


if __name__ == "__main__":
    unittest.main()
