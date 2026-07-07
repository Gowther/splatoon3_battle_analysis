from __future__ import annotations

import unittest
from pathlib import Path

from src.heatmap.color_calibration import display_path, preset_for_hue, resolve_preset, team_from_hue


CALIB = {
    "max_preset_distance": 11,
    "dynamic_hue_margin": 10,
    "min_saturation": 90,
    "min_value": 90,
}


class ColorCalibrationTests(unittest.TestCase):
    def test_mint_hues_resolve_to_named_preset(self) -> None:
        self.assertEqual(preset_for_hue(80, 11), "mint")
        self.assertEqual(preset_for_hue(81, 11), "mint")

    def test_existing_neighbor_presets_keep_their_names(self) -> None:
        self.assertEqual(preset_for_hue(67, 11), "green")
        self.assertEqual(preset_for_hue(95, 11), "cyan")

    def test_team_from_hue_uses_mint_preset_before_dynamic_name(self) -> None:
        name, config, source = team_from_hue(81, [], CALIB)

        self.assertEqual(name, "mint")
        self.assertEqual(source, "preset")
        self.assertEqual(config, resolve_preset("mint"))

    def test_display_path_allows_external_runtime_outputs(self) -> None:
        self.assertEqual(display_path(Path("/tmp/splatoon3_runtime.yaml")), "/tmp/splatoon3_runtime.yaml")


if __name__ == "__main__":
    unittest.main()
