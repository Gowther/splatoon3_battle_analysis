from __future__ import annotations

import unittest

import numpy as np

from src.heatmap.render_heatmaps import build_heat, color_from_hsv_ranges, team_display_color


class HeatmapRenderingTests(unittest.TestCase):
    def test_heat_scale_percentile_is_configurable(self) -> None:
        points = [
            {"x": 10, "y": 10, "confidence": 1.0},
            {"x": 11, "y": 10, "confidence": 1.0},
            {"x": 30, "y": 30, "confidence": 0.6},
        ]
        mask = np.ones((48, 48), dtype=np.uint8)
        base = {"rendering": {"heat_point_radius_px": 2, "heat_blur_sigma_px": 1.0}}
        high_clip = {"rendering": {**base["rendering"], "heat_scale_percentile": 99.9}}
        low_clip = {"rendering": {**base["rendering"], "heat_scale_percentile": 90.0}}

        high_heat = build_heat(points, (48, 48), mask, high_clip)
        low_heat = build_heat(points, (48, 48), mask, low_clip)

        self.assertLessEqual(float(high_heat.mean()), float(low_heat.mean()))
        self.assertLessEqual(float(high_heat.max()), 1.0)

    def test_team_display_color_derives_unlisted_calibrated_team(self) -> None:
        config = {
            "rendering": {},
            "teams": {
                "mint": {
                    "hsv_ranges": [
                        {"lower": [70, 80, 80], "upper": [92, 255, 255]},
                    ]
                }
            },
        }

        color = team_display_color("mint", config)

        self.assertNotEqual(color, (255, 255, 255))
        self.assertGreater(color[1], color[2])

    def test_team_display_color_uses_explicit_override(self) -> None:
        config = {
            "rendering": {"team_colors_bgr": {"hue_081": [1, 2, 3]}},
            "teams": {"hue_081": {"hsv_ranges": [{"lower": [71, 90, 90], "upper": [91, 255, 255]}]}},
        }

        self.assertEqual(team_display_color("hue_081", config), (1, 2, 3))

    def test_color_from_hsv_ranges_handles_red_wrap_preset(self) -> None:
        color = color_from_hsv_ranges(
            [
                {"lower": [0, 80, 80], "upper": [4, 255, 255]},
                {"lower": [172, 80, 80], "upper": [179, 255, 255]},
            ]
        )

        self.assertGreater(color[2], 200)


if __name__ == "__main__":
    unittest.main()
