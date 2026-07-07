from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.heatmap.config_loader import deep_merge, load_config


class HeatmapConfigLoaderTests(unittest.TestCase):
    def test_deep_merge_keeps_nested_defaults(self) -> None:
        merged = deep_merge(
            {"rendering": {"heat_max_alpha": 0.58, "heat_low_cutoff": 0.1}},
            {"rendering": {"heat_low_cutoff": 0.2}},
        )

        self.assertEqual(merged["rendering"]["heat_max_alpha"], 0.58)
        self.assertEqual(merged["rendering"]["heat_low_cutoff"], 0.2)

    def test_load_config_resolves_base_and_output_placeholders(self) -> None:
        config = load_config("src/heatmap/config_f_match_1.yaml")

        self.assertEqual(config["match"]["id"], "f_match_1")
        self.assertEqual(config["match"]["output_dir"], "outputs/heatmap_f_match_1")
        self.assertEqual(config["outputs"]["frames_dir"], "outputs/heatmap_f_match_1/frames")
        self.assertEqual(config["state_join"]["state_csv"], "outputs/heatmap_f_match_1/ui_state.csv")
        self.assertEqual(config["rendering"]["heat_scale_percentile"], 99.85)
        self.assertEqual(config["frame_quality"]["invalid_ranges_seconds"][1], [340.0, 368.6])

    def test_load_config_supports_nested_base_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.yaml"
            child = root / "child.yaml"
            base.write_text(
                "\n".join(
                    [
                        "match:",
                        "  output_dir: outputs/heatmap_{match_id}",
                        "outputs:",
                        "  frames_dir: \"{output_dir}/frames\"",
                        "rendering:",
                        "  heat_max_alpha: 0.58",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            child.write_text(
                "\n".join(
                    [
                        f"base_config: {base}",
                        "match:",
                        "  id: sample",
                        "rendering:",
                        "  heat_max_alpha: 0.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(child)

        self.assertEqual(config["match"]["output_dir"], "outputs/heatmap_sample")
        self.assertEqual(config["outputs"]["frames_dir"], "outputs/heatmap_sample/frames")
        self.assertEqual(config["rendering"]["heat_max_alpha"], 0.4)


if __name__ == "__main__":
    unittest.main()
