from __future__ import annotations

import unittest

from src.match_intake import build_intake_plan, default_analysis_id, upsert_registry_match


class MatchIntakeTests(unittest.TestCase):
    def test_default_analysis_id_uses_window(self) -> None:
        self.assertEqual(default_analysis_id("match_12", 10.0, 150.0), "match_12_10_150")

    def test_build_intake_plan_for_missing_video_is_non_destructive(self) -> None:
        plan = build_intake_plan(
            "match_test",
            "footages/missing.mp4",
            start_seconds=1.0,
            stop_seconds=2.0,
            sample_fps=5.0,
            device="cpu",
        )
        self.assertEqual(plan["analysis_id"], "match_test_1_2")
        self.assertFalse(plan["video_probe"]["exists"])
        self.assertEqual(plan["registry_entry"]["analysis_windows"][0]["device"], "cpu")

    def test_upsert_registry_match_preserves_existing_heatmap(self) -> None:
        registry = {
            "matches": [
                {
                    "id": "match_11",
                    "video": "footages/match_11.mp4",
                    "purpose": ["heatmap_baseline"],
                    "heatmap": {"id": "heatmap_match11"},
                }
            ]
        }
        status = upsert_registry_match(
            registry,
            {
                "id": "match_11",
                "video": "footages/match_11.mp4",
                "purpose": ["analysis_candidate"],
                "analysis_windows": [{"id": "match_11_20_40"}],
            },
        )
        match = registry["matches"][0]
        self.assertEqual(status, "updated")
        self.assertEqual(match["heatmap"]["id"], "heatmap_match11")
        self.assertEqual(match["purpose"], ["heatmap_baseline", "analysis_candidate"])
        self.assertEqual(match["analysis_windows"][0]["id"], "match_11_20_40")


if __name__ == "__main__":
    unittest.main()
