from __future__ import annotations

import unittest

from src.sample_intake import match_id_from_video, render_sample_intake_report, resolve_match_ids, scan_analysis_windows_command


class SampleIntakeTests(unittest.TestCase):
    def test_match_id_from_video_uses_stem(self) -> None:
        self.assertEqual(match_id_from_video("footages/n_match_6.mp4"), "n_match_6")

    def test_resolve_match_ids_requires_matching_counts(self) -> None:
        with self.assertRaises(ValueError):
            resolve_match_ids(["a.mp4", "b.mp4"], ["a"])

    def test_scan_command_includes_all_match_ids(self) -> None:
        command = scan_analysis_windows_command(
            ".venv/bin/python",
            ["n_match_6", "n_match_7"],
            registry="config/data_registry.json",
            evaluation_config="config/evaluation_matches.json",
            window_seconds=30.0,
            stride_seconds=40.0,
            start_seconds=20.0,
            stop_margin_seconds=20.0,
            sample_fps=2.0,
            selected_sample_fps=5.0,
            device="mps",
            warmup_frames=5,
            force=True,
        )

        self.assertIn("--write-selection", command)
        self.assertEqual(command.count("--match-id"), 2)
        self.assertIn("n_match_7", command)
        self.assertIn("--force", command)

    def test_render_report_includes_write_status(self) -> None:
        report = render_sample_intake_report(
            [
                {
                    "match_id": "n_match_6",
                    "analysis_id": "n_match_6_20_40",
                    "video": "footages/n_match_6.mp4",
                    "video_probe": {"exists": True, "readable": True, "duration_seconds": 200.0},
                    "registry_entry": {"id": "n_match_6"},
                }
            ],
            write_results=[{"registry_status": "added", "evaluation_status": "added"}],
        )

        self.assertIn("n_match_6", report)
        self.assertIn("registry_status: added", report)


if __name__ == "__main__":
    unittest.main()
