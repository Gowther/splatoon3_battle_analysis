from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis_pipeline import REQUIRED_DETECTION_CLASSES, analyze_results
from src.analysis_preview import PreviewSaveState, preview_frame_name
from src.analysis_runtime import CSV_HEADER, preview_dir_from_arg, write_analysis_csv
from src.analysis_warmup import WeaponWarmupState, record_weapon_vote
from src.run_analysis import CSV_HEADER as RUN_ANALYSIS_CSV_HEADER
from src.run_analysis import REQUIRED_DETECTION_CLASSES as RUN_ANALYSIS_REQUIRED_DETECTION_CLASSES


class RunAnalysisRefactorTests(unittest.TestCase):
    def test_run_analysis_reexports_csv_header_for_compatibility(self) -> None:
        self.assertEqual(RUN_ANALYSIS_CSV_HEADER, CSV_HEADER)

    def test_run_analysis_reexports_detection_contract_for_compatibility(self) -> None:
        self.assertEqual(RUN_ANALYSIS_REQUIRED_DETECTION_CLASSES, REQUIRED_DETECTION_CLASSES)

    def test_write_analysis_csv_can_include_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.csv"
            write_analysis_csv(path, [["1.0", *[""] * 32]], include_header=True)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0], CSV_HEADER)
        self.assertEqual(rows[1][0], "1.0")

    def test_write_analysis_csv_can_omit_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.csv"
            write_analysis_csv(path, [["1.0", *[""] * 32]], include_header=False)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "1.0")

    def test_analyze_results_populates_values_by_csv_field_name(self) -> None:
        ids = {
            "asari_object": 10,
            "hoko_canmon": 11,
            "area_object": 12,
            "yagura_kanmon": 13,
            "player": 14,
            "message": 15,
        }
        class_counts = {10: 1, 11: 2, 12: 3, 13: 4, 14: 1}
        lamps = [[0, 0, 0, 0, 0, state] for state in range(1, 9)]

        with (
            patch("src.analysis_pipeline.detections", return_value=[]),
            patch("src.analysis_pipeline.player_lamps", return_value=lamps),
            patch(
                "src.analysis_pipeline.by_class",
                side_effect=lambda _rows, class_id: [object()] * class_counts.get(class_id, 0),
            ),
            patch("src.analysis_pipeline.count_numbers", return_value=(90, 80)),
            patch("src.analysis_pipeline.penalty_numbers", return_value=(5, 7)),
            patch("src.analysis_pipeline.first_image_for_class", return_value=None),
        ):
            row = analyze_results(
                object(),
                1.25,
                "2026-09-01T12:00:00",
                ids,
                object(),
                object(),
                [f"weapon_{index}" for index in range(1, 9)],
                0.5,
                0.5,
                0.5,
                0.5,
            )

        values = dict(zip(CSV_HEADER, row))
        self.assertEqual(len(row), len(CSV_HEADER))
        self.assertEqual(values["player_state_8"], 8)
        self.assertEqual(values["weapon_8"], "weapon_8")
        self.assertEqual(values["count_left"], 90)
        self.assertEqual(values["penalty_right"], 7)
        self.assertEqual(values["yagura_count"], 4)
        self.assertTrue(values["player_detected"])
        self.assertEqual(values["timestamp"], "2026-09-01T12:00:00")

    def test_preview_dir_from_arg_creates_absolute_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "previews"

            result = preview_dir_from_arg(str(path))

            self.assertEqual(result, path)
            self.assertTrue(path.exists())

    def test_preview_save_state_tracks_limit(self) -> None:
        state = PreviewSaveState(Path("/tmp/previews"), limit=2, saved=2)

        self.assertTrue(state.enabled)
        self.assertFalse(state.can_save)
        self.assertEqual(preview_frame_name(12, 3.46), "frame_00012_3.5s.jpg")

    def test_weapon_warmup_state_votes_until_complete(self) -> None:
        state = WeaponWarmupState()
        vote = ["a", "b", "c", "d", "e", "f", "g", "h"]

        record_weapon_vote(state, vote, warmup_frames=2)
        self.assertIsNone(state.final_weapons)
        record_weapon_vote(state, vote, warmup_frames=2)

        self.assertEqual(state.final_weapons, vote)


if __name__ == "__main__":
    unittest.main()
