from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.model_error_report import PLAYER_STATE_FIELDS, WEAPON_FIELDS
from src.training_sample_export import (
    analysis_windows_by_id,
    build_training_sample_package,
    count_jump_events,
    render_markdown,
)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["elapsed_time", *PLAYER_STATE_FIELDS, *WEAPON_FIELDS, "count_left", "count_right", "message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TrainingSampleExportTests(unittest.TestCase):
    def test_count_jump_events_identifies_large_changes(self) -> None:
        events = count_jump_events(
            [
                {"elapsed_time": "1.0", "count_left": "80"},
                {"elapsed_time": "1.2", "count_left": "8"},
            ]
        )

        self.assertEqual(events[0]["field"], "count_left")
        self.assertEqual(events[0]["previous_value"], 80)
        self.assertEqual(events[0]["value"], 8)

    def test_analysis_windows_by_id_maps_video_context(self) -> None:
        windows = analysis_windows_by_id(
            {
                "matches": [
                    {
                        "id": "m1",
                        "video": "footages/m1.mp4",
                        "analysis_windows": [{"id": "m1_best", "start_seconds": 1.0}],
                    }
                ]
            }
        )

        self.assertEqual(windows["m1_best"]["match_id"], "m1")
        self.assertEqual(windows["m1_best"]["video"], "footages/m1.mp4")

    def test_build_training_sample_package_writes_candidate_queues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_csv = root / "raw.csv"
            smoothed_csv = root / "smoothed.csv"
            complete_states = {field: "1" for field in PLAYER_STATE_FIELDS}
            complete_weapons = {field: "w" for field in WEAPON_FIELDS}
            write_rows(
                raw_csv,
                [
                    {"elapsed_time": "1.0", **complete_states, **complete_weapons, "count_left": "80"},
                    {"elapsed_time": "1.2", **complete_states, **complete_weapons, "count_left": "8"},
                    {"elapsed_time": "1.4", **complete_states, **complete_weapons, "message": "go"},
                ],
            )
            write_rows(
                smoothed_csv,
                [
                    {"elapsed_time": "1.0", **complete_states, **complete_weapons, "count_left": "80"},
                    {"elapsed_time": "1.2", "count_left": "80"},
                    {"elapsed_time": "1.4", **complete_states, **complete_weapons, "message": "go"},
                ],
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "matches": [
                            {
                                "id": "m1",
                                "video": str(root / "video.mp4"),
                                "analysis_windows": [{"id": "m1_best"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            evaluation = root / "evaluation.json"
            evaluation.write_text(
                json.dumps(
                    [
                        {
                            "id": "m1_best",
                            "kind": "analysis",
                            "raw_csv": str(raw_csv),
                            "smoothed_csv": str(smoothed_csv),
                        }
                    ]
                ),
                encoding="utf-8",
            )

            report = build_training_sample_package(
                registry_path=registry,
                evaluation_results_path=evaluation,
                output_dir=root / "package",
                include_heatmap=False,
                export_frames=False,
            )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["target_rows"]["ui_detector_yolo"], 1)
        self.assertEqual(report["target_rows"]["count_ocr_yolo"], 1)
        self.assertEqual(report["target_rows"]["message_ocr_yolo"], 1)

    def test_render_markdown_includes_next_steps(self) -> None:
        markdown = render_markdown(
            {
                "status": "ready",
                "output_dir": "outputs/package",
                "evaluation_results": "outputs/evaluation.json",
                "target_rows": {"ui_detector_yolo": 2},
                "training_status": {"ui_detector_yolo": {"status": "ready"}},
                "analysis": {"targets": {"ui_detector_yolo": {"csv": "queue.csv", "rows": 2}}},
                "next_steps": [
                    {
                        "id": "label_ui_detector_candidates",
                        "reason": "label",
                        "command": "review queue.csv",
                    }
                ],
            }
        )

        self.assertIn("# Training Sample Candidates", markdown)
        self.assertIn("label_ui_detector_candidates", markdown)
        self.assertIn("queue.csv", markdown)


if __name__ == "__main__":
    unittest.main()
