from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.annotation_round import annotation_priority_tasks, annotation_progress, evaluate_progress_gates, resolve_round


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "match_id",
        "time",
        "frame_index",
        "team",
        "annotation_id",
        "x",
        "y",
        "visibility",
        "frame_complete",
        "source_confidence",
        "source_track_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class HeatmapAnnotationRoundTests(unittest.TestCase):
    def test_annotation_progress_counts_labeled_and_complete_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.csv"
            write_rows(
                path,
                [
                    {"match_id": "a", "time": "1", "frame_index": "10", "team": "blue", "x": "10", "y": "20", "visibility": "visible", "frame_complete": "true"},
                    {"match_id": "a", "time": "1", "frame_index": "10", "team": "blue", "x": "", "y": "", "visibility": "occluded", "frame_complete": "true"},
                ],
            )

            progress = annotation_progress(path)

        self.assertEqual(progress["status"], "ready_for_evaluation")
        self.assertEqual(progress["labeled_rows"], 1)
        self.assertEqual(progress["skipped_rows"], 1)
        self.assertEqual(progress["complete_frame_team_groups"], 1)

    def test_resolve_round_supports_all_configured_matches(self) -> None:
        selected = resolve_round({"heatmap_matches": ["m1"], "defaults": {"frames_per_match": 3}}, "all")

        self.assertEqual(selected["matches"], ["m1"])
        self.assertEqual(selected["frames_per_match"], 3)

    def test_evaluate_progress_gates_flags_missing_labels(self) -> None:
        checks = evaluate_progress_gates(
            {"labeled_rows": 1, "complete_frame_team_groups": 0},
            min_labeled_rows=2,
            min_complete_groups=1,
        )

        self.assertFalse(checks["min_labeled_rows"]["ok"])
        self.assertFalse(checks["min_complete_groups"]["ok"])

    def test_annotation_priority_tasks_prefers_jump_resets_and_one_per_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.csv"
            write_rows(
                path,
                [
                    {
                        "match_id": "a",
                        "time": "1",
                        "frame_index": "10",
                        "team": "blue",
                        "annotation_id": "matched",
                        "x": "",
                        "y": "",
                        "visibility": "visible",
                        "frame_complete": "false",
                        "source_confidence": "0.4",
                        "source_track_status": "matched",
                    },
                    {
                        "match_id": "a",
                        "time": "1",
                        "frame_index": "10",
                        "team": "blue",
                        "annotation_id": "jump",
                        "x": "",
                        "y": "",
                        "visibility": "visible",
                        "frame_complete": "false",
                        "source_confidence": "0.8",
                        "source_track_status": "jump_reset",
                    },
                    {
                        "match_id": "a",
                        "time": "2",
                        "frame_index": "20",
                        "team": "orange",
                        "annotation_id": "new",
                        "x": "",
                        "y": "",
                        "visibility": "visible",
                        "frame_complete": "false",
                        "source_confidence": "0.7",
                        "source_track_status": "new",
                    },
                ],
            )

            tasks = annotation_priority_tasks(path)

        self.assertEqual([task["annotation_id"] for task in tasks], ["jump", "new"])


if __name__ == "__main__":
    unittest.main()
