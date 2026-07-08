from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.annotation_round import (
    annotation_priority_tasks,
    annotation_progress,
    blocking_reason,
    evaluate_progress_gates,
    label_readiness,
    next_actions,
    render_markdown,
    resolve_round,
)


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

    def test_label_readiness_uses_recommended_manual_label_gate(self) -> None:
        readiness = label_readiness({"labeled_rows": 30, "complete_frame_team_groups": 10})

        self.assertEqual(readiness["status"], "ready")
        self.assertTrue(readiness["checks"]["min_labeled_rows"]["ok"])

    def test_blocking_reason_explains_missing_manual_labels(self) -> None:
        reason = blocking_reason("needs_labels", {"labeled_rows": 0}, {})

        self.assertIn("manual x/y labels", reason)

    def test_next_actions_include_parameter_experiments_after_readiness(self) -> None:
        actions = next_actions(
            round_id="r1",
            annotation_csv=Path("outputs/round/annotation_template.csv"),
            package_dir=Path("outputs/round"),
            progress={"labeled_rows": 30},
            priority_tasks=[],
            readiness={"status": "ready"},
        )

        self.assertEqual(actions[-1]["id"], "run_parameter_experiments")

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

    def test_render_markdown_includes_annotation_ui_summary(self) -> None:
        markdown = render_markdown(
            {
                "status": "needs_labels",
                "blocking_reason": "manual labels required",
                "round": {"id": "r1", "matches": [], "frames_per_match": 1},
                "progress": {},
                "progress_checks": {},
                "label_readiness": {"status": "needs_labels", "checks": {}},
                "next_actions": [],
                "priority_tasks": [],
                "annotation_ui": {
                    "status": "ready",
                    "output_html": "outputs/round/annotation_ui.html",
                    "rows": 12,
                    "priority_rows": 4,
                    "priority_limit": 4,
                },
                "quality_loop": {},
            }
        )

        self.assertIn("## Annotation UI", markdown)
        self.assertIn("outputs/round/annotation_ui.html", markdown)
        self.assertIn("priority_rows: 4", markdown)


if __name__ == "__main__":
    unittest.main()
