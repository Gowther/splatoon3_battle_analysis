from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.quality_loop import build_quality_loop_report, render_markdown


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class HeatmapQualityLoopTests(unittest.TestCase):
    def test_evaluates_manual_annotation_against_registered_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "player_tracks.csv"
            annotations = root / "annotation_template.csv"
            write_csv(
                tracks,
                [
                    "match_id",
                    "time",
                    "frame_index",
                    "team",
                    "player_id",
                    "x",
                    "y",
                    "confidence",
                    "track_status",
                ],
                [
                    {
                        "match_id": "f_match_test",
                        "time": "10.0",
                        "frame_index": "100",
                        "team": "blue",
                        "player_id": "blue_slot_1",
                        "x": "100",
                        "y": "200",
                        "confidence": "0.9",
                        "track_status": "matched",
                    }
                ],
            )
            write_csv(
                annotations,
                [
                    "match_id",
                    "time",
                    "frame_index",
                    "team",
                    "annotation_id",
                    "x",
                    "y",
                    "visibility",
                    "frame_complete",
                ],
                [
                    {
                        "match_id": "f_match_test",
                        "time": "10.0",
                        "frame_index": "100",
                        "team": "blue",
                        "annotation_id": "label_1",
                        "x": "103",
                        "y": "204",
                        "visibility": "visible",
                        "frame_complete": "true",
                    }
                ],
            )
            registry = {
                "matches": [
                    {
                        "id": "f_match_test",
                        "heatmap": {
                            "id": "heatmap_f_match_test",
                            "player_tracks": str(tracks),
                        },
                    }
                ]
            }

            report = build_quality_loop_report(
                registry,
                annotation_csv=annotations,
                threshold_px=10.0,
                min_recall=1.0,
                max_mean_error_px=10.0,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["metrics"]["matched_labels"], 1)
        self.assertEqual(report["metrics"]["recall"], 1.0)

    def test_missing_annotations_need_labels(self) -> None:
        report = build_quality_loop_report({"matches": []}, annotation_csv=Path("/tmp/does-not-exist.csv"))

        self.assertEqual(report["status"], "needs_labels")

    def test_render_markdown_includes_quality_loop_status(self) -> None:
        markdown = render_markdown({"status": "needs_labels", "metrics": {}, "checks": {}})

        self.assertIn("# Heatmap Quality Loop", markdown)
        self.assertIn("`needs_labels`", markdown)


if __name__ == "__main__":
    unittest.main()
