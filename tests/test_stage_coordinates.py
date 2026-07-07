from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.stage_coordinates import (
    StageBox,
    build_stage_coordinate_report,
    normalize_point,
    normalize_rows,
    write_normalized_csv,
)


class StageCoordinateTests(unittest.TestCase):
    def test_normalize_point_maps_roi_center_to_half(self) -> None:
        box = StageBox(0, 180, 1760, 980)

        point = normalize_point(880, 580, box)

        self.assertAlmostEqual(point["stage_x"], 0.5)
        self.assertAlmostEqual(point["stage_y"], 0.5)
        self.assertTrue(point["inside_roi"])

    def test_normalize_rows_flags_outside_and_invalid_rows(self) -> None:
        rows, summary = normalize_rows(
            [{"x": "5", "y": "5"}, {"x": "-5", "y": "5"}, {"x": "", "y": "5"}],
            StageBox(0, 0, 10, 10),
        )

        self.assertEqual(summary["input_rows"], 3)
        self.assertEqual(summary["normalized_rows"], 2)
        self.assertEqual(summary["outside_roi_rows"], 1)
        self.assertEqual(summary["invalid_rows"], 1)
        self.assertEqual(rows[0]["stage_x"], "0.500000")
        self.assertEqual(rows[1]["stage_inside_roi"], "False")

    def test_build_report_uses_config_roi_and_missing_points(self) -> None:
        report = build_stage_coordinate_report(
            {"map_view": {"coordinate_space": "video_pixels", "roi": {"x1": 0, "y1": 10, "x2": 20, "y2": 30}}},
            points_csv="/tmp/does-not-exist.csv",
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["points"]["status"], "missing")
        self.assertEqual(report["transform"]["source_roi"]["y1"], 10.0)

    def test_write_normalized_csv_adds_stage_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "points.csv"
            output = root / "normalized.csv"
            with source.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "x", "y"])
                writer.writeheader()
                writer.writerow({"time": "1.0", "x": "5", "y": "5"})

            summary = write_normalized_csv(source, output, StageBox(0, 0, 10, 10))
            output_text = output.read_text(encoding="utf-8")

        self.assertEqual(summary["normalized_rows"], 1)
        self.assertIn("stage_x", output_text)


if __name__ == "__main__":
    unittest.main()
