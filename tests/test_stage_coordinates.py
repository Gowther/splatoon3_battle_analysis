from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.stage_coordinates import (
    StageBox,
    build_stage_coordinate_report,
    control_point_summary,
    control_points_from_config,
    coordinate_schema,
    homography_from_control_points,
    load_control_point_asset,
    merge_control_point_asset,
    normalize_point,
    normalize_point_homography,
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

    def test_homography_control_points_map_center_to_half(self) -> None:
        points = [
            {"source": [0, 0], "target": [0, 0]},
            {"source": [10, 0], "target": [1, 0]},
            {"source": [10, 10], "target": [1, 1]},
            {"source": [0, 10], "target": [0, 1]},
        ]
        matrix = homography_from_control_points(control_points_from_config({"map_view": {"control_points": points}}))

        point = normalize_point_homography(5, 5, matrix)

        self.assertAlmostEqual(point["stage_x"], 0.5)
        self.assertAlmostEqual(point["stage_y"], 0.5)
        self.assertTrue(point["inside_roi"])

    def test_build_report_uses_homography_when_control_points_exist(self) -> None:
        report = build_stage_coordinate_report(
            {
                "map_view": {
                    "coordinate_space": "video_pixels",
                    "roi": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                    "control_points": [
                        {"source": [0, 0], "target": [0, 0]},
                        {"source": [10, 0], "target": [1, 0]},
                        {"source": [10, 10], "target": [1, 1]},
                        {"source": [0, 10], "target": [0, 1]},
                    ],
                }
            }
        )

        self.assertEqual(report["transform"]["method"], "homography")
        self.assertEqual(report["transform"]["homography_status"], "ready")
        self.assertEqual(report["output_schema"]["columns"][0]["name"], "stage_x")

    def test_control_point_asset_can_be_merged_into_config(self) -> None:
        asset = {
            "path": "config/stage_control_points/test.json",
            "stage_id": "test_stage",
            "control_points": [
                {"source_x": 0.0, "source_y": 0.0, "stage_x": 0.0, "stage_y": 0.0},
                {"source_x": 10.0, "source_y": 0.0, "stage_x": 1.0, "stage_y": 0.0},
                {"source_x": 10.0, "source_y": 10.0, "stage_x": 1.0, "stage_y": 1.0},
                {"source_x": 0.0, "source_y": 10.0, "stage_x": 0.0, "stage_y": 1.0},
            ],
        }

        merged = merge_control_point_asset({"map_view": {"roi": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}}, asset)

        self.assertEqual(merged["stage_coordinates"]["stage_id"], "test_stage")
        self.assertEqual(len(control_points_from_config(merged)), 4)

    def test_control_point_summary_flags_missing_points(self) -> None:
        summary = control_point_summary([], StageBox(0, 0, 10, 10))

        self.assertEqual(summary["status"], "needs_control_points")
        self.assertEqual(summary["missing_count"], 4)

    def test_load_control_point_asset_normalizes_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control_points.json"
            path.write_text(
                '{"stage_id":"test","control_points":[{"source":[0,0],"target":[0,0]}]}',
                encoding="utf-8",
            )

            asset = load_control_point_asset(path)

        self.assertEqual(asset["stage_id"], "test")
        self.assertEqual(asset["control_points"][0]["source_x"], 0.0)

    def test_template_control_point_asset_does_not_enable_homography(self) -> None:
        report = build_stage_coordinate_report(
            {"map_view": {"roi": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}},
            control_point_asset={
                "template": True,
                "control_points": [
                    {"source_x": 0.0, "source_y": 0.0, "stage_x": 0.0, "stage_y": 0.0},
                    {"source_x": 10.0, "source_y": 0.0, "stage_x": 1.0, "stage_y": 0.0},
                    {"source_x": 10.0, "source_y": 10.0, "stage_x": 1.0, "stage_y": 1.0},
                    {"source_x": 0.0, "source_y": 10.0, "stage_x": 0.0, "stage_y": 1.0},
                ],
            },
        )

        self.assertEqual(report["transform"]["method"], "roi_linear_normalization")
        self.assertEqual(report["transform"]["homography_status"], "template_only")

    def test_coordinate_schema_documents_stage_columns(self) -> None:
        schema = coordinate_schema("homography")

        self.assertEqual([column["name"] for column in schema["columns"]], ["stage_x", "stage_y", "stage_inside_roi"])


if __name__ == "__main__":
    unittest.main()
