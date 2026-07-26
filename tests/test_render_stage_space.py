from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.heatmap.render_stage_space import (
    DEFAULT_CANVAS_SIZE,
    DEFAULT_MARGIN,
    build_stage_canvas,
    build_stage_heat,
    group_by_team,
    parse_stage_points,
    read_stage_rows,
    render_markdown,
    render_stage_heatmaps,
    render_stage_routes,
    stage_to_pixel,
)


CONFIG = {
    "match": {"id": "match_test"},
    "map_view": {"roi": {"x1": 0, "y1": 180, "x2": 1760, "y2": 980}},
    "rendering": {
        "heat_point_radius_px": 22,
        "heat_blur_sigma_px": 34,
        "heat_max_alpha": 0.58,
        "route_line_thickness_px": 2,
        "route_point_radius_px": 4,
        "route_max_draw_step_px": 120,
    },
    "teams": {"yellow": {}, "blue": {}},
}


def write_stage_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "match_id",
        "time",
        "team",
        "track_slot",
        "confidence",
        "track_status",
        "step_distance",
        "stage_x",
        "stage_y",
        "stage_inside_roi",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def stage_row(**overrides) -> dict:
    row = {
        "match_id": "match_test",
        "time": "20.000",
        "team": "yellow",
        "track_slot": "1",
        "confidence": "0.8",
        "track_status": "matched",
        "step_distance": "10",
        "stage_x": "0.5",
        "stage_y": "0.5",
        "stage_inside_roi": "True",
    }
    row.update(overrides)
    return row


class StageGeometryTests(unittest.TestCase):
    def test_stage_corners_map_to_canvas_corners(self) -> None:
        top_left = stage_to_pixel(0.0, 0.0)
        bottom_right = stage_to_pixel(1.0, 1.0)

        self.assertEqual(top_left, (DEFAULT_MARGIN, DEFAULT_MARGIN))
        self.assertEqual(bottom_right, (DEFAULT_CANVAS_SIZE - DEFAULT_MARGIN, DEFAULT_CANVAS_SIZE - DEFAULT_MARGIN))

    def test_stage_center_maps_to_canvas_center(self) -> None:
        self.assertEqual(stage_to_pixel(0.5, 0.5), (DEFAULT_CANVAS_SIZE // 2, DEFAULT_CANVAS_SIZE // 2))

    def test_canvas_is_square_and_fixed(self) -> None:
        canvas = build_stage_canvas(canvas_size=400)

        self.assertEqual(canvas.shape, (400, 400, 3))

    def test_same_stage_point_lands_identically_for_any_match(self) -> None:
        # This is the property that makes stage renders comparable across matches.
        self.assertEqual(stage_to_pixel(0.3, 0.7), stage_to_pixel(0.3, 0.7))


class ParseStagePointsTests(unittest.TestCase):
    def test_rows_without_stage_columns_are_dropped(self) -> None:
        points = parse_stage_points([{"x": "100", "y": "200"}])

        self.assertEqual(points, [])

    def test_rows_outside_the_stage_box_are_dropped(self) -> None:
        points = parse_stage_points(
            [
                {"stage_x": "0.5", "stage_y": "0.5"},
                {"stage_x": "1.4", "stage_y": "0.5"},
                {"stage_x": "0.5", "stage_y": "-0.2"},
            ]
        )

        self.assertEqual(len(points), 1)

    def test_missing_confidence_defaults_to_one(self) -> None:
        points = parse_stage_points([{"stage_x": "0.5", "stage_y": "0.5"}])

        self.assertEqual(points[0]["confidence"], 1.0)

    def test_track_fields_are_kept_for_route_drawing(self) -> None:
        points = parse_stage_points(
            [{"stage_x": "0.5", "stage_y": "0.5", "track_status": "matched", "step_distance": "12"}]
        )

        self.assertEqual(points[0]["track_status"], "matched")
        self.assertEqual(points[0]["step_distance"], "12")

    def test_group_by_team(self) -> None:
        points = parse_stage_points(
            [
                {"stage_x": "0.1", "stage_y": "0.1", "team": "yellow"},
                {"stage_x": "0.2", "stage_y": "0.2", "team": "blue"},
                {"stage_x": "0.3", "stage_y": "0.3", "team": "yellow"},
            ]
        )

        grouped = group_by_team(points)

        self.assertEqual(sorted(grouped), ["blue", "yellow"])
        self.assertEqual(len(grouped["yellow"]), 2)


class StageHeatTests(unittest.TestCase):
    def test_heat_peaks_near_the_point(self) -> None:
        points = parse_stage_points([{"stage_x": "0.5", "stage_y": "0.5", "confidence": "1.0"}])

        heat = build_stage_heat(points, CONFIG, canvas_size=300, margin=20)

        center = stage_to_pixel(0.5, 0.5, canvas_size=300, margin=20)
        self.assertGreater(heat[center[1], center[0]], 0.5)
        self.assertLess(heat[25, 25], heat[center[1], center[0]])

    def test_empty_points_produce_empty_heat(self) -> None:
        heat = build_stage_heat([], CONFIG, canvas_size=200, margin=10)

        self.assertEqual(float(np.max(heat)), 0.0)

    def test_heat_is_normalized_to_one(self) -> None:
        points = parse_stage_points([{"stage_x": "0.4", "stage_y": "0.6"}])

        heat = build_stage_heat(points, CONFIG, canvas_size=300, margin=20)

        self.assertLessEqual(float(np.max(heat)), 1.0)


class StageRouteTests(unittest.TestCase):
    def rendered_line_pixels(self, rows: list[dict]) -> int:
        canvas = render_stage_routes(parse_stage_points(rows), CONFIG, canvas_size=300, margin=20)
        return int(np.count_nonzero(np.any(canvas > 200, axis=2)))

    def test_far_jumps_are_not_connected(self) -> None:
        # A long step is a track reset, not movement. Connecting it would fill
        # the canvas with lines between unrelated positions.
        near = self.rendered_line_pixels(
            [
                stage_row(time="20.000", stage_x="0.2", stage_y="0.2", step_distance="10"),
                stage_row(time="21.000", stage_x="0.8", stage_y="0.8", step_distance="10"),
            ]
        )
        far = self.rendered_line_pixels(
            [
                stage_row(time="20.000", stage_x="0.2", stage_y="0.2", step_distance="900"),
                stage_row(time="21.000", stage_x="0.8", stage_y="0.8", step_distance="900"),
            ]
        )

        self.assertGreater(near, far)

    def test_unmatched_status_is_not_connected(self) -> None:
        matched = self.rendered_line_pixels(
            [
                stage_row(time="20.000", stage_x="0.2", stage_y="0.2"),
                stage_row(time="21.000", stage_x="0.8", stage_y="0.8"),
            ]
        )
        unmatched = self.rendered_line_pixels(
            [
                stage_row(time="20.000", stage_x="0.2", stage_y="0.2", track_status="new"),
                stage_row(time="21.000", stage_x="0.8", stage_y="0.8", track_status="new"),
            ]
        )

        self.assertGreater(matched, unmatched)


class RenderStageHeatmapsTests(unittest.TestCase):
    def test_missing_csv_reports_no_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = render_stage_heatmaps(Path(tmp) / "absent.csv", CONFIG, Path(tmp) / "out")

        self.assertEqual(report["status"], "no_points")
        self.assertEqual(report["rendered"], {})

    def test_csv_without_stage_columns_reports_no_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "tracks.csv"
            csv_path.write_text("x,y\n100,200\n", encoding="utf-8")

            report = render_stage_heatmaps(csv_path, CONFIG, Path(tmp) / "out")

        self.assertEqual(report["status"], "no_points")

    def test_renders_one_image_per_team_plus_combined_and_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stage.csv"
            write_stage_csv(
                csv_path,
                [
                    stage_row(team="yellow", stage_x="0.3", stage_y="0.3"),
                    stage_row(team="yellow", stage_x="0.4", stage_y="0.4", time="21.000"),
                    stage_row(team="blue", stage_x="0.6", stage_y="0.6", track_slot="2"),
                ],
            )
            output_dir = Path(tmp) / "rendered_stage"

            report = render_stage_heatmaps(csv_path, CONFIG, output_dir, canvas_size=300, margin=20)

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["stage_points"], 3)
            self.assertEqual(report["teams"], ["blue", "yellow"])
            self.assertEqual(sorted(report["rendered"]), ["heatmap_blue", "heatmap_combined", "heatmap_yellow", "routes"])
            for name in ("stage_heatmap_blue.png", "stage_heatmap_yellow.png", "stage_heatmap_combined.png", "stage_routes.png"):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_dropped_rows_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stage.csv"
            write_stage_csv(csv_path, [stage_row(), stage_row(stage_x="1.9")])

            report = render_stage_heatmaps(csv_path, CONFIG, Path(tmp) / "out", canvas_size=200, margin=10)

        self.assertEqual(report["input_rows"], 2)
        self.assertEqual(report["stage_points"], 1)
        self.assertEqual(report["dropped_rows"], 1)

    def test_read_stage_rows_of_missing_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_stage_rows(Path(tmp) / "nope.csv"), [])

    def test_markdown_mentions_images_and_fixed_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stage.csv"
            write_stage_csv(csv_path, [stage_row()])
            report = render_stage_heatmaps(csv_path, CONFIG, Path(tmp) / "out", canvas_size=200, margin=10)

            markdown = render_markdown(report)

        self.assertIn("# Stage Space Rendering", markdown)
        self.assertIn("stage_routes.png", markdown)

    def test_markdown_for_empty_report_explains_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = render_stage_heatmaps(Path(tmp) / "absent.csv", CONFIG, Path(tmp) / "out")

            markdown = render_markdown(report)

        self.assertIn("Promote a control-point asset", markdown)


if __name__ == "__main__":
    unittest.main()
