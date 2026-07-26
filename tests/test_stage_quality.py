from __future__ import annotations

import unittest

from src.heatmap.stage_coordinates import StageBox, homography_from_control_points, parse_control_point
from src.heatmap.stage_quality import (
    build_control_point_quality_report,
    convex_hull,
    corner_sanity_report,
    coverage_report,
    frame_drift_report,
    polygon_area,
    render_quality_markdown,
)


MATCH_CONFIG = {"map_view": {"roi": {"x1": 0, "y1": 180, "x2": 1760, "y2": 980}}}

SPREAD_POINTS = [
    {"name": "tl", "source": [0, 180], "target": [0.0, 0.0]},
    {"name": "tr", "source": [1760, 180], "target": [1.0, 0.0]},
    {"name": "br", "source": [1760, 980], "target": [1.0, 1.0]},
    {"name": "bl", "source": [0, 980], "target": [0.0, 1.0]},
]

# Four points inside one small corner of the ROI. They reproject perfectly but
# the fitted homography sends the rest of the ROI far outside the stage box.
CLUSTERED_POINTS = [
    {"name": "a", "source": [100, 200], "target": [0.05, 0.05]},
    {"name": "b", "source": [260, 205], "target": [0.15, 0.05]},
    {"name": "c", "source": [265, 290], "target": [0.15, 0.16]},
    {"name": "d", "source": [105, 285], "target": [0.05, 0.15]},
]


def asset(points: list[dict], *, stage_id: str = "stage", template: bool = False) -> dict:
    return {"stage_id": stage_id, "template": template, "control_points": points}


class GeometryTests(unittest.TestCase):
    def test_convex_hull_of_square_keeps_four_corners(self) -> None:
        hull = convex_hull([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])

        self.assertEqual(len(hull), 4)
        self.assertNotIn((5, 5), hull)

    def test_polygon_area_of_unit_square(self) -> None:
        self.assertAlmostEqual(polygon_area([(0, 0), (2, 0), (2, 3), (0, 3)]), 6.0)

    def test_polygon_area_is_zero_for_degenerate_input(self) -> None:
        self.assertEqual(polygon_area([(0, 0), (1, 1)]), 0.0)


class CoverageTests(unittest.TestCase):
    def test_spread_points_cover_the_roi(self) -> None:
        points = [parse_control_point(point) for point in SPREAD_POINTS]

        report = coverage_report(points, StageBox(0, 180, 1760, 980))

        self.assertEqual(report["status"], "ready")
        self.assertAlmostEqual(report["coverage"], 1.0, places=6)

    def test_clustered_points_report_low_coverage(self) -> None:
        points = [parse_control_point(point) for point in CLUSTERED_POINTS]

        report = coverage_report(points, StageBox(0, 180, 1760, 980))

        self.assertEqual(report["status"], "low_coverage")
        self.assertLess(report["coverage"], 0.05)


class CornerSanityTests(unittest.TestCase):
    def test_spread_points_map_roi_corners_into_the_stage_box(self) -> None:
        matrix = homography_from_control_points([parse_control_point(point) for point in SPREAD_POINTS])

        report = corner_sanity_report(matrix, StageBox(0, 180, 1760, 980))

        self.assertEqual(report["status"], "ready")
        self.assertLess(report["max_excursion"], 1e-6)
        self.assertEqual(len(report["corners"]), 4)

    def test_clustered_points_send_corners_outside_the_stage_box(self) -> None:
        matrix = homography_from_control_points([parse_control_point(point) for point in CLUSTERED_POINTS])

        report = corner_sanity_report(matrix, StageBox(0, 180, 1760, 980))

        self.assertEqual(report["status"], "corners_out_of_stage")
        self.assertGreater(report["max_excursion"], 1.0)


class FrameDriftTests(unittest.TestCase):
    def test_no_shared_landmarks_is_not_available(self) -> None:
        report = frame_drift_report({"f1": [{"name": "a", "source": [1, 2], "target": [0, 0]}]})

        self.assertEqual(report["status"], "not_available")
        self.assertEqual(report["compared_landmarks"], 0)

    def test_stable_landmarks_pass(self) -> None:
        report = frame_drift_report(
            {
                "f30": [{"name": "tl", "source": [0, 180], "target": [0, 0]}],
                "f60": [{"name": "tl", "source": [3, 182], "target": [0, 0]}],
            }
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["compared_landmarks"], 1)
        self.assertLess(report["max_drift"], 12.0)

    def test_moved_camera_is_flagged_unstable(self) -> None:
        report = frame_drift_report(
            {
                "f30": [{"name": "tl", "source": [0, 180], "target": [0, 0]}],
                "f60": [{"name": "tl", "source": [85, 240], "target": [0, 0]}],
            }
        )

        self.assertEqual(report["status"], "unstable")
        self.assertGreater(report["max_drift"], 12.0)
        self.assertEqual(report["landmarks"][0]["name"], "tl")


class QualityReportTests(unittest.TestCase):
    def test_spread_asset_is_ready(self) -> None:
        report = build_control_point_quality_report(MATCH_CONFIG, asset(SPREAD_POINTS))

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["failed_checks"], [])
        self.assertEqual(report["checks"]["reprojection"]["status"], "ready")

    def test_clustered_asset_fails_geometry_but_not_reprojection(self) -> None:
        report = build_control_point_quality_report(MATCH_CONFIG, asset(CLUSTERED_POINTS))

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["checks"]["reprojection"]["status"], "ready")
        self.assertIn("coverage", report["failed_checks"])
        self.assertIn("corners", report["failed_checks"])

    def test_template_asset_is_blocked(self) -> None:
        report = build_control_point_quality_report(MATCH_CONFIG, asset(SPREAD_POINTS, template=True))

        self.assertEqual(report["status"], "needs_control_points")
        self.assertTrue(report["blockers"])

    def test_too_few_points_is_blocked(self) -> None:
        report = build_control_point_quality_report(MATCH_CONFIG, asset(SPREAD_POINTS[:3]))

        self.assertEqual(report["status"], "needs_control_points")
        self.assertTrue(any("four control points" in item for item in report["blockers"]))

    def test_drift_failure_marks_report_needs_review(self) -> None:
        report = build_control_point_quality_report(
            MATCH_CONFIG,
            asset(SPREAD_POINTS),
            labeled_frames={
                "f30": [{"name": "tl", "source": [0, 180], "target": [0, 0]}],
                "f60": [{"name": "tl", "source": [85, 240], "target": [0, 0]}],
            },
        )

        self.assertEqual(report["status"], "needs_review")
        self.assertIn("frame_drift", report["failed_checks"])

    def test_markdown_renders_all_checks(self) -> None:
        report = build_control_point_quality_report(MATCH_CONFIG, asset(CLUSTERED_POINTS))

        markdown = render_quality_markdown(report)

        self.assertIn("# Stage Control Point Quality", markdown)
        self.assertIn("coverage", markdown)
        self.assertIn("ROI Corners In Stage Space", markdown)

    def test_markdown_handles_blocked_report_without_numeric_checks(self) -> None:
        report = build_control_point_quality_report(MATCH_CONFIG, asset(SPREAD_POINTS, template=True))

        markdown = render_quality_markdown(report)

        self.assertIn("n/a", markdown)
        self.assertIn("Blockers", markdown)


if __name__ == "__main__":
    unittest.main()
