from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.core.paths import ROOT
from src.heatmap.stage_coordinates import StageBox
from src.heatmap.stage_reference import build_draft_asset
from src.stage_labeling_workbench import (
    build_stage_labeling_state,
    describe_package,
    normalize_labeled_points,
    promote_stage_labels,
    save_stage_labels,
)


def make_package(package_dir: Path, stage_id: str) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "stage_id": stage_id,
        "config": "src/heatmap/config_match9.yaml",
        "source_roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        "grid_divisions": 10,
        "frames": [
            {"time": 30.0, "status": "exported", "path": "frames/reference_00030.000s.jpg"},
            {"time": 60.0, "status": "unreadable", "path": ""},
        ],
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    draft = build_draft_asset(stage_id, StageBox(0, 0, 100, 100))
    (package_dir / "control_points_draft.json").write_text(json.dumps(draft), encoding="utf-8")


def square_points() -> list[dict]:
    return [
        {"name": "top_left", "source_x": 0, "source_y": 0, "stage_x": 0.0, "stage_y": 0.0},
        {"name": "top_right", "source_x": 100, "source_y": 0, "stage_x": 1.0, "stage_y": 0.0},
        {"name": "bottom_right", "source_x": 100, "source_y": 100, "stage_x": 1.0, "stage_y": 1.0},
        {"name": "bottom_left", "source_x": 0, "source_y": 100, "stage_x": 0.0, "stage_y": 1.0},
    ]


def clustered_points() -> list[dict]:
    # Four non-collinear points packed into the top-left corner of a 100x100
    # ROI. The homography solves cleanly (near-zero reprojection error) but the
    # points cover almost none of the map, so the geometry gates reject them.
    return [
        {"name": "a", "source_x": 0, "source_y": 0, "stage_x": 0.0, "stage_y": 0.0},
        {"name": "b", "source_x": 8, "source_y": 0, "stage_x": 0.08, "stage_y": 0.0},
        {"name": "c", "source_x": 8, "source_y": 8, "stage_x": 0.08, "stage_y": 0.08},
        {"name": "d", "source_x": 0, "source_y": 8, "stage_x": 0.0, "stage_y": 0.08},
    ]


class StageLabelingWorkbenchTests(unittest.TestCase):
    def setUp(self) -> None:
        # safe_project_file only accepts paths inside the project, so fixtures
        # live under outputs/ and are removed afterwards.
        self.base = ROOT / "outputs" / "test_stage_labeling_workbench"
        self.package_dir = self.base / "stage_a"
        make_package(self.package_dir, "stage_a")
        self.promoted = ROOT / "config" / "stage_control_points" / "stage_a.json"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.base, ignore_errors=True)
        if self.promoted.exists():
            self.promoted.unlink()

    def test_describe_package_reads_manifest_and_draft(self) -> None:
        described = describe_package(self.package_dir)

        self.assertEqual(described["stage_id"], "stage_a")
        self.assertTrue(described["draft_template"])
        self.assertEqual(described["control_point_count"], 4)
        self.assertEqual(len(described["frames"]), 1)
        self.assertFalse(described["promoted"])

    def test_describe_package_returns_none_without_manifest(self) -> None:
        empty = self.base / "not_a_package"
        empty.mkdir(parents=True, exist_ok=True)

        self.assertIsNone(describe_package(empty))

    def test_state_lists_packages(self) -> None:
        state = build_stage_labeling_state(self.base)

        self.assertEqual(state["package_count"], 1)
        self.assertEqual(state["packages"][0]["stage_id"], "stage_a")
        self.assertEqual(state["min_control_points"], 4)

    def test_normalize_labeled_points_requires_numeric_fields(self) -> None:
        with self.assertRaises(ValueError):
            normalize_labeled_points([{"name": "a", "source_x": "not-a-number"}])
        with self.assertRaises(ValueError):
            normalize_labeled_points("not-a-list")

    def test_normalize_labeled_points_fills_missing_names(self) -> None:
        points = normalize_labeled_points(
            [{"source_x": 1, "source_y": 2, "stage_x": 0.1, "stage_y": 0.2}]
        )

        self.assertEqual(points[0]["name"], "point_1")
        self.assertEqual(points[0]["source"], [1.0, 2.0])

    def test_save_with_enough_points_clears_template_and_validates(self) -> None:
        result = save_stage_labels(
            {
                "package_dir": str(self.package_dir),
                "stage_id": "stage_a",
                "points": square_points(),
            }
        )

        self.assertTrue(result["saved"])
        self.assertFalse(result["template"])
        self.assertEqual(result["validation"]["status"], "ready")
        saved = json.loads((self.package_dir / "control_points_draft.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["template"])
        self.assertEqual(len(saved["control_points"]), 4)

    def test_save_with_too_few_points_keeps_template(self) -> None:
        result = save_stage_labels(
            {
                "package_dir": str(self.package_dir),
                "points": square_points()[:2],
            }
        )

        self.assertTrue(result["template"])
        self.assertNotEqual(result["validation"]["status"], "ready")

    def test_save_rejects_unknown_package(self) -> None:
        missing = self.base / "missing"
        missing.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(ValueError):
            save_stage_labels({"package_dir": str(missing), "points": square_points()})

    def test_promote_refuses_template_draft(self) -> None:
        result = promote_stage_labels({"package_dir": str(self.package_dir)})

        self.assertFalse(result["promoted"])
        self.assertFalse(self.promoted.exists())

    def test_promote_writes_validated_asset(self) -> None:
        save_stage_labels(
            {"package_dir": str(self.package_dir), "stage_id": "stage_a", "points": square_points()}
        )

        result = promote_stage_labels({"package_dir": str(self.package_dir)})

        self.assertTrue(result["promoted"])
        self.assertTrue(self.promoted.exists())
        promoted = json.loads(self.promoted.read_text(encoding="utf-8"))
        self.assertEqual(promoted["stage_id"], "stage_a")
        self.assertFalse(promoted["template"])
        self.assertIn("report_stage_coordinates", result["next_step"])

    def test_promote_blocked_by_geometry_gate(self) -> None:
        # These four points are non-collinear, so the homography solves and
        # reprojection is near-perfect — the basic validator passes them. But
        # they sit clustered in one corner of the ROI, so the coverage and
        # corner gates reject them. This is the exact case that slipped through
        # the web promote path before the geometry gate was wired in.
        result = save_stage_labels(
            {"package_dir": str(self.package_dir), "stage_id": "stage_a", "points": clustered_points()}
        )
        self.assertEqual(result["validation"]["status"], "ready")
        self.assertEqual(result["quality"]["status"], "needs_review")

        promotion = promote_stage_labels({"package_dir": str(self.package_dir)})

        self.assertFalse(promotion["promoted"])
        self.assertFalse(self.promoted.exists())
        self.assertIn("coverage", promotion["blocked_by"])


if __name__ == "__main__":
    unittest.main()
