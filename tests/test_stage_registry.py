from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.heatmap.stage_coordinates import StageBox
from src.heatmap.stage_registry import (
    build_stage_registry_report,
    compare_control_points,
    load_stage_registry,
    register_match,
    render_registry_markdown,
    resolve_stage_asset,
    sample_grid,
    stage_entry,
    stage_for_match,
    write_stage_registry,
)


ROI = StageBox(0, 180, 1760, 980)
CONFIG = {"map_view": {"roi": {"x1": 0, "y1": 180, "x2": 1760, "y2": 980}}}

CORNERS = [
    {"name": "tl", "source": [0, 180], "target": [0.0, 0.0]},
    {"name": "tr", "source": [1760, 180], "target": [1.0, 0.0]},
    {"name": "br", "source": [1760, 980], "target": [1.0, 1.0]},
    {"name": "bl", "source": [0, 980], "target": [0.0, 1.0]},
]

# Same landmarks, a few pixels off. This is ordinary hand-labeling wobble.
NOISY = [
    {"name": "tl", "source": [4, 183], "target": [0.0, 0.0]},
    {"name": "tr", "source": [1756, 178], "target": [1.0, 0.0]},
    {"name": "br", "source": [1763, 977], "target": [1.0, 1.0]},
    {"name": "bl", "source": [-2, 984], "target": [0.0, 1.0]},
]

# One landmark assigned the wrong stage target. This passes every single-asset
# check in stage_quality, so only cross-validation can catch it.
MISLABELED = [
    {"name": "tl", "source": [0, 180], "target": [0.0, 0.0]},
    {"name": "tr", "source": [1760, 180], "target": [1.0, 0.0]},
    {"name": "br", "source": [1760, 980], "target": [0.75, 1.0]},
    {"name": "bl", "source": [0, 980], "target": [0.0, 1.0]},
]


def write_asset(path: Path, stage_id: str, points: list[dict], *, template: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stage_id": stage_id, "template": template, "control_points": points}),
        encoding="utf-8",
    )


class RegistryStorageTests(unittest.TestCase):
    def test_missing_registry_loads_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = load_stage_registry(Path(tmp) / "absent.json")

        self.assertEqual(registry["stages"], [])

    def test_register_creates_and_extends_a_stage(self) -> None:
        registry = register_match({"stages": []}, "gorge", "match_9")
        registry = register_match(registry, "gorge", "match_10")

        entry = stage_entry(registry, "gorge")
        self.assertEqual(entry["matches"], ["match_10", "match_9"])
        self.assertEqual(len(registry["stages"]), 1)

    def test_register_is_idempotent(self) -> None:
        registry = register_match({"stages": []}, "gorge", "match_9")
        registry = register_match(registry, "gorge", "match_9")

        self.assertEqual(stage_entry(registry, "gorge")["matches"], ["match_9"])

    def test_registering_elsewhere_moves_the_match(self) -> None:
        registry = register_match({"stages": []}, "gorge", "match_9")
        registry = register_match(registry, "mall", "match_9")

        self.assertEqual(stage_entry(registry, "gorge")["matches"], [])
        self.assertEqual(stage_entry(registry, "mall")["matches"], ["match_9"])
        self.assertEqual(stage_for_match(registry, "match_9")["stage_id"], "mall")

    def test_registry_round_trips_through_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage_registry.json"
            write_stage_registry(register_match({"stages": []}, "gorge", "match_9"), path)

            self.assertEqual(load_stage_registry(path)["stages"][0]["stage_id"], "gorge")


class ResolveStageAssetTests(unittest.TestCase):
    def test_match_inherits_the_stage_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp)
            write_asset(asset_dir / "gorge.json", "gorge", CORNERS)
            registry = register_match(register_match({"stages": []}, "gorge", "match_9"), "gorge", "match_10")

            asset = resolve_stage_asset(registry, "match_10", asset_dir=asset_dir)

        self.assertIsNotNone(asset)
        self.assertEqual(asset["inherited_from_stage"], "gorge")
        self.assertEqual(asset["shared_with_matches"], ["match_9"])

    def test_unregistered_match_inherits_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = register_match({"stages": []}, "gorge", "match_9")

            self.assertIsNone(resolve_stage_asset(registry, "match_99", asset_dir=Path(tmp)))

    def test_template_asset_is_not_inherited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp)
            write_asset(asset_dir / "gorge.json", "gorge", CORNERS, template=True)
            registry = register_match({"stages": []}, "gorge", "match_9")

            self.assertIsNone(resolve_stage_asset(registry, "match_9", asset_dir=asset_dir))


class CompareControlPointsTests(unittest.TestCase):
    def test_sample_grid_spans_the_roi(self) -> None:
        grid = sample_grid(ROI, divisions=3)

        self.assertEqual(len(grid), 9)
        self.assertIn((0.0, 180.0), grid)
        self.assertIn((1760.0, 980.0), grid)

    def test_identical_labelings_agree_exactly(self) -> None:
        result = compare_control_points(CORNERS, CORNERS, ROI)

        self.assertEqual(result["status"], "ready")
        self.assertLess(result["max_disagreement"], 1e-9)

    def test_noisy_labeling_of_the_same_stage_agrees(self) -> None:
        result = compare_control_points(CORNERS, NOISY, ROI)

        self.assertEqual(result["status"], "ready")
        self.assertLess(result["max_disagreement"], 0.05)

    def test_mislabeled_landmark_is_caught(self) -> None:
        result = compare_control_points(CORNERS, MISLABELED, ROI)

        self.assertEqual(result["status"], "disagrees")
        self.assertGreater(result["max_disagreement"], 0.05)
        self.assertEqual(len(result["worst_source_point"]), 2)

    def test_unsolvable_comparison_reports_invalid(self) -> None:
        result = compare_control_points(CORNERS, CORNERS[:2], ROI)

        self.assertEqual(result["status"], "invalid")


class StageRegistryReportTests(unittest.TestCase):
    def build(self, tmp: str, second_points: list[dict]) -> dict:
        asset_dir = Path(tmp)
        write_asset(asset_dir / "gorge.json", "gorge", CORNERS)
        write_asset(asset_dir / "match_9.json", "gorge", CORNERS)
        write_asset(asset_dir / "match_10.json", "gorge", second_points)
        registry = register_match(register_match({"stages": []}, "gorge", "match_9"), "gorge", "match_10")
        return build_stage_registry_report(
            registry,
            {"match_9": CONFIG, "match_10": CONFIG},
            asset_dir=asset_dir,
        )

    def test_empty_registry_reports_unregistered_matches(self) -> None:
        report = build_stage_registry_report({"stages": []}, {"match_9": CONFIG})

        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["unregistered_matches"], ["match_9"])

    def test_consistent_labelings_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.build(tmp, NOISY)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["failing_stages"], [])
        self.assertEqual(report["stages"][0]["match_count"], 2)

    def test_mislabeled_match_fails_the_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.build(tmp, MISLABELED)

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["failing_stages"], ["gorge"])
        self.assertEqual(report["stages"][0]["disagreeing_matches"], ["match_10"])

    def test_stage_without_asset_needs_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = register_match({"stages": []}, "gorge", "match_9")

            report = build_stage_registry_report(registry, {"match_9": CONFIG}, asset_dir=Path(tmp))

        self.assertEqual(report["stages"][0]["status"], "needs_asset")
        self.assertFalse(report["stages"][0]["has_asset"])

    def test_markdown_reports_cross_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            markdown = render_registry_markdown(self.build(tmp, MISLABELED))

        self.assertIn("# Stage Registry", markdown)
        self.assertIn("Cross Validation", markdown)
        self.assertIn("match_10", markdown)


if __name__ == "__main__":
    unittest.main()
