from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.heatmap.render_stage_space import parse_stage_points
from src.heatmap.stage_aggregate import (
    aggregate_heat,
    build_stage_aggregate,
    collect_stage_matches,
    load_match_points,
    render_difference,
    render_markdown,
    render_side_by_side,
)
from src.heatmap.stage_registry import register_match
from src.heatmap.stage_artifacts import stage_metadata_path, write_stage_metadata


CONFIG = {
    "match": {"id": "match_a"},
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

FIELDS = ["match_id", "time", "team", "track_slot", "confidence", "track_status", "step_distance", "stage_x", "stage_y"]


def write_stage_csv(path: Path, points: list[tuple[float, float, str]], match_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (stage_x, stage_y, team) in enumerate(points):
            writer.writerow(
                {
                    "match_id": "m",
                    "time": f"{20 + index}.000",
                    "team": team,
                    "track_slot": "1",
                    "confidence": "0.9",
                    "track_status": "matched",
                    "step_distance": "10",
                    "stage_x": f"{stage_x}",
                    "stage_y": f"{stage_y}",
                }
            )
    write_stage_metadata(path, {
        "match_id": match_id or path.stem,
        "stage_id": "gorge",
        "status": "ready",
        "quality": "calibrated",
        "method": "homography",
        "quality_gate": {"status": "ready"},
    })


def match_payload(match_id: str, points: list[tuple[float, float, str]]) -> dict:
    rows = [
        {"stage_x": str(x), "stage_y": str(y), "team": team, "track_slot": "1", "time": "20.000"}
        for x, y, team in points
    ]
    return {"match_id": match_id, "points": parse_stage_points(rows), "input": "", "input_rows": len(rows), "status": "ready"}


class LoadAndCollectTests(unittest.TestCase):
    def test_missing_csv_reports_no_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_match_points(Path(tmp) / "absent.csv")

        self.assertEqual(loaded["status"], "no_points")
        self.assertEqual(loaded["points"], [])

    def test_unknown_stage_is_reported(self) -> None:
        collected = collect_stage_matches("nope", {"stages": []}, {})

        self.assertEqual(collected["status"], "unknown_stage")

    def test_matches_without_stage_csv_are_listed_missing(self) -> None:
        registry = register_match({"stages": []}, "gorge", "match_a")

        collected = collect_stage_matches("gorge", registry, {})

        self.assertEqual(collected["status"], "no_data")
        self.assertEqual(collected["missing"], ["match_a"])

    def test_registered_matches_with_data_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_a = Path(tmp) / "a.csv"
            write_stage_csv(csv_a, [(0.3, 0.3, "yellow")], match_id="match_a")
            registry = register_match(register_match({"stages": []}, "gorge", "match_a"), "gorge", "match_b")

            collected = collect_stage_matches("gorge", registry, {"match_a": csv_a})

        self.assertEqual(collected["status"], "ready")
        self.assertEqual([item["match_id"] for item in collected["matches"]], ["match_a"])
        self.assertEqual(collected["missing"], ["match_b"])

    def test_uncalibrated_or_stale_coordinates_are_not_comparable(self) -> None:
        import json

        for issue in ("missing", "provisional", "wrong_stage", "wrong_match", "failed_gate", "changed_csv"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "match_a.csv"
                write_stage_csv(path, [(0.3, 0.3, "yellow")])
                metadata_path = stage_metadata_path(path)
                metadata = json.loads(metadata_path.read_text())
                if issue == "missing":
                    metadata_path.unlink()
                elif issue == "changed_csv":
                    with path.open("a") as handle:
                        handle.write("\n")
                else:
                    if issue == "provisional":
                        metadata["quality"] = "provisional"
                    elif issue == "wrong_stage":
                        metadata["stage_id"] = "another_stage"
                    elif issue == "wrong_match":
                        metadata["match_id"] = "another_match"
                    else:
                        metadata["quality_gate"]["status"] = "needs_review"
                    metadata_path.write_text(json.dumps(metadata))
                registry = register_match({"stages": []}, "gorge", "match_a")
                report = build_stage_aggregate("gorge", registry, {"match_a": path}, CONFIG, Path(tmp) / "out")
                self.assertEqual(report["status"], "needs_calibration")
                self.assertIn("match_a", report["rejected_matches"])
                self.assertEqual(report["rendered"], {})
                self.assertFalse((Path(tmp) / "out").exists())


class AggregateHeatTests(unittest.TestCase):
    def test_empty_matches_produce_empty_heat(self) -> None:
        heat = aggregate_heat([], CONFIG, canvas_size=200, margin=10)

        self.assertEqual(float(np.max(heat)), 0.0)

    def test_team_filter_selects_only_that_team(self) -> None:
        match = match_payload("m", [(0.2, 0.2, "yellow"), (0.8, 0.8, "blue")])

        yellow = aggregate_heat([match], CONFIG, team="yellow", canvas_size=300, margin=20)

        self.assertGreater(float(np.max(yellow[:150, :150])), float(np.max(yellow[200:, 200:])))

    def test_averaging_keeps_a_long_match_from_dominating(self) -> None:
        # One match with many points in a spot should not outweigh another match
        # purely by row count; per-match fields are normalized before averaging.
        heavy = match_payload("heavy", [(0.3, 0.3, "yellow")] * 40)
        light = match_payload("light", [(0.7, 0.7, "yellow")])

        combined = aggregate_heat([heavy, light], CONFIG, canvas_size=300, margin=20)

        left = float(np.max(combined[:150, :150]))
        right = float(np.max(combined[150:, 150:]))
        self.assertGreater(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertLess(left / max(right, 1e-6), 5.0)


class RenderTests(unittest.TestCase):
    def test_side_by_side_width_scales_with_match_count(self) -> None:
        one = render_side_by_side([match_payload("a", [(0.5, 0.5, "yellow")])], CONFIG, canvas_size=200, margin=10)
        two = render_side_by_side(
            [match_payload("a", [(0.5, 0.5, "yellow")]), match_payload("b", [(0.5, 0.5, "blue")])],
            CONFIG,
            canvas_size=200,
            margin=10,
        )

        self.assertEqual(one.shape[1], 200)
        self.assertEqual(two.shape[1], 400)
        self.assertEqual(two.shape[0], 200)

    def test_side_by_side_caps_columns(self) -> None:
        matches = [match_payload(f"m{index}", [(0.5, 0.5, "yellow")]) for index in range(6)]

        panel = render_side_by_side(matches, CONFIG, canvas_size=100, margin=5, max_columns=3)

        self.assertEqual(panel.shape[1], 300)

    def test_side_by_side_without_matches_is_still_a_canvas(self) -> None:
        panel = render_side_by_side([], CONFIG, canvas_size=120, margin=5)

        self.assertEqual(panel.shape, (120, 120, 3))

    def test_difference_of_identical_matches_has_no_signal(self) -> None:
        points = [(0.4, 0.4, "yellow")]
        same = render_difference(match_payload("a", points), match_payload("b", points), CONFIG, canvas_size=200, margin=10)
        differing = render_difference(
            match_payload("a", [(0.25, 0.25, "yellow")]),
            match_payload("b", [(0.75, 0.75, "yellow")]),
            CONFIG,
            canvas_size=200,
            margin=10,
        )

        self.assertLess(int(np.count_nonzero(same > 90)), int(np.count_nonzero(differing > 90)))


class BuildStageAggregateTests(unittest.TestCase):
    def setup_stage(self, tmp: str, matches: dict[str, list[tuple[float, float, str]]]) -> tuple[dict, dict]:
        registry: dict = {"stages": []}
        paths: dict[str, Path] = {}
        for match_id, points in matches.items():
            registry = register_match(registry, "gorge", match_id)
            csv_path = Path(tmp) / f"{match_id}.csv"
            write_stage_csv(csv_path, points)
            paths[match_id] = csv_path
        return registry, paths

    def test_single_match_skips_comparison_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, paths = self.setup_stage(tmp, {"match_a": [(0.3, 0.3, "yellow")]})

            report = build_stage_aggregate(
                "gorge", registry, paths, CONFIG, Path(tmp) / "out", canvas_size=200, margin=10
            )

        self.assertEqual(report["status"], "single_match")
        self.assertEqual(sorted(report["rendered"]), ["occupancy"])

    def test_two_matches_render_all_three_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, paths = self.setup_stage(
                tmp,
                {
                    "match_a": [(0.3, 0.3, "yellow"), (0.35, 0.35, "blue")],
                    "match_b": [(0.7, 0.7, "yellow")],
                },
            )
            output_dir = Path(tmp) / "out"

            report = build_stage_aggregate(
                "gorge", registry, paths, CONFIG, output_dir, canvas_size=200, margin=10
            )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["match_count"], 2)
            self.assertEqual(report["total_points"], 3)
            self.assertEqual(sorted(report["rendered"]), ["difference", "occupancy", "side_by_side"])
            for name in ("gorge_occupancy.png", "gorge_side_by_side.png", "gorge_difference.png"):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_per_match_point_counts_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, paths = self.setup_stage(
                tmp, {"match_a": [(0.3, 0.3, "yellow")] * 3, "match_b": [(0.7, 0.7, "blue")]}
            )

            report = build_stage_aggregate("gorge", registry, paths, CONFIG, Path(tmp) / "out", canvas_size=150, margin=10)

        self.assertEqual(report["per_match_points"], {"match_a": 3, "match_b": 1})

    def test_stage_without_data_renders_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = register_match({"stages": []}, "gorge", "match_a")

            report = build_stage_aggregate("gorge", registry, {}, CONFIG, Path(tmp) / "out")

        self.assertEqual(report["status"], "no_data")
        self.assertEqual(report["rendered"], {})

    def test_markdown_explains_single_match_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, paths = self.setup_stage(tmp, {"match_a": [(0.3, 0.3, "yellow")]})
            report = build_stage_aggregate("gorge", registry, paths, CONFIG, Path(tmp) / "out", canvas_size=150, margin=10)

            markdown = render_markdown(report)

        self.assertIn("# Stage Aggregate", markdown)
        self.assertIn("Label a second match", markdown)

    def test_markdown_lists_images_when_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, paths = self.setup_stage(
                tmp, {"match_a": [(0.3, 0.3, "yellow")], "match_b": [(0.7, 0.7, "blue")]}
            )
            report = build_stage_aggregate("gorge", registry, paths, CONFIG, Path(tmp) / "out", canvas_size=150, margin=10)

            markdown = render_markdown(report)

        self.assertIn("side_by_side", markdown)
        self.assertIn("Points Per Match", markdown)


if __name__ == "__main__":
    unittest.main()
