from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.parameter_experiments import (
    annotation_has_labels,
    annotation_label_summary,
    build_parameter_experiment_plan,
    build_variant_config,
    set_nested,
    trajectory_metrics_for_registry,
)


class HeatmapParameterExperimentTests(unittest.TestCase):
    def test_set_nested_updates_deep_value(self) -> None:
        config = {"a": {"b": 1}}

        set_nested(config, "a.c", 2)

        self.assertEqual(config["a"]["c"], 2)

    def test_build_variant_config_rewrites_output_paths(self) -> None:
        config = {
            "match": {"id": "m1", "output_dir": "outputs/old"},
            "outputs": {"player_tracks_csv": "outputs/old/player_tracks.csv"},
            "marker_detection": {"min_confidence": 0.45},
        }
        variant = build_variant_config(
            config,
            {"id": "soft", "overrides": {"marker_detection.min_confidence": 0.4}},
            "outputs/new",
        )

        self.assertEqual(variant["match"]["output_dir"], "outputs/new")
        self.assertEqual(variant["outputs"]["player_tracks_csv"], "outputs/new/player_tracks.csv")
        self.assertEqual(variant["marker_detection"]["min_confidence"], 0.4)

    def test_plan_marks_missing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = root / "annotation.csv"
            with annotation.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["match_id", "x", "y"])
                writer.writeheader()
                writer.writerow({"match_id": "m1", "x": "", "y": ""})
            registry = root / "registry.json"
            registry.write_text(
                '{"matches":[{"id":"m1","heatmap":{"config":"src/heatmap/config_f_match_3.yaml"}}]}\n',
                encoding="utf-8",
            )

            plan = build_parameter_experiment_plan(
                annotation_csv=annotation,
                registry_path=registry,
                output_root=root / "experiments",
                write_configs=False,
                candidates=[{"id": "soft", "overrides": {}}],
            )
            label_summary = annotation_label_summary(annotation)

        self.assertFalse(annotation_has_labels(annotation))
        self.assertEqual(label_summary["unlabeled_rows"], 1)
        self.assertEqual(plan["label_summary"]["unlabeled_rows"], 1)
        self.assertEqual(plan["status"], "needs_labels")
        self.assertIn("manual annotation", plan["blocking_reason"])
        self.assertEqual(plan["runs"][0]["status"], "needs_labels")

    def test_trajectory_metrics_aggregate_jump_reset_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "player_tracks.csv"
            gaps = root / "player_track_gaps.csv"
            with tracks.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "frame_index", "team", "player_id", "track_status"])
                writer.writeheader()
                writer.writerow({"time": "1", "frame_index": "10", "team": "blue", "player_id": "blue_1", "track_status": "matched"})
                writer.writerow({"time": "2", "frame_index": "20", "team": "blue", "player_id": "blue_1", "track_status": "jump_reset"})
            with gaps.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time"])
                writer.writeheader()
                writer.writerow({"time": "2"})

            metrics = trajectory_metrics_for_registry(
                {
                    "matches": [
                        {
                            "id": "m1",
                            "heatmap": {
                                "player_tracks": str(tracks),
                                "player_track_gaps": str(gaps),
                                "player_routes_dir": str(root / "routes"),
                            },
                        }
                    ]
                }
            )

        self.assertEqual(metrics["aggregate"]["track_rows"], 2)
        self.assertEqual(metrics["aggregate"]["gap_ratio"], 0.5)
        self.assertEqual(metrics["aggregate"]["jump_reset_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
