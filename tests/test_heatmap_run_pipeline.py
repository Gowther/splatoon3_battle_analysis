from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.heatmap.run_pipeline import (
    clean_generated_outputs,
    configured_output_paths,
    run_stage_normalization,
    write_run_manifest,
)
from src.heatmap.stage_coordinates import discover_control_point_asset


class HeatmapRunPipelineTests(unittest.TestCase):
    def test_clean_generated_outputs_only_removes_paths_under_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "heatmap"
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(parents=True)
            generated_csv = output_dir / "points.csv"
            generated_csv.write_text("x\n1\n", encoding="utf-8")
            outside_file = root / "outside.csv"
            outside_file.write_text("keep\n", encoding="utf-8")
            config = {
                "match": {"output_dir": str(output_dir)},
                "outputs": {
                    "frames_dir": str(frames_dir),
                    "clean_points_csv": str(generated_csv),
                    "outside_csv": str(outside_file),
                },
                "state_join": {},
            }

            removed = clean_generated_outputs(config)

            self.assertIn(str(frames_dir), removed)
            self.assertIn(str(generated_csv), removed)
            self.assertFalse(frames_dir.exists())
            self.assertFalse(generated_csv.exists())
            self.assertTrue(outside_file.exists())

    def test_configured_output_paths_include_known_run_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "heatmap"
            config = {"match": {"output_dir": str(output_dir)}, "outputs": {}, "state_join": {}}

            paths = {path.name for path in configured_output_paths(config)}

        self.assertIn("resolved_config.yaml", paths)
        self.assertIn("color_calibration_report.csv", paths)
        self.assertIn("run_manifest.json", paths)

    def test_write_run_manifest_records_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "heatmap"
            report = output_dir / "report.md"
            report.parent.mkdir(parents=True)
            report.write_text("# report\n", encoding="utf-8")
            config = {
                "match": {"id": "sample", "input_video": "footages/sample.mp4", "output_dir": str(output_dir)},
                "outputs": {"report_md": str(report)},
                "state_join": {},
            }
            args = argparse.Namespace(
                device="cpu",
                warmup_frames=10,
                contact_limit=24,
                skip_ui_analysis=False,
                only_report=False,
                clean_output=True,
                event_csv=None,
                teams=None,
                disable_auto_colors=False,
            )

            manifest_path = write_run_manifest(
                config,
                args,
                source_config_path=Path("src/heatmap/config_sample.yaml"),
                resolved_config_path=output_dir / "resolved_config.yaml",
                color_report_path=output_dir / "color_calibration_report.csv",
                report_path=report,
                command_hint="python -m src.heatmap.run_pipeline --config sample",
                cleaned_paths=["outputs/old.csv"],
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["match_id"], "sample")
        self.assertTrue(payload["options"]["clean_output"])
        self.assertEqual(payload["cleaned_paths"], ["outputs/old.csv"])
        self.assertEqual(payload["stage_normalization"]["status"], "no_asset")

    def test_configured_output_paths_include_stage_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "heatmap"
            config = {"match": {"output_dir": str(output_dir)}, "outputs": {}, "state_join": {}}

            paths = {path.name for path in configured_output_paths(config)}

        self.assertIn("player_tracks_stage.csv", paths)

    def test_run_stage_normalization_without_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "heatmap"
            config = {
                "match": {"id": "no_asset_match", "output_dir": str(output_dir)},
                "outputs": {"player_tracks_csv": str(output_dir / "player_tracks.csv")},
                "map_view": {"roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
            }

            result = run_stage_normalization(config)

        self.assertEqual(result["status"], "no_asset")

    def test_run_stage_normalization_with_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "heatmap"
            output_dir.mkdir(parents=True)
            tracks_csv = output_dir / "player_tracks.csv"
            tracks_csv.write_text("x,y\n50,50\n25,75\n", encoding="utf-8")
            asset_path = root / "assets" / "stage_x.json"
            asset_path.parent.mkdir(parents=True)
            asset_path.write_text(
                json.dumps(
                    {
                        "stage_id": "stage_x",
                        "template": False,
                        "control_points": [
                            {"source": [0, 0], "target": [0.0, 0.0]},
                            {"source": [100, 0], "target": [1.0, 0.0]},
                            {"source": [100, 100], "target": [1.0, 1.0]},
                            {"source": [0, 100], "target": [0.0, 1.0]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "match": {"id": "with_asset_match", "output_dir": str(output_dir)},
                "outputs": {"player_tracks_csv": str(tracks_csv)},
                "map_view": {"coordinate_space": "video_pixels", "roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
                "stage_coordinates": {"control_point_asset": str(asset_path)},
            }

            result = run_stage_normalization(config)
            stage_csv = output_dir / "player_tracks_stage.csv"

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["method"], "homography")
            self.assertEqual(result["normalized_rows"], 2)
            self.assertTrue(stage_csv.exists())
            header = stage_csv.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("stage_x", header)
            self.assertIn("stage_inside_roi", header)

    def test_run_stage_normalization_ignores_template_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "heatmap"
            output_dir.mkdir(parents=True)
            (output_dir / "player_tracks.csv").write_text("x,y\n50,50\n", encoding="utf-8")
            asset_path = root / "template.json"
            asset_path.write_text(
                json.dumps(
                    {
                        "stage_id": "stage_x",
                        "template": True,
                        "control_points": [
                            {"source": [0, 0], "target": [0.0, 0.0]},
                            {"source": [100, 0], "target": [1.0, 0.0]},
                            {"source": [100, 100], "target": [1.0, 1.0]},
                            {"source": [0, 100], "target": [0.0, 1.0]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "match": {"id": "template_match", "output_dir": str(output_dir)},
                "outputs": {"player_tracks_csv": str(output_dir / "player_tracks.csv")},
                "map_view": {"roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
                "stage_coordinates": {"control_point_asset": str(asset_path)},
            }

            result = run_stage_normalization(config)

        self.assertEqual(result["status"], "no_asset")


class DiscoverControlPointAssetTests(unittest.TestCase):
    def make_asset(self, path: Path, stage_id: str, *, template: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stage_id": stage_id,
                    "template": template,
                    "control_points": [
                        {"source": [0, 0], "target": [0.0, 0.0]},
                        {"source": [10, 0], "target": [1.0, 0.0]},
                        {"source": [10, 10], "target": [1.0, 1.0]},
                        {"source": [0, 10], "target": [0.0, 1.0]},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_finds_asset_by_match_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "assets"
            self.make_asset(asset_dir / "match_7.json", "match_7")

            asset = discover_control_point_asset({"match": {"id": "match_7"}}, asset_dir=asset_dir)

        self.assertIsNotNone(asset)
        self.assertEqual(asset["stage_id"], "match_7")

    def test_explicit_asset_path_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_asset(root / "assets" / "match_7.json", "by_match_id")
            explicit = root / "explicit.json"
            self.make_asset(explicit, "explicit_stage")

            asset = discover_control_point_asset(
                {
                    "match": {"id": "match_7"},
                    "stage_coordinates": {"control_point_asset": str(explicit)},
                },
                asset_dir=root / "assets",
            )

        self.assertEqual(asset["stage_id"], "explicit_stage")

    def test_returns_none_for_missing_or_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asset_dir = Path(tmp) / "assets"
            self.make_asset(asset_dir / "match_8.json", "match_8", template=True)

            self.assertIsNone(discover_control_point_asset({"match": {"id": "match_8"}}, asset_dir=asset_dir))
            self.assertIsNone(discover_control_point_asset({"match": {"id": "match_9"}}, asset_dir=asset_dir))


if __name__ == "__main__":
    unittest.main()
