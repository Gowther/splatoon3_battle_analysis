from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.csv_contracts import PLAYER_TRACK_CSV_CONTRACT, STAGE_PLAYER_TRACK_CSV_CONTRACT
from src.heatmap.run_pipeline import (
    clean_generated_outputs,
    configured_output_paths,
    main as run_pipeline_main,
    run_pipeline,
    run_stage_normalization,
    run_stage_rendering,
    runtime_model_report,
    write_run_manifest,
)
from src.heatmap.stage_coordinates import discover_control_point_asset


class HeatmapRunPipelineTests(unittest.TestCase):
    def test_ui_state_is_generated_before_marker_detection(self) -> None:
        args = argparse.Namespace(
            contact_limit=24,
            skip_ui_analysis=False,
            device="cpu",
            warmup_frames=10,
        )
        config = {
            "match": {"input_video": "footages/test.mp4"},
            "sampling": {"start_seconds": 1.0, "stop_seconds": 2.0, "sample_fps": 5.0},
            "state_join": {"state_csv": "outputs/test/ui_state.csv"},
        }

        with patch("src.heatmap.run_pipeline.run_command") as run:
            run_pipeline(args, config, Path("outputs/test/resolved_config.yaml"))

        labels = [call.args[0] for call in run.call_args_list]
        self.assertLess(labels.index("run UI state analysis"), labels.index("detect markers"))

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
        self.assertEqual(payload["schema_version"], 2)
        self.assertTrue(payload["options"]["clean_output"])
        self.assertEqual(payload["cleaned_paths"], ["outputs/old.csv"])
        self.assertEqual(payload["stage_normalization"]["status"], "no_asset")

    def test_stage_output_and_manifest_obey_versioned_artifact_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "heatmap"
            output_dir.mkdir(parents=True)
            video = root / "match.mp4"
            video.write_bytes(b"tiny video fixture")
            source_config = root / "source.yaml"
            source_config.write_text("match: fixture\n", encoding="utf-8")
            resolved_config = output_dir / "resolved_config.yaml"
            resolved_config.write_text("match: fixture\nresolved: true\n", encoding="utf-8")
            report = output_dir / "report.md"
            report.write_text("# fixture report\n", encoding="utf-8")
            color_report = output_dir / "color_calibration_report.csv"
            color_report.write_text("team,hue\nblue,120\n", encoding="utf-8")
            player_tracks = output_dir / "player_tracks.csv"
            stage_tracks = output_dir / "player_tracks_stage.csv"
            row = {field: "" for field in PLAYER_TRACK_CSV_CONTRACT.fields}
            row.update({"match_id": "fixture", "time": "1.0", "team": "blue", "x": "50", "y": "25"})
            with player_tracks.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PLAYER_TRACK_CSV_CONTRACT.fields)
                writer.writeheader()
                writer.writerow(row)
            config = {
                "match": {"id": "fixture", "input_video": str(video), "output_dir": str(output_dir)},
                "outputs": {
                    "player_tracks_csv": str(player_tracks),
                    "player_tracks_stage_csv": str(stage_tracks),
                    "report_md": str(report),
                },
                "state_join": {},
                "map_view": {"roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
            }
            args = argparse.Namespace(
                device="cpu",
                warmup_frames=10,
                contact_limit=24,
                skip_ui_analysis=True,
                only_report=False,
                clean_output=False,
                event_csv=None,
                teams=None,
                disable_auto_colors=False,
            )

            stage_report = run_stage_normalization(config)
            manifest_path = write_run_manifest(
                config,
                args,
                source_config_path=source_config,
                resolved_config_path=resolved_config,
                color_report_path=color_report,
                report_path=report,
                command_hint="python -m src.heatmap.run_pipeline --config fixture",
                cleaned_paths=[],
                stage_normalization=stage_report,
                model_report={"schema_version": 1, "status": "not_used", "models": []},
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = {item["label"]: item for item in payload["artifacts"]}
            with stage_tracks.open(newline="", encoding="utf-8") as handle:
                stage_header = next(csv.reader(handle))

        self.assertEqual(stage_header, list(STAGE_PLAYER_TRACK_CSV_CONTRACT.fields))
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["contract_mismatches"], [])
        self.assertEqual(len(payload["run_fingerprint"]), 64)
        self.assertEqual(payload["inputs"]["input_video"]["sha256"], hashlib.sha256(b"tiny video fixture").hexdigest())
        self.assertEqual(artifacts["outputs.player_tracks_csv"]["contract_status"], "passed")
        self.assertEqual(artifacts["outputs.player_tracks_stage_csv"]["contract_status"], "passed")
        self.assertIn("sha256", artifacts["report"])

    def test_manifest_marks_existing_csv_with_wrong_header_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "heatmap"
            output_dir.mkdir(parents=True)
            tracks = output_dir / "team_tracks.csv"
            tracks.write_text("wrong,header\n1,2\n", encoding="utf-8")
            report = output_dir / "report.md"
            report.write_text("# report\n", encoding="utf-8")
            config = {
                "match": {"id": "bad_schema", "input_video": "missing.mp4", "output_dir": str(output_dir)},
                "outputs": {"tracks_csv": str(tracks), "report_md": str(report)},
                "state_join": {},
            }
            args = argparse.Namespace(
                device="cpu",
                warmup_frames=10,
                contact_limit=24,
                skip_ui_analysis=True,
                only_report=False,
                clean_output=False,
                event_csv=None,
                teams=None,
                disable_auto_colors=False,
            )

            manifest_path = write_run_manifest(
                config,
                args,
                source_config_path=Path("missing.yaml"),
                resolved_config_path=output_dir / "missing-resolved.yaml",
                color_report_path=output_dir / "missing-colors.csv",
                report_path=report,
                command_hint="fixture",
                cleaned_paths=[],
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = next(item for item in payload["artifacts"] if item["label"] == "outputs.tracks_csv")

        self.assertEqual(payload["status"], "needs_review")
        self.assertEqual(payload["contract_mismatches"], ["outputs.tracks_csv"])
        self.assertEqual(artifact["contract_status"], "mismatch")
        self.assertEqual(artifact["actual_columns"], ["wrong", "header"])

    def test_main_passes_model_provenance_to_run_manifest(self) -> None:
        args = argparse.Namespace(
            config="fixture.yaml",
            device="cpu",
            warmup_frames=10,
            contact_limit=24,
            skip_ui_analysis=False,
            only_report=True,
            clean_output=False,
            event_csv=None,
            teams=None,
            disable_auto_colors=False,
        )
        config = {"match": {"id": "fixture", "input_video": "fixture.mp4", "output_dir": "outputs/fixture"}}
        model_report = {"schema_version": 1, "status": "passed", "models": [{"id": "fixture_model"}]}

        with (
            patch("src.heatmap.run_pipeline.parse_args", return_value=args),
            patch("src.heatmap.run_pipeline.load_config", return_value=config),
            patch(
                "src.heatmap.run_pipeline.resolve_config",
                return_value=(config, Path("outputs/fixture/resolved.yaml"), Path("outputs/fixture/colors.csv")),
            ),
            patch("src.heatmap.run_pipeline.run_stage_normalization", return_value={"status": "no_points"}),
            patch("src.heatmap.run_pipeline.run_stage_rendering", return_value={"status": "skipped"}),
            patch(
                "src.heatmap.run_pipeline.run_death_position_pipeline",
                return_value={"status": "empty", "event_count": 0},
            ),
            patch("src.heatmap.run_pipeline.write_report", return_value=Path("outputs/fixture/report.md")),
            patch("src.heatmap.run_pipeline.runtime_model_report", return_value=model_report),
            patch("src.heatmap.run_pipeline.write_run_manifest", return_value=Path("outputs/fixture/run_manifest.json")) as write,
            patch("builtins.print"),
        ):
            result = run_pipeline_main()

        self.assertEqual(result, 0)
        self.assertEqual(write.call_args.kwargs["model_report"], model_report)

    def test_runtime_model_report_only_hashes_models_when_ui_analysis_runs(self) -> None:
        reused_args = argparse.Namespace(skip_ui_analysis=True, only_report=False)
        self.assertEqual(runtime_model_report(reused_args)["status"], "not_used")

        analysis_args = argparse.Namespace(skip_ui_analysis=False, only_report=False)
        registry = {"schema_version": 1, "models": []}
        expected = {"schema_version": 1, "status": "passed", "models": []}
        with (
            patch("src.heatmap.run_manifest.load_model_registry", return_value=registry),
            patch("src.heatmap.run_manifest.build_model_registry_report", return_value=expected) as build,
        ):
            report = runtime_model_report(analysis_args)

        self.assertEqual(report, expected)
        build.assert_called_once_with(registry, verify_hash=True)

    def test_configured_output_paths_include_stage_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "heatmap"
            config = {"match": {"output_dir": str(output_dir)}, "outputs": {}, "state_join": {}}

            paths = {path.name for path in configured_output_paths(config)}

        self.assertIn("player_tracks_stage.csv", paths)

    def test_run_stage_normalization_without_asset_uses_provisional_roi_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "heatmap"
            output_dir.mkdir(parents=True)
            (output_dir / "player_tracks.csv").write_text("x,y\n50,25\n", encoding="utf-8")
            config = {
                "match": {"id": "no_asset_match", "output_dir": str(output_dir)},
                "outputs": {"player_tracks_csv": str(output_dir / "player_tracks.csv")},
                "map_view": {"roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
            }

            result = run_stage_normalization(config)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["method"], "roi_linear_normalization")
        self.assertEqual(result["quality"], "provisional")
        self.assertEqual(result["normalized_rows"], 1)

    def test_failed_geometry_does_not_export_or_certify_tracks(self) -> None:
        from src.heatmap.stage_artifacts import stage_artifact_status, stage_metadata_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "tracks.csv"
            tracks.write_text("x,y\n5,5\n", encoding="utf-8")
            points = [
                {"source": [0, 0], "target": [0, 0]},
                {"source": [10, 0], "target": [1, 0]},
                {"source": [10, 10], "target": [1, 1]},
                {"source": [0, 10], "target": [0, 1]},
            ]
            config = {
                "match": {"id": "geometry_fixture", "output_dir": tmp},
                "outputs": {"player_tracks_csv": str(tracks)},
                "map_view": {"roi": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "control_points": points},
                "stage_coordinates": {"stage_id": "fixture_stage"},
            }
            valid = run_stage_normalization(config)
            output = Path(valid["output"])
            previous_csv = output.read_bytes()
            self.assertEqual(stage_artifact_status(output)["status"], "ready")

            points.append({"source": [5, 5], "target": [0.9, 0.9]})
            invalid = run_stage_normalization(config)

            self.assertEqual(invalid["status"], "needs_review")
            self.assertEqual(invalid["quality"], "rejected")
            self.assertIn("reprojection", invalid["quality_gate"]["failed_checks"])
            self.assertEqual(invalid["output"], "")
            self.assertEqual(output.read_bytes(), previous_csv)
            self.assertEqual(stage_artifact_status(output)["status"], "needs_calibration")
            self.assertEqual(json.loads(stage_metadata_path(output).read_text())["quality"], "rejected")

    def test_clustered_control_points_fail_coverage_even_with_exact_reprojection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracks = Path(tmp) / "tracks.csv"
            tracks.write_text("x,y\n5,5\n", encoding="utf-8")
            config = {
                "match": {"id": "cluster_fixture", "output_dir": tmp},
                "outputs": {"player_tracks_csv": str(tracks)},
                "map_view": {
                    "roi": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                    "control_points": [
                        {"source": [x, y], "target": [x / 100, y / 100]}
                        for x, y in ((0, 0), (10, 0), (10, 10), (0, 10))
                    ],
                },
            }
            result = run_stage_normalization(config)
            self.assertEqual(result["status"], "needs_review")
            self.assertIn("coverage", result["quality_gate"]["failed_checks"])
            self.assertFalse((Path(tmp) / "player_tracks_stage.csv").exists())

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

    def test_run_stage_normalization_ignores_template_asset_and_uses_roi_fallback(self) -> None:
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

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["method"], "roi_linear_normalization")
        self.assertEqual(result["homography_status"], "needs_control_points")

    def test_stage_rendering_is_skipped_without_normalized_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {"match": {"output_dir": tmp}, "outputs": {}}

            result = run_stage_rendering(config, {"status": "no_points"})

        self.assertEqual(result["status"], "skipped")


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
