from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.heatmap.run_pipeline import clean_generated_outputs, configured_output_paths, write_run_manifest


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


if __name__ == "__main__":
    unittest.main()
