from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.runtime_benchmarks import DEFAULT_BENCHMARKS, build_runtime_benchmark_report, parse_runtime_report_arg, render_markdown


class RuntimeBenchmarksTests(unittest.TestCase):
    def test_build_runtime_benchmark_report_summarizes_ready_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "unit.json"
            report_path.write_text('{"name":"unit","total_seconds":1.5,"total_display":"1.50s","step_count":1}\n', encoding="utf-8")

            report = build_runtime_benchmark_report([("unit", report_path), ("missing", root / "missing.json")])

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["ready"], 1)
        self.assertEqual(report["summary"]["missing"], 1)

    def test_build_runtime_benchmark_report_flags_slow_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "unit.json"
            report_path.write_text('{"name":"unit","total_seconds":99.0,"total_display":"99.00s","step_count":1}\n', encoding="utf-8")

            report = build_runtime_benchmark_report([("unit_tests", report_path)])

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["summary"]["slow"], 1)
        self.assertEqual(report["reports"][0]["status"], "slow")

    def test_parse_runtime_report_arg_supports_label_prefix(self) -> None:
        label, path = parse_runtime_report_arg("unit=/tmp/unit.json")

        self.assertEqual(label, "unit")
        self.assertEqual(path, Path("/tmp/unit.json"))

    def test_render_markdown_includes_planned_commands(self) -> None:
        markdown = render_markdown(build_runtime_benchmark_report([]))

        self.assertIn("Planned Commands", markdown)

    def test_default_benchmarks_cover_run_analysis_and_heatmap_pipeline(self) -> None:
        commands = "\n".join(item["command"] for item in DEFAULT_BENCHMARKS)

        self.assertIn("python -m src.run_analysis", commands)
        self.assertIn("python -m src.heatmap.run_pipeline", commands)
        self.assertIn("--only-report", commands)
        self.assertTrue(all("expected_max_seconds" in item for item in DEFAULT_BENCHMARKS))


if __name__ == "__main__":
    unittest.main()
