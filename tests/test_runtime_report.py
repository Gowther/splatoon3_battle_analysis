from __future__ import annotations

import unittest

from src.runtime_report import build_runtime_report, format_seconds, render_markdown


class RuntimeReportTests(unittest.TestCase):
    def test_build_runtime_report_sums_steps(self) -> None:
        report = build_runtime_report("sample", [{"label": "a", "duration_seconds": 1.2}, {"label": "b", "duration_seconds": 2.3}])

        self.assertEqual(report["total_seconds"], 3.5)
        self.assertEqual(report["step_count"], 2)

    def test_render_markdown_includes_command_detail(self) -> None:
        report = build_runtime_report("sample", [{"label": "a", "command": "python x.py", "duration_seconds": 0.1}])
        markdown = render_markdown(report)

        self.assertIn("python x.py", markdown)
        self.assertEqual(format_seconds(0.1), "100ms")


if __name__ == "__main__":
    unittest.main()
