from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.report_io import strict_exit_code, write_json_report, write_text_report


class ReportIoTests(unittest.TestCase):
    def test_write_text_report_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "report.md"

            result = write_text_report(path, "# Report\n")

            self.assertEqual(result, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "# Report\n")

    def test_write_json_report_uses_consistent_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"

            write_json_report(path, {"status": "passed"})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "passed"})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_strict_exit_code_accepts_ready_and_passed(self) -> None:
        self.assertEqual(strict_exit_code("passed", strict=True), 0)
        self.assertEqual(strict_exit_code("ready", strict=True), 0)
        self.assertEqual(strict_exit_code("needs_data", strict=True), 1)
        self.assertEqual(strict_exit_code("needs_data", strict=False), 0)


if __name__ == "__main__":
    unittest.main()
