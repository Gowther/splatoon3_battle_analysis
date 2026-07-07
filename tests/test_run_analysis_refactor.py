from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.run_analysis import CSV_HEADER, preview_dir_from_arg, write_analysis_csv


class RunAnalysisRefactorTests(unittest.TestCase):
    def test_write_analysis_csv_can_include_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.csv"
            write_analysis_csv(path, [["1.0", *[""] * 32]], include_header=True)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(rows[0], CSV_HEADER)
        self.assertEqual(rows[1][0], "1.0")

    def test_write_analysis_csv_can_omit_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.csv"
            write_analysis_csv(path, [["1.0", *[""] * 32]], include_header=False)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "1.0")

    def test_preview_dir_from_arg_creates_absolute_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "previews"

            result = preview_dir_from_arg(str(path))

            self.assertEqual(result, path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
