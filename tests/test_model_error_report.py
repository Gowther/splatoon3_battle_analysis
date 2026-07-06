from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.model_error_report import analyze_csv, build_error_report


FIELDNAMES = [
    "elapsed_time",
    *[f"player_state_{index}" for index in range(1, 9)],
    "count_left",
    "count_right",
    "penalty_left",
    "penalty_right",
    *[f"weapon_{index}" for index in range(1, 9)],
    "stage",
    "asari_count",
    "hoko_count",
    "area_count",
    "yagura_count",
    "message",
    "player_detected",
    "reserved_28",
    "timestamp",
    "reserved_30",
    "reserved_31",
    "reserved_32",
]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {field: "" for field in FIELDNAMES}
            full.update(row)
            writer.writerow(full)


class ModelErrorReportTests(unittest.TestCase):
    def test_analyze_csv_flags_count_jump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "analysis.csv"
            write_rows(
                csv_path,
                [
                    {"elapsed_time": "1.0", "count_left": "90", "weapon_1": "A", "player_state_1": "0"},
                    {"elapsed_time": "1.2", "count_left": "7", "weapon_1": "A", "player_state_1": "0"},
                ],
            )
            result = analyze_csv(csv_path)
        categories = {issue["category"] for issue in result["issues"]}
        self.assertIn("count_ocr", categories)
        self.assertEqual(result["metrics"]["count_jump_warnings"], 1)

    def test_build_error_report_passes_clean_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "clean.csv"
            write_rows(
                csv_path,
                [
                    {
                        "elapsed_time": "1.0",
                        **{f"player_state_{index}": "0" for index in range(1, 9)},
                        **{f"weapon_{index}": f"W{index}" for index in range(1, 9)},
                        "count_left": "90",
                        "count_right": "80",
                        "area_count": "1",
                        "player_detected": "True",
                    }
                ],
            )
            report = build_error_report([csv_path])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["file_count"], 1)


if __name__ == "__main__":
    unittest.main()
