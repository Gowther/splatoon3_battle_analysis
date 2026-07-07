from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.model_error_report import analyze_csv, build_error_report, paths_from_evaluation_results


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

    def test_low_frequency_message_ocr_samples_do_not_need_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "messages.csv"
            write_rows(
                csv_path,
                [
                    {
                        "elapsed_time": str(index),
                        **{f"player_state_{player}": "0" for player in range(1, 9)},
                        **{f"weapon_{player}": f"W{player}" for player in range(1, 9)},
                        "count_left": "90",
                        "count_right": "80",
                        "area_count": "1",
                        "message": "ホ" if index in (2, 7) else "",
                    }
                    for index in range(20)
                ],
            )
            result = analyze_csv(csv_path)

        self.assertEqual(result["metrics"]["message_rows"], 2)
        self.assertNotIn("message_ocr", {issue["category"] for issue in result["issues"]})

    def test_dense_message_ocr_samples_still_get_info_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dense_messages.csv"
            write_rows(
                csv_path,
                [
                    {
                        "elapsed_time": str(index),
                        **{f"player_state_{player}": "0" for player in range(1, 9)},
                        **{f"weapon_{player}": f"W{player}" for player in range(1, 9)},
                        "count_left": "90",
                        "count_right": "80",
                        "area_count": "1",
                        "message": "ホ",
                    }
                    for index in range(12)
                ],
            )
            result = analyze_csv(csv_path)

        self.assertIn("message_ocr", {issue["category"] for issue in result["issues"]})

    def test_paths_from_evaluation_results_can_filter_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "evaluation_results.json"
            results_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "current",
                            "kind": "analysis",
                            "raw_csv": "outputs/evaluation/current/raw.csv",
                            "smoothed_csv": "outputs/evaluation/current/smoothed.csv",
                        },
                        {
                            "id": "old_candidate",
                            "kind": "analysis",
                            "raw_csv": "outputs/evaluation/old/raw.csv",
                            "smoothed_csv": "outputs/evaluation/old/smoothed.csv",
                        },
                        {
                            "id": "heatmap_only",
                            "kind": "heatmap",
                            "report": "outputs/heatmap/report.md",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            paths = paths_from_evaluation_results(results_path, use_smoothed=True, only_ids={"current"})

        self.assertEqual(paths, [Path("outputs/evaluation/current/smoothed.csv")])


if __name__ == "__main__":
    unittest.main()
