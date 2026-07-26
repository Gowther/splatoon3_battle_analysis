import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from scripts.serve_active_learning_workbench import DATA_REVIEW_HTML
from src.core.paths import ROOT
from src.data_review_workbench import (
    build_time_snapshot,
    describe_data_source,
    record_data_review,
)


class DataReviewWorkbenchTests(unittest.TestCase):
    def test_describe_analysis_csv_and_snapshot_nearest_row(self):
        with self.workspace_tmp() as tmp:
            video = tmp / "n_match_99.mp4"
            video.write_bytes(b"fake video")
            csv_path = tmp / "n_match_99_smoothed.csv"
            csv_path.write_text(
                "elapsed_time,player_state_1,count_left,count_right,message\n"
                "1.0,0,50,50,\n"
                "1.5,14,48,50,lead\n",
                encoding="utf-8",
            )

            source = describe_data_source(csv_path)
            snapshot = build_time_snapshot(str(video), 1.4, [str(csv_path)])

        self.assertIsNotNone(source)
        self.assertEqual(source["kind"], "analysis_csv")
        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(snapshot["sources"][0]["selected_time"], 1.5)
        self.assertEqual(snapshot["sources"][0]["rows"][0]["values"]["count_left"], "48")

    def test_heatmap_snapshot_includes_rows_inside_window(self):
        with self.workspace_tmp() as tmp:
            video = tmp / "f_match_99.mp4"
            video.write_bytes(b"fake video")
            csv_path = tmp / "f_match_99_tracks.csv"
            csv_path.write_text(
                "match_id,time,team,track_slot,player_id,x,y,confidence\n"
                "f_match_99,1.0,blue,1,p1,10,20,0.9\n"
                "f_match_99,1.2,blue,2,p2,30,40,0.8\n"
                "f_match_99,2.0,yellow,1,p3,50,60,0.7\n",
                encoding="utf-8",
            )

            snapshot = build_time_snapshot(str(video), 1.1, [str(csv_path)], window=0.15)

        self.assertEqual(snapshot["sources"][0]["kind"], "heatmap_tracks")
        self.assertEqual(snapshot["sources"][0]["display_row_count"], 2)

    def test_record_data_review_writes_jsonl(self):
        with self.workspace_tmp() as tmp:
            video = tmp / "match_99.mp4"
            video.write_bytes(b"fake video")
            csv_path = tmp / "match_99_smoothed.csv"
            csv_path.write_text("elapsed_time,count_left\n1.0,50\n", encoding="utf-8")
            review_path = tmp / "reviews.jsonl"

            result = record_data_review(
                {
                    "video_path": str(video),
                    "time": 1.0,
                    "source_paths": [str(csv_path)],
                    "decision": "incorrect",
                    "incorrect_fields": ["count_left"],
                    "note": "count_left should be 49",
                    "snapshot": {"status": "ready"},
                },
                review_path=review_path,
            )
            review_text = review_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["summary"]["count"], 1)
        self.assertIn('"decision": "incorrect"', review_text)

    def test_data_review_html_contains_core_controls(self):
        self.assertIn("数据核验工作台", DATA_REVIEW_HTML)
        self.assertIn("/api/data-review/snapshot", DATA_REVIEW_HTML)
        self.assertIn("保存当前判断", DATA_REVIEW_HTML)

    @contextmanager
    def workspace_tmp(self):
        tmp_root = ROOT / ".cache" / "tests"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_name:
            yield Path(tmp_name)
