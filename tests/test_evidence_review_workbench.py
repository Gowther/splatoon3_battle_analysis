import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.serve_active_learning_workbench import EVIDENCE_REVIEW_HTML
from src.core.paths import ROOT
from src.evidence_review_workbench import build_video_evidence, record_evidence_review, record_weapon_correction


class EvidenceReviewWorkbenchTests(unittest.TestCase):
    def test_build_video_evidence_exports_weapon_and_death_screenshots(self):
        with self.workspace_tmp() as tmp:
            video = tmp / "test_match_99.mp4"
            self.write_video(video)
            csv_path = tmp / "match_99" / "ui_state.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(
                "elapsed_time,player_state_1,player_state_2,player_state_3,player_state_4,player_state_5,player_state_6,player_state_7,player_state_8,"
                "weapon_1,weapon_2,weapon_3,weapon_4,weapon_5,weapon_6,weapon_7,weapon_8\n"
                "0.0,0,0,0,0,0,0,0,0,A,B,C,D,E,F,G,H\n"
                "1.0,1,0,0,0,0,0,0,0,A,B,C,D,E,F,G,H\n",
                encoding="utf-8",
            )

            result = build_video_evidence(str(video), output_root=tmp / "evidence")

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["weapon"]["status"], "ready")
            self.assertEqual(result["weapon"]["weapons"][0]["weapon"], "A")
            self.assertEqual(result["weapon"]["crop_mode"], "manual")
            self.assertEqual(result["weapon"]["slot_boxes"], [])
            self.assertEqual(result["weapon"]["weapons"][0]["box"], {})
            self.assertTrue((ROOT / result["weapon"]["image_path"]).exists())
            self.assertEqual(result["death"]["event_count"], 1)
            self.assertTrue((ROOT / result["death"]["events"][0]["image_path"]).exists())

    def test_record_evidence_review_writes_jsonl(self):
        with self.workspace_tmp() as tmp:
            review_path = tmp / "reviews.jsonl"
            result = record_evidence_review(
                {
                    "item_type": "death",
                    "item_id": "death:test",
                    "video_path": "footages/test.mp4",
                    "source_path": "outputs/test.csv",
                    "decision": "incorrect",
                    "note": "time is late",
                    "payload": {"time": 1.0},
                },
                review_path=review_path,
            )
            text = review_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "saved")
        self.assertIn('"item_type": "death"', text)
        self.assertIn('"decision": "incorrect"', text)

    def test_record_weapon_correction_writes_separate_64px_dataset(self):
        with self.workspace_tmp() as tmp:
            image_path = tmp / "weapon_evidence.jpg"
            Image.new("RGB", (192, 108), (40, 50, 60)).save(image_path)
            labels_path = tmp / "labels.txt"
            labels_path.write_text("A\nB\n", encoding="utf-8")
            correction_root = tmp / "weapon_corrections"
            result = record_weapon_correction(
                {
                    "video_path": "footages/test_match.mp4",
                    "source_path": "outputs/test.csv",
                    "evidence_image_path": str(image_path),
                    "time": 12.5,
                    "slot": 3,
                    "predicted_weapon": "A",
                    "actual_weapon": "B",
                    "crop_box": {"left": 20, "top": 15, "width": 48, "height": 34},
                },
                correction_root=correction_root,
                correction_log_path=correction_root / "corrections.jsonl",
                labels_path=labels_path,
            )
            original_path = ROOT / result["record"]["original_path"]
            augmented_paths = [ROOT / item for item in result["record"]["augmented_paths"]]

            self.assertEqual(result["status"], "saved")
            self.assertEqual(original_path.parent, correction_root / "B")
            self.assertFalse(str(original_path).startswith(str(ROOT / "main_training_dataset")))
            self.assertEqual(len(augmented_paths), 4)
            self.assertTrue((correction_root / "corrections.jsonl").exists())
            with Image.open(original_path) as image:
                self.assertEqual(image.size, (64, 64))
            for path in augmented_paths:
                self.assertTrue(path.exists())

    def test_record_weapon_correction_requires_manual_crop_box(self):
        with self.workspace_tmp() as tmp:
            image_path = tmp / "weapon_evidence.jpg"
            Image.new("RGB", (192, 108), (40, 50, 60)).save(image_path)
            labels_path = tmp / "labels.txt"
            labels_path.write_text("A\nB\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crop_box is required"):
                record_weapon_correction(
                    {
                        "evidence_image_path": str(image_path),
                        "slot": 1,
                        "actual_weapon": "B",
                    },
                    correction_root=tmp / "weapon_corrections",
                    correction_log_path=tmp / "weapon_corrections" / "corrections.jsonl",
                    labels_path=labels_path,
                )

    def test_evidence_review_html_contains_core_controls(self):
        self.assertIn("证据核验工作台", EVIDENCE_REVIEW_HTML)
        self.assertIn("/api/evidence-review/video", EVIDENCE_REVIEW_HTML)
        self.assertIn("/api/evidence-review/weapon-correction", EVIDENCE_REVIEW_HTML)
        self.assertIn("待验证训练集", EVIDENCE_REVIEW_HTML)
        self.assertIn("在截图上拖框", EVIDENCE_REVIEW_HTML)
        self.assertIn("死亡时间点", EVIDENCE_REVIEW_HTML)

    def write_video(self, path: Path) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 5.0, (96, 64))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(12):
                frame = np.full((64, 96, 3), 20 + index * 10, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    @contextmanager
    def workspace_tmp(self):
        tmp_root = ROOT / "outputs" / ".test_evidence_review"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_name:
            yield Path(tmp_name)


if __name__ == "__main__":
    unittest.main()
