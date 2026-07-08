from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.death_ocr_candidates import (
    DEATH_OCR_CANDIDATE_FIELDS,
    OcrRegion,
    build_death_ocr_candidates,
    pixel_box,
    selected_frame_entries,
    write_csv,
)


class DeathOcrCandidateTests(unittest.TestCase):
    def test_selected_frame_entries_prefers_frames_after_event(self) -> None:
        asset = {
            "time": "10.0",
            "frame_times": "8.000;10.000;11.500;13.000",
            "frame_paths": "a.jpg;b.jpg;c.jpg;d.jpg",
        }

        selected = selected_frame_entries(asset, max_frames_per_event=2)

        self.assertEqual([row["frame_path"] for row in selected], ["c.jpg", "b.jpg"])

    def test_pixel_box_clamps_roi(self) -> None:
        self.assertEqual(pixel_box(100, 50, (-1.0, 0.2, 2.0, 0.4)), (0, 10, 100, 20))

    def test_build_death_ocr_candidates_writes_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.jpg"
            image = np.full((100, 200, 3), 200, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(frame), image))
            assets = [
                {
                    "event_id": "death:m1:slot1:2p000",
                    "match_id": "m1",
                    "time": "2.000",
                    "victim": "team_1_slot_1",
                    "victim_slot": "1",
                    "source_video": "video.mp4",
                    "frame_times": "2.000",
                    "frame_paths": str(frame),
                }
            ]

            report = build_death_ocr_candidates(
                assets,
                output_dir=root / "ocr",
                regions=(OcrRegion("test_region", (0.0, 0.0, 0.5, 0.5), "review_test"),),
            )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["candidate_count"], 1)
            candidate = report["candidates"][0]
            self.assertEqual(candidate["target"], "death_event_ocr")
            self.assertEqual(candidate["reason"], "review_test")
            self.assertEqual(candidate["event_id"], "death:m1:slot1:2p000")
            self.assertEqual(candidate["x2"], 100)
            self.assertEqual(candidate["y2"], 50)
            self.assertTrue(Path(candidate["crop_path"]).exists())

            csv_path = root / "candidates.csv"
            write_csv(csv_path, DEATH_OCR_CANDIDATE_FIELDS, report["candidates"])
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["frame_path"], candidate["crop_path"])


if __name__ == "__main__":
    unittest.main()
