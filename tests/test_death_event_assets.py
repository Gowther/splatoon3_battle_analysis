from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.death_event_assets import (
    DEATH_ASSET_FIELDS,
    enrich_events_with_assets,
    export_death_event_assets,
    sample_times_for_event,
    write_csv,
)


class DeathEventAssetTests(unittest.TestCase):
    def test_sample_times_are_clamped_and_unique(self) -> None:
        row = {"time": "2.0", "clip_start": "1.0", "clip_end": "3.0"}

        samples = sample_times_for_event(row, offsets=(-4, -1, 0, 1, 4))

        self.assertEqual(samples, [1.0, 2.0, 3.0])

    def test_export_death_event_assets_writes_review_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "source.jpg"
            image = np.full((64, 96, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            events = [
                {
                    "event_id": "death:m1:slot1:2p000",
                    "match_id": "m1",
                    "time": "2.0",
                    "victim": "team_1_slot_1",
                    "victim_slot": "1",
                    "clip_start": "0.0",
                    "clip_end": "4.0",
                }
            ]

            report = export_death_event_assets(
                events,
                video_path=image_path,
                output_dir=root / "assets",
                frame_offsets=(0.0, 1.0),
                write_clips=False,
            )

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["ready_count"], 1)
            asset = report["assets"][0]
            frame_paths = asset["frame_paths"].split(";")
            self.assertEqual(asset["frame_times"], "2.000;3.000")
            self.assertEqual(len(frame_paths), 2)
            for path in frame_paths:
                self.assertTrue((Path.cwd() / path).exists() if not Path(path).is_absolute() else Path(path).exists())

            manifest = root / "manifest.csv"
            write_csv(manifest, DEATH_ASSET_FIELDS, report["assets"])
            with manifest.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["event_id"], "death:m1:slot1:2p000")
            self.assertEqual(rows[0]["status"], "ready")

    def test_enrich_events_with_assets_adds_clip_path_and_asset_note(self) -> None:
        enriched = enrich_events_with_assets(
            [{"event_id": "e1", "notes": "base"}],
            [{"event_id": "e1", "clip_path": "clips/e1.mp4", "asset_dir": "assets/e1"}],
        )

        self.assertEqual(enriched[0]["clip_path"], "clips/e1.mp4")
        self.assertEqual(enriched[0]["notes"], "base; asset_dir=assets/e1")


if __name__ == "__main__":
    unittest.main()
