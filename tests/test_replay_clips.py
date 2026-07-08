from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.replay_clips import REPLAY_CLIP_FIELDS, build_replay_clip_plan, event_score, write_csv


class ReplayClipTests(unittest.TestCase):
    def test_event_score_rewards_attribution(self) -> None:
        score = event_score(
            {
                "attribution_status": "attributed",
                "killer": "team_2_slot_1",
                "cause_weapon": "Blaster",
                "attribution_confidence": "0.8",
            }
        )

        self.assertGreater(score, 85)

    def test_build_replay_clip_plan_adds_multi_kill_highlight(self) -> None:
        events = [
            {
                "event_id": "e1",
                "match_id": "m1",
                "time": "10.0",
                "clip_start": "6.0",
                "clip_end": "13.0",
                "killer": "team_2_slot_1",
                "victim": "team_1_slot_1",
                "cause_weapon": "Blaster",
                "attribution_status": "attributed",
                "attribution_confidence": "0.8",
            },
            {
                "event_id": "e2",
                "match_id": "m1",
                "time": "14.0",
                "clip_start": "10.0",
                "clip_end": "17.0",
                "killer": "team_2_slot_1",
                "victim": "team_1_slot_2",
                "cause_weapon": "Blaster",
                "attribution_status": "attributed",
                "attribution_confidence": "0.8",
            },
        ]

        report = build_replay_clip_plan(events, output_dir=Path("outputs/replay_clips"))

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["event_count"], 2)
        self.assertEqual(report["highlight_count"], 1)
        self.assertEqual(report["clips"][0]["clip_type"], "multi_kill_candidate")
        self.assertEqual(report["clips"][0]["event_ids"], "e1;e2")

    def test_write_replay_clip_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.csv"
            write_csv(path, REPLAY_CLIP_FIELDS, [{"clip_id": "c1", "clip_type": "death_event"}])
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["clip_id"], "c1")
        self.assertEqual(rows[0]["clip_type"], "death_event")


if __name__ == "__main__":
    unittest.main()
