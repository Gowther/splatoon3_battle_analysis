from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.replay_timeline import (
    TIMELINE_FIELDS,
    EventSource,
    build_replay_timeline,
    normalize_source_events,
    write_csv,
)


class ReplayTimelineTests(unittest.TestCase):
    def test_normalize_source_events_applies_time_offset(self) -> None:
        events = normalize_source_events([EventSource("pov", [{"event_id": "e1", "time": "12.0"}], time_offset=-2.0)])

        self.assertEqual(events[0]["unified_time"], 10.0)
        self.assertEqual(events[0]["local_time"], 12.0)

    def test_build_replay_timeline_merges_same_event_across_sources(self) -> None:
        report = build_replay_timeline(
            [
                EventSource("main", [{"event_id": "e1", "time": "10.0", "event": "death", "victim": "v"}]),
                EventSource("pov", [{"event_id": "e1", "time": "12.0", "event": "death", "victim": "v"}], time_offset=-2.0),
            ]
        )

        self.assertEqual(report["timeline_event_count"], 1)
        row = report["timeline"][0]
        self.assertEqual(row["sources"], "main;pov")
        self.assertEqual(row["source_count"], 2)
        self.assertEqual(row["unified_time"], "10.000")

    def test_build_replay_timeline_attaches_clip_rows(self) -> None:
        report = build_replay_timeline(
            [EventSource("main", [{"event_id": "e1", "time": "10.0", "event": "death"}])],
            [{"clip_id": "clip1", "event_ids": "e1", "clip_path": "clips/clip1.mp4"}],
        )

        row = report["timeline"][0]
        self.assertEqual(row["clip_ids"], "clip1")
        self.assertEqual(row["clip_paths"], "clips/clip1.mp4")

    def test_write_timeline_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.csv"
            write_csv(path, TIMELINE_FIELDS, [{"timeline_id": "t1", "unified_time": "1.000"}])
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["timeline_id"], "t1")


if __name__ == "__main__":
    unittest.main()
