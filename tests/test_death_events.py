from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.death_events import (
    DEATH_EVENT_FIELDS,
    build_death_event_report,
    extract_death_events,
    normalize_state,
    write_event_csv,
    write_event_json,
)


class DeathEventTests(unittest.TestCase):
    def test_normalize_state_supports_default_ids_and_names(self) -> None:
        self.assertEqual(normalize_state("1"), "dead")
        self.assertEqual(normalize_state("dead"), "dead")
        self.assertEqual(normalize_state("0"), "alive")
        self.assertEqual(normalize_state("14"), "alive")
        self.assertEqual(normalize_state("special"), "alive")
        self.assertEqual(normalize_state("8"), "unknown")

    def test_extract_death_events_from_player_state_transition(self) -> None:
        events = extract_death_events(
            [
                {"elapsed_time": "1.0", "player_state_1": "0", "weapon_1": "Splattershot"},
                {"elapsed_time": "2.0", "player_state_1": "1", "weapon_1": "Splattershot"},
                {"elapsed_time": "2.2", "player_state_1": "1", "weapon_1": "Splattershot"},
            ],
            match_id="match_a",
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.time, 2.0)
        self.assertEqual(event.event, "death")
        self.assertEqual(event.team, "team_1")
        self.assertEqual(event.victim, "team_1_slot_1")
        self.assertEqual(event.victim_slot, 1)
        self.assertEqual(event.victim_weapon, "Splattershot")
        self.assertEqual(event.source, "player_state_transition")
        self.assertEqual(event.clip_start, 0.0)
        self.assertEqual(event.clip_end, 6.0)

    def test_extract_death_events_ignores_repeated_or_initial_dead_rows(self) -> None:
        events = extract_death_events(
            [
                {"elapsed_time": "1.0", "player_state_1": "1"},
                {"elapsed_time": "2.0", "player_state_1": "1"},
                {"elapsed_time": "3.0", "player_state_1": "0"},
                {"elapsed_time": "4.0", "player_state_1": "1"},
                {"elapsed_time": "5.0", "player_state_1": "1"},
            ],
            match_id="match_b",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].time, 4.0)

    def test_extract_death_events_requires_known_alive_before_dead_by_default(self) -> None:
        events = extract_death_events(
            [
                {"elapsed_time": "1.0", "player_state_1": "8"},
                {"elapsed_time": "2.0", "player_state_1": "1"},
                {"elapsed_time": "3.0", "player_state_1": "0"},
                {"elapsed_time": "4.0", "player_state_1": "1"},
            ],
            match_id="match_c",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].time, 4.0)

    def test_build_report_and_write_outputs(self) -> None:
        rows = [
            {"elapsed_time": "1.0", "player_state_5": "alive", "weapon_5": "Splat Roller"},
            {"elapsed_time": "2.0", "player_state_5": "dead", "weapon_5": "Splat Roller"},
        ]
        events = extract_death_events(rows, match_id="match_d")
        report = build_death_event_report(rows, match_id="match_d", events=events)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["event_count"], 1)
        self.assertEqual(report["slots_with_death_events"], [5])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "death_events.csv"
            json_path = root / "death_events.json"
            write_event_csv(csv_path, events)
            write_event_json(json_path, report)

            with csv_path.open(newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(csv_rows[0]["time"], "2.000")
        self.assertEqual(csv_rows[0]["team"], "team_2")
        self.assertEqual(csv_rows[0]["victim"], "team_2_slot_1")
        self.assertEqual(list(csv_rows[0].keys()), DEATH_EVENT_FIELDS)
        self.assertEqual(payload["events"][0]["time"], 2.0)


if __name__ == "__main__":
    unittest.main()
