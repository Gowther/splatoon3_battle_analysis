from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.death_attribution import (
    ATTRIBUTION_FIELDS,
    attribute_death_events,
    weapon_matches_text,
    write_csv,
)


class DeathAttributionTests(unittest.TestCase):
    def test_weapon_matches_text_normalizes_names(self) -> None:
        matches = weapon_matches_text("splatted by Custom Splattershot Jr", ["Custom-Splattershot-Jr", "Blaster"])

        self.assertEqual(matches, ["Custom-Splattershot-Jr"])

    def test_attribute_death_event_from_ocr_weapon_and_unique_snapshot(self) -> None:
        events = [
            {
                "event_id": "e1",
                "match_id": "m1",
                "time": "2.000",
                "event": "death",
                "victim_slot": "1",
                "victim": "team_1_slot_1",
            }
        ]
        ocr_rows = [{"event_id": "e1", "ocr_text": "Splattershot"}]
        analysis_rows = [{"elapsed_time": "2.000", "weapon_5": "Splattershot", "weapon_6": "Blaster"}]

        report = attribute_death_events(events, ocr_rows, analysis_rows, ["Splattershot", "Blaster"])

        row = report["events"][0]
        self.assertEqual(report["status"], "ready")
        self.assertEqual(row["attribution_status"], "attributed")
        self.assertEqual(row["cause_weapon"], "Splattershot")
        self.assertEqual(row["killer"], "team_2_slot_1")
        self.assertEqual(row["killer_slot"], 5)
        self.assertEqual(row["killer_weapon"], "Splattershot")
        self.assertEqual(row["review_required"], "false")

    def test_attribute_death_event_marks_multiple_weapon_matches_for_review(self) -> None:
        events = [{"event_id": "e1", "time": "2.000", "victim_slot": "1"}]
        ocr_rows = [{"event_id": "e1", "ocr_text": "Blaster"}]
        analysis_rows = [{"elapsed_time": "2.000", "weapon_5": "Blaster", "weapon_6": "Blaster"}]

        report = attribute_death_events(events, ocr_rows, analysis_rows, ["Blaster"])

        row = report["events"][0]
        self.assertEqual(row["attribution_status"], "weapon_only")
        self.assertEqual(row["review_required"], "true")
        self.assertIn("team_2_slot_1:Blaster", row["killer_candidates"])
        self.assertIn("team_2_slot_2:Blaster", row["killer_candidates"])

    def test_attribute_death_event_without_ocr_stays_reviewable(self) -> None:
        report = attribute_death_events([{"event_id": "e1", "time": "2.000"}], [], [], ["Blaster"])

        row = report["events"][0]
        self.assertEqual(row["attribution_status"], "no_ocr")
        self.assertEqual(row["review_required"], "true")

    def test_write_attribution_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attributed.csv"
            write_csv(path, ATTRIBUTION_FIELDS, [{"event_id": "e1", "attribution_status": "no_ocr"}])
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["event_id"], "e1")
        self.assertEqual(rows[0]["attribution_status"], "no_ocr")


if __name__ == "__main__":
    unittest.main()
