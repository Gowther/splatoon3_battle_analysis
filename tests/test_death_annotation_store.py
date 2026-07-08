from __future__ import annotations

import unittest

from src.death_annotation_store import (
    build_death_annotation_report,
    merge_review_rows,
    parse_notes,
    staging_item_to_death_review_row,
)


class DeathAnnotationStoreTests(unittest.TestCase):
    def test_parse_notes_extracts_structured_values(self) -> None:
        parsed = parse_notes("killer=team_2_slot_1; cause_weapon=Blaster; confidence=0.8")

        self.assertEqual(parsed["killer"], "team_2_slot_1")
        self.assertEqual(parsed["cause_weapon"], "Blaster")
        self.assertEqual(parsed["confidence"], "0.8")

    def test_staging_item_to_death_review_row_prefers_annotation_text(self) -> None:
        row = staging_item_to_death_review_row(
            {
                "id": "death:1",
                "updated_at": "now",
                "candidate": {
                    "match_id": "m1",
                    "elapsed_time": "2.0",
                    "frame_path": "crop.jpg",
                    "source_id": "e1",
                    "raw": {"event_id": "e1", "region": "death_message_center", "ocr_text": "raw"},
                },
                "annotation": {"text": "Blaster", "notes": "killer=team_2_slot_1; cause_weapon=Blaster"},
            }
        )

        self.assertEqual(row["candidate_id"], "death:1")
        self.assertEqual(row["event_id"], "e1")
        self.assertEqual(row["corrected_text"], "Blaster")
        self.assertEqual(row["killer"], "team_2_slot_1")
        self.assertEqual(row["cause_weapon"], "Blaster")

    def test_merge_review_rows_replaces_by_candidate_id(self) -> None:
        rows = merge_review_rows(
            [{"candidate_id": "a", "corrected_text": "old"}],
            [{"candidate_id": "a", "corrected_text": "new"}, {"candidate_id": "b", "corrected_text": "other"}],
        )

        self.assertEqual([row["candidate_id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["corrected_text"], "new")

    def test_build_death_annotation_report_reports_coverage(self) -> None:
        report = build_death_annotation_report(
            [{"candidate_id": "a", "event_id": "e1", "corrected_text": "Blaster", "cause_weapon": "Blaster"}],
            [{"candidate_id": "a"}, {"candidate_id": "b"}],
        )

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["coverage_ratio"], 0.5)
        self.assertEqual(report["labels_with_cause_weapon"], 1)


if __name__ == "__main__":
    unittest.main()
