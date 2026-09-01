from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis_runtime import CSV_HEADER, write_analysis_csv
from src.csv_contracts import (
    ANALYSIS_CSV_CONTRACT,
    CORE_CSV_CONTRACTS,
    DEATH_EVENT_CSV_CONTRACT,
    DEATH_POSITION_CSV_CONTRACT,
    HEATMAP_ANNOTATION_CSV_CONTRACT,
    PLAYER_TRACK_CSV_CONTRACT,
    RAW_MARKER_CSV_CONTRACT,
    TRACK_CSV_CONTRACT,
)
from src.death_events import DEATH_EVENT_FIELDS
from src.heatmap.annotation_samples import ANNOTATION_FIELDS
from src.heatmap.clean_points import TRACK_FIELDNAMES
from src.heatmap.death_positions import POSITION_FIELDS
from src.heatmap.infer_player_tracks import PLAYER_TRACK_FIELDS


class CsvContractsTests(unittest.TestCase):
    def test_contract_ids_and_fields_are_unique(self) -> None:
        self.assertEqual(len({contract.contract_id for contract in CORE_CSV_CONTRACTS}), len(CORE_CSV_CONTRACTS))
        for contract in CORE_CSV_CONTRACTS:
            self.assertEqual(contract.schema_version, 1)
            self.assertEqual(len(contract.fields), len(set(contract.fields)))

    def test_existing_public_field_lists_match_versioned_contracts(self) -> None:
        self.assertEqual(CSV_HEADER, list(ANALYSIS_CSV_CONTRACT.fields))
        self.assertEqual(TRACK_FIELDNAMES, list(TRACK_CSV_CONTRACT.fields))
        self.assertEqual(PLAYER_TRACK_FIELDS, list(PLAYER_TRACK_CSV_CONTRACT.fields))
        self.assertEqual(ANNOTATION_FIELDS, list(HEATMAP_ANNOTATION_CSV_CONTRACT.fields))
        self.assertEqual(DEATH_EVENT_FIELDS, list(DEATH_EVENT_CSV_CONTRACT.fields))
        self.assertEqual(POSITION_FIELDS, list(DEATH_POSITION_CSV_CONTRACT.fields))

    def test_analysis_contract_preserves_legacy_positions(self) -> None:
        row = ANALYSIS_CSV_CONTRACT.positional_row(
            {"elapsed_time": 1.25, "weapon_8": "test_weapon", "timestamp": "2026-09-01T12:00:00"}
        )

        self.assertEqual(len(row), 33)
        self.assertEqual(row[0], 1.25)
        self.assertEqual(row[20], "test_weapon")
        self.assertEqual(row[29], "2026-09-01T12:00:00")

    def test_analysis_writer_rejects_misaligned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "analysis.csv"

            with self.assertRaisesRegex(ValueError, "expected 33"):
                write_analysis_csv(output, [["too", "short"]])

            self.assertFalse(output.exists())

    def test_pipeline_contracts_keep_required_handoff_fields(self) -> None:
        self.assertTrue({"time", "team", "x", "y", "confidence"}.issubset(RAW_MARKER_CSV_CONTRACT.fields))
        self.assertTrue({"track_slot", "track_status"}.issubset(TRACK_CSV_CONTRACT.fields))
        self.assertTrue(
            {"match_id", "time", "team", "track_slot", "player_id", "x", "y", "track_status"}.issubset(
                PLAYER_TRACK_CSV_CONTRACT.fields
            )
        )


if __name__ == "__main__":
    unittest.main()
