from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CsvContract:
    contract_id: str
    schema_version: int
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not self.fields:
            raise ValueError("CSV contract must define at least one field")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"CSV contract contains duplicate fields: {self.contract_id}")

    def index(self, field: str) -> int:
        try:
            return self.fields.index(field)
        except ValueError as exc:
            raise KeyError(f"unknown field for {self.contract_id}: {field}") from exc

    def positional_row(self, values: Mapping[str, object]) -> list[object]:
        unknown = set(values).difference(self.fields)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"unknown fields for {self.contract_id}: {names}")
        return [values.get(field) for field in self.fields]

    def validate_row(self, row: Sequence[object], *, row_number: int | None = None) -> None:
        if len(row) == len(self.fields):
            return
        location = f" row {row_number}" if row_number is not None else ""
        raise ValueError(
            f"{self.contract_id}{location} has {len(row)} columns; expected {len(self.fields)}"
        )

    def validate_header(self, fields: Sequence[str]) -> None:
        actual = tuple(fields)
        if actual == self.fields:
            return
        raise ValueError(
            f"{self.contract_id} header does not match schema version {self.schema_version}"
        )

    def manifest_record(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "columns": list(self.fields),
        }


ANALYSIS_CSV_CONTRACT = CsvContract(
    "analysis.frame_state",
    1,
    (
        "elapsed_time",
        *(f"player_state_{index}" for index in range(1, 9)),
        "count_left",
        "count_right",
        "penalty_left",
        "penalty_right",
        *(f"weapon_{index}" for index in range(1, 9)),
        "stage",
        "asari_count",
        "hoko_count",
        "area_count",
        "yagura_count",
        "message",
        "player_detected",
        "reserved_28",
        "timestamp",
        "reserved_30",
        "reserved_31",
        "reserved_32",
    ),
)

RAW_MARKER_CSV_CONTRACT = CsvContract(
    "heatmap.raw_markers",
    1,
    (
        "match_id",
        "time",
        "frame_index",
        "team",
        "player_id",
        "track_slot_hint",
        "x",
        "y",
        "confidence",
        "source",
        "area",
        "label_distance",
        "frame_path",
    ),
)

CLEAN_POINT_CSV_CONTRACT = CsvContract(
    "heatmap.clean_points",
    1,
    (
        "match_id",
        "time",
        "frame_index",
        "team",
        "player_id",
        "track_slot_hint",
        "x",
        "y",
        "confidence",
        "source",
        "clean_stage",
        "frame_path",
    ),
)

REJECTED_POINT_CSV_CONTRACT = CsvContract(
    "heatmap.rejected_points",
    1,
    (*RAW_MARKER_CSV_CONTRACT.fields, "reject_reason"),
)

TRACK_CSV_CONTRACT = CsvContract(
    "heatmap.tracks",
    1,
    (
        "match_id",
        "time",
        "frame_index",
        "team",
        "track_slot",
        "player_id",
        "x",
        "y",
        "confidence",
        "track_status",
        "step_distance",
        "time_delta",
        "prediction_error",
        "tracking_confidence",
        "observation_count",
        "source",
        "frame_path",
    ),
)

PLAYER_TRACK_CSV_CONTRACT = CsvContract(
    "heatmap.player_tracks",
    1,
    (
        "match_id",
        "time",
        "frame_index",
        "team",
        "track_slot",
        "player_id",
        "weapon_hint",
        "x",
        "y",
        "confidence",
        "identity_confidence",
        "tracking_confidence",
        "track_status",
        "step_distance",
        "time_delta",
        "prediction_error",
        "observation_count",
        "identity_method",
        "identity_note",
        "frame_path",
    ),
)

STAGE_PLAYER_TRACK_CSV_CONTRACT = CsvContract(
    "heatmap.player_tracks_stage",
    1,
    (*PLAYER_TRACK_CSV_CONTRACT.fields, "stage_x", "stage_y", "stage_inside_roi"),
)

TRACK_GAP_CSV_CONTRACT = CsvContract(
    "heatmap.track_gaps",
    1,
    (
        "match_id",
        "time",
        "frame_index",
        "team",
        "track_slot",
        "player_id",
        "track_status",
        "step_distance",
        "note",
    ),
)

HEATMAP_ANNOTATION_CSV_CONTRACT = CsvContract(
    "heatmap.annotations",
    1,
    (
        "match_id",
        "heatmap_id",
        "time",
        "frame_index",
        "team",
        "slot_hint",
        "annotation_id",
        "x",
        "y",
        "visibility",
        "frame_complete",
        "notes",
        "frame_path",
        "preview_path",
        "source_prediction_x",
        "source_prediction_y",
        "source_confidence",
        "source_track_status",
        "source_player_id",
    ),
)

PREDICTION_REFERENCE_CSV_CONTRACT = CsvContract(
    "heatmap.prediction_reference",
    1,
    (
        "match_id",
        "heatmap_id",
        "time",
        "frame_index",
        "team",
        "track_slot",
        "player_id",
        "x",
        "y",
        "confidence",
        "identity_confidence",
        "track_status",
        "step_distance",
        "frame_path",
        "preview_path",
    ),
)

DEATH_EVENT_CSV_CONTRACT = CsvContract(
    "events.deaths",
    1,
    (
        "event_id",
        "match_id",
        "time",
        "event",
        "team",
        "player",
        "killer",
        "victim",
        "clip_path",
        "segment_id",
        "victim_slot",
        "victim_weapon",
        "killer_slot",
        "killer_weapon",
        "cause_weapon",
        "cause_text",
        "confidence",
        "source",
        "evidence",
        "clip_start",
        "clip_end",
        "notes",
    ),
)

DEATH_POSITION_CSV_CONTRACT = CsvContract(
    "heatmap.death_positions",
    1,
    (
        "event_id",
        "match_id",
        "event_time",
        "event",
        "team",
        "victim",
        "victim_slot",
        "victim_weapon",
        "track_slot",
        "player_id",
        "x",
        "y",
        "point_time",
        "point_delta_seconds",
        "after_point_time",
        "after_point_delta_seconds",
        "stage_x",
        "stage_y",
        "stage_inside_roi",
        "location_status",
        "location_reason",
        "assignment_method",
        "assignment_confidence",
        "candidate_count",
        "candidate_margin",
        "source_frame",
        "clip_start",
        "clip_end",
        "evidence",
        "notes",
    ),
)

CORE_CSV_CONTRACTS = (
    ANALYSIS_CSV_CONTRACT,
    RAW_MARKER_CSV_CONTRACT,
    CLEAN_POINT_CSV_CONTRACT,
    REJECTED_POINT_CSV_CONTRACT,
    TRACK_CSV_CONTRACT,
    PLAYER_TRACK_CSV_CONTRACT,
    STAGE_PLAYER_TRACK_CSV_CONTRACT,
    TRACK_GAP_CSV_CONTRACT,
    HEATMAP_ANNOTATION_CSV_CONTRACT,
    PREDICTION_REFERENCE_CSV_CONTRACT,
    DEATH_EVENT_CSV_CONTRACT,
    DEATH_POSITION_CSV_CONTRACT,
)
