from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.death_events import DeathEvent
from src.heatmap.death_positions import build_death_position_rows, build_position_report, render_death_positions


def death(*, time: float = 10.0, slot: int = 1, team: str = "yellow") -> DeathEvent:
    return DeathEvent(
        event_id="death:test:1",
        match_id="test",
        time=time,
        event="death",
        team=team,
        player=f"{team}_slot_{slot}",
        killer="",
        victim=f"{team}_slot_{slot}",
        clip_path="",
        segment_id="death_1",
        victim_slot=slot,
        victim_weapon="Splattershot",
        killer_slot="",
        killer_weapon="",
        cause_weapon="",
        cause_text="",
        confidence=0.8,
        source="player_state_transition",
        evidence="alive->dead",
        clip_start=2.0,
        clip_end=14.0,
        notes="",
    )


def track(slot: int, time: float, x: float, y: float, *, team: str = "yellow") -> dict:
    return {
        "match_id": "test",
        "time": str(time),
        "team": team,
        "track_slot": str(slot),
        "player_id": f"{team}_slot_{slot}",
        "x": str(x),
        "y": str(y),
        "confidence": "0.8",
        "tracking_confidence": "0.8",
        "frame_path": f"frame_{time}.jpg",
    }


class DeathPositionTests(unittest.TestCase):
    def test_verified_slot_uses_matching_team_slot(self) -> None:
        rows = build_death_position_rows(
            [death(slot=1)],
            [track(1, 9, 100, 200), track(2, 9, 500, 600)],
            verified_slot_mapping=True,
        )

        self.assertEqual(rows[0]["location_status"], "located")
        self.assertEqual(rows[0]["location_reason"], "verified_hud_slot")
        self.assertEqual(rows[0]["track_slot"], "1")
        self.assertEqual(rows[0]["x"], "100.00")

    def test_unverified_assignment_prefers_track_that_disappears(self) -> None:
        rows = build_death_position_rows(
            [death()],
            [
                track(1, 9, 100, 200),
                track(2, 9, 500, 600),
                track(2, 10, 510, 610),
            ],
            verified_slot_mapping=False,
            ambiguity_margin=0.1,
        )

        self.assertEqual(rows[0]["track_slot"], "1")
        self.assertEqual(rows[0]["location_status"], "located_unverified")
        self.assertEqual(rows[0]["location_reason"], "candidate_identity_unverified")
        self.assertGreater(float(rows[0]["candidate_margin"]), 0.1)

    def test_equal_candidates_are_marked_ambiguous(self) -> None:
        rows = build_death_position_rows(
            [death()],
            [track(1, 9, 100, 200), track(2, 9, 500, 600)],
            verified_slot_mapping=False,
            ambiguity_margin=0.1,
        )

        self.assertEqual(rows[0]["location_status"], "ambiguous")
        self.assertEqual(rows[0]["location_reason"], "candidate_identity_ambiguous")
        self.assertEqual(rows[0]["x"], "100.00")

    def test_simultaneous_team_deaths_use_distinct_tracks(self) -> None:
        rows = build_death_position_rows(
            [death(slot=1), death(slot=2)],
            [track(1, 9, 100, 200), track(2, 9, 500, 600)],
            verified_slot_mapping=False,
        )

        self.assertEqual({row["track_slot"] for row in rows}, {"1", "2"})
        self.assertEqual({(row["x"], row["y"]) for row in rows}, {("100.00", "200.00"), ("500.00", "600.00")})

    def test_missing_recent_point_is_unknown_and_has_no_coordinates(self) -> None:
        rows = build_death_position_rows(
            [death()],
            [track(1, 5, 100, 200)],
            pre_window=6.0,
            max_point_delta=2.0,
        )

        self.assertEqual(rows[0]["location_status"], "unknown")
        self.assertEqual(rows[0]["location_reason"], "latest_track_point_too_old")
        self.assertEqual(rows[0]["x"], "")
        self.assertEqual(rows[0]["point_delta_seconds"], "5.000")

    def test_stage_coordinates_follow_selected_track_point(self) -> None:
        stage = [{**track(1, 9, 100, 200), "stage_x": "0.25", "stage_y": "0.75", "stage_inside_roi": "True"}]

        rows = build_death_position_rows(
            [death()],
            [track(1, 9, 100, 200)],
            stage_rows=stage,
            verified_slot_mapping=True,
        )

        self.assertEqual(rows[0]["stage_x"], "0.25")
        self.assertEqual(rows[0]["stage_y"], "0.75")

    def test_report_counts_structured_location_reasons(self) -> None:
        report = build_position_report(
            [
                {"location_status": "located", "location_reason": "verified_hud_slot"},
                {"location_status": "unknown", "location_reason": "no_predeath_track_candidate"},
            ],
            match_id="test",
        )

        self.assertEqual(report["located_count"], 1)
        self.assertEqual(report["unknown_count"], 1)
        self.assertEqual(
            report["reason_counts"],
            {"no_predeath_track_candidate": 1, "verified_hud_slot": 1},
        )

    def test_renderer_writes_source_and_stage_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rendered = root / "rendered"
            rendered_stage = root / "rendered_stage"
            rendered.mkdir()
            rendered_stage.mkdir()
            cv2.imwrite(str(rendered / "team_routes.png"), np.zeros((300, 400, 3), dtype=np.uint8))
            cv2.imwrite(str(rendered_stage / "stage_routes.png"), np.zeros((300, 300, 3), dtype=np.uint8))
            config = {
                "match": {"output_dir": str(root)},
                "outputs": {"rendered_dir": str(rendered), "rendered_stage_dir": str(rendered_stage)},
                "rendering": {},
                "teams": {"yellow": {"hsv_ranges": []}},
            }
            rows = [
                {
                    "event_time": "10.0",
                    "team": "yellow",
                    "track_slot": "1",
                    "x": "100",
                    "y": "120",
                    "stage_x": "0.25",
                    "stage_y": "0.75",
                    "location_status": "located",
                }
            ]

            report = render_death_positions(rows, config)

            self.assertEqual(report["status"], "ready")
            self.assertTrue((rendered / "routes_with_deaths.png").is_file())
            self.assertTrue((rendered_stage / "stage_routes_with_deaths.png").is_file())


if __name__ == "__main__":
    unittest.main()
