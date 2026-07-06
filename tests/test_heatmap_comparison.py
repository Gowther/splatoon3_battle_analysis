from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.comparison_report import build_comparison_report


TRACK_FIELDS = [
    "match_id",
    "heatmap_id",
    "frame_index",
    "time",
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
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class HeatmapComparisonTests(unittest.TestCase):
    def test_build_comparison_report_for_registered_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks = root / "player_tracks.csv"
            gaps = root / "player_track_gaps.csv"
            routes = root / "routes"
            routes.mkdir()
            (routes / "alpha_slot_1.png").write_bytes(b"placeholder")
            write_csv(
                tracks,
                TRACK_FIELDS,
                [
                    {
                        "match_id": "match_test",
                        "heatmap_id": "heatmap_test",
                        "frame_index": "1",
                        "time": "1.0",
                        "team": "alpha",
                        "track_slot": "1",
                        "player_id": "alpha_slot_1",
                        "x": "10",
                        "y": "20",
                        "confidence": "0.9",
                        "identity_confidence": "0.8",
                        "track_status": "matched",
                        "step_distance": "12",
                        "frame_path": "",
                    }
                ],
            )
            write_csv(gaps, TRACK_FIELDS, [])
            registry = {
                "matches": [
                    {
                        "id": "match_test",
                        "video": "footages/match_test.mp4",
                        "heatmap": {
                            "id": "heatmap_test",
                            "player_tracks": str(tracks),
                            "player_track_gaps": str(gaps),
                            "player_routes_dir": str(routes),
                            "teams": ["alpha"],
                            "quality_gates": {
                                "min_track_rows": 1,
                                "max_gap_ratio": 0.0,
                                "max_jump_reset_ratio": 0.0,
                                "min_route_images": 1,
                            },
                        },
                    }
                ]
            }
            report = build_comparison_report(registry)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["aggregate"]["match_count"], 1)
        self.assertEqual(report["matches"][0]["metrics"]["track_rows"], 1)


if __name__ == "__main__":
    unittest.main()
