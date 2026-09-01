from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.heatmap.annotation_ui import (
    build_annotation_ui,
    prioritize_annotation_rows,
    relative_asset_path,
    render_annotation_html,
)


class HeatmapAnnotationUiTests(unittest.TestCase):
    def test_relative_asset_path_uses_relative_output_html_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "annotation_package"
            output_html = Path(os.path.relpath(package / "annotation_ui.html", Path.cwd()))
            frame = package / "frames" / "frame.jpg"

            asset_path = relative_asset_path(str(frame), output_html)

        self.assertEqual(asset_path, "frames/frame.jpg")

    def test_render_annotation_html_embeds_rows_and_click_handler(self) -> None:
        html = render_annotation_html(
            [
                {
                    "match_id": "m1",
                    "time": "10.0",
                    "team": "blue",
                    "annotation_id": "a1",
                    "frame_path": "frames/f.jpg",
                    "preview_path": "previews/f.jpg",
                }
            ],
            output_html=Path("/tmp/package/annotation_ui.html"),
        )

        self.assertIn("const rows =", html)
        self.assertIn("frameImage.addEventListener", html)
        self.assertIn('<html lang="zh-CN">', html)
        self.assertIn("下载标注 CSV", html)
        self.assertIn("非俯视图：跳过本帧本队", html)
        self.assertIn("skip_reason=non_overhead_view", html)
        self.assertIn('<option value="visible">可见，正常标注</option>', html)
        self.assertIn('visibilityInput.addEventListener("change"', html)
        self.assertIn("页面不会直接修改原 CSV", html)

    def test_build_annotation_ui_writes_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "annotation_template.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["match_id", "time", "team", "annotation_id"])
                writer.writeheader()
                writer.writerow({"match_id": "m1", "time": "1.0", "team": "blue", "annotation_id": "a1"})
            html_path = root / "annotation_ui.html"

            report = build_annotation_ui(csv_path, html_path)
            html = html_path.read_text(encoding="utf-8")

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["rows"], 1)
        self.assertIn("热力图人工标注", html)

    def test_prioritize_annotation_rows_moves_unlabeled_jump_resets_first(self) -> None:
        rows = [
            {
                "match_id": "m1",
                "time": "1.0",
                "team": "blue",
                "annotation_id": "matched",
                "source_track_status": "matched",
                "source_confidence": "0.2",
            },
            {
                "match_id": "m1",
                "time": "1.0",
                "team": "blue",
                "annotation_id": "jump",
                "source_track_status": "jump_reset",
                "source_confidence": "0.9",
            },
            {
                "match_id": "m1",
                "time": "2.0",
                "team": "orange",
                "annotation_id": "new",
                "source_track_status": "new",
                "source_confidence": "0.8",
            },
        ]

        ordered = prioritize_annotation_rows(rows, 2)

        self.assertEqual([row["annotation_id"] for row in ordered], ["jump", "new", "matched"])
        self.assertEqual(ordered[0]["_row_index"], "2")

    def test_build_annotation_ui_can_prioritize_rows_without_dropping_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "annotation_template.csv"
            fieldnames = [
                "match_id",
                "time",
                "team",
                "annotation_id",
                "source_track_status",
                "source_confidence",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "match_id": "m1",
                            "time": "1.0",
                            "team": "blue",
                            "annotation_id": "matched",
                            "source_track_status": "matched",
                            "source_confidence": "0.2",
                        },
                        {
                            "match_id": "m1",
                            "time": "1.0",
                            "team": "blue",
                            "annotation_id": "jump",
                            "source_track_status": "jump_reset",
                            "source_confidence": "0.9",
                        },
                    ]
                )
            html_path = root / "annotation_ui.html"

            report = build_annotation_ui(csv_path, html_path, priority_limit=1)
            html = html_path.read_text(encoding="utf-8")

        self.assertEqual(report["priority_rows"], 1)
        self.assertEqual(report["rows"], 2)
        self.assertLess(html.find('"annotation_id": "jump"'), html.find('"annotation_id": "matched"'))
        self.assertIn('"_row_index": "2"', html)


if __name__ == "__main__":
    unittest.main()
