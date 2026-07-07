from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.annotation_ui import build_annotation_ui, render_annotation_html


class HeatmapAnnotationUiTests(unittest.TestCase):
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
        self.assertIn("Download CSV", html)

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

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["rows"], 1)


if __name__ == "__main__":
    unittest.main()
