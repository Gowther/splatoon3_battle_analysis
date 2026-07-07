from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.heatmap.infer_player_tracks import clean_route_images


class HeatmapPlayerTracksTests(unittest.TestCase):
    def test_clean_route_images_removes_only_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stale_png = output_dir / "old_route.png"
            keep_txt = output_dir / "README.txt"
            stale_png.write_bytes(b"png")
            keep_txt.write_text("keep", encoding="utf-8")

            removed = clean_route_images(output_dir)

            self.assertEqual(removed, 1)
            self.assertFalse(stale_png.exists())
            self.assertTrue(keep_txt.exists())


if __name__ == "__main__":
    unittest.main()
