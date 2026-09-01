from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.core.paths import display_path, relative_asset_path, resolve_project_path


class CorePathsTests(unittest.TestCase):
    def test_resolve_project_path_handles_optional_relative_and_absolute_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            absolute = root / "absolute.csv"

            self.assertIsNone(resolve_project_path(None, root))
            self.assertIsNone(resolve_project_path("", root))
            self.assertEqual(resolve_project_path("outputs/result.csv", root), root / "outputs/result.csv")
            self.assertEqual(resolve_project_path(absolute, root), absolute)

    def test_display_path_keeps_external_paths_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertEqual(display_path(root / "outputs/result.csv", root), "outputs/result.csv")
            self.assertEqual(display_path(Path("/tmp/external.csv"), root), "/tmp/external.csv")
            self.assertEqual(display_path(None, root), "")

    def test_relative_asset_path_is_relative_to_document_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "annotation_package"
            document = Path(os.path.relpath(package / "annotation_ui.html", Path.cwd()))
            frame = package / "frames" / "frame.jpg"

            self.assertEqual(relative_asset_path(frame, document), "frames/frame.jpg")
            self.assertEqual(relative_asset_path("", document), "")
            self.assertEqual(
                relative_asset_path(
                    "outputs/package/frames/frame.jpg",
                    "outputs/package/annotation_ui.html",
                    root,
                ),
                "frames/frame.jpg",
            )


if __name__ == "__main__":
    unittest.main()
