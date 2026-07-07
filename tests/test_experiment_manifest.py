from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiment_manifest import build_experiment_manifest, parse_labeled_path, render_markdown


class ExperimentManifestTests(unittest.TestCase):
    def test_build_experiment_manifest_hashes_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text("{}\n", encoding="utf-8")

            manifest = build_experiment_manifest(experiment_id="exp", sources=[("source", path)])

        self.assertEqual(manifest["experiment_id"], "exp")
        self.assertTrue(manifest["sources"][0]["exists"])
        self.assertIn("sha256", manifest["sources"][0])

    def test_parse_labeled_path_defaults_to_stem(self) -> None:
        self.assertEqual(parse_labeled_path("label=/tmp/a.json"), ("label", "/tmp/a.json"))
        self.assertEqual(parse_labeled_path("/tmp/a.json"), ("a", "/tmp/a.json"))

    def test_render_markdown_includes_verification(self) -> None:
        manifest = build_experiment_manifest(experiment_id="exp", sources=[], verification=["passed"])
        markdown = render_markdown(manifest)

        self.assertIn("passed", markdown)


if __name__ == "__main__":
    unittest.main()
