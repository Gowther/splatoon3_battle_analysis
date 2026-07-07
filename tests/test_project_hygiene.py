from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.project_hygiene import build_hygiene_report, render_markdown


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class ProjectHygieneTests(unittest.TestCase):
    def test_known_layout_passes_with_local_and_legacy_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dirname in ("src", "scripts", "tests", "outputs", "footages", ".models", "legacy"):
                (root / dirname).mkdir()
            touch(root / "PROJECT_LAYOUT.md")
            touch(root / "main_weapon_list.txt")

            report = build_hygiene_report(root)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["unexpected_root_entries"], [])
        self.assertIn(".models", {issue["items"][0] for issue in report["issues"] if issue["category"] == "legacy_reference"})

    def test_unexpected_root_entry_needs_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "scratch.py")

            report = build_hygiene_report(root)

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["unexpected_root_entries"], ["scratch.py"])

    def test_root_generated_file_needs_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            touch(root / "20240101_debug.csv")

            report = build_hygiene_report(root)

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["generated_root_files"], ["20240101_debug.csv"])

    def test_stray_pycache_is_informational(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "__pycache__").mkdir(parents=True)
            (root / ".venv" / "lib" / "__pycache__").mkdir(parents=True)

            report = build_hygiene_report(root)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["stray_pycache_dirs"], ["src/__pycache__"])

    def test_render_markdown_includes_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_hygiene_report(Path(tmp))

        markdown = render_markdown(report)

        self.assertIn("# Project Hygiene Report", markdown)
        self.assertIn("- status: `passed`", markdown)


if __name__ == "__main__":
    unittest.main()
