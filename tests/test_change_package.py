from __future__ import annotations

import unittest

from src.change_package import build_change_package, change_group, change_theme, parse_git_status, render_markdown


class ChangePackageTests(unittest.TestCase):
    def test_parse_git_status_groups_paths(self) -> None:
        changes = parse_git_status(" M src/a.py\n?? scripts/b.py\n M yolov5/train.py\n")

        self.assertEqual(len(changes), 3)
        self.assertEqual(change_group("yolov5/train.py"), "legacy_yolov5_review_separately")

    def test_render_markdown_includes_verification(self) -> None:
        report = build_change_package(" M src/a.py\n", verification=["tests passed"])
        markdown = render_markdown(report)

        self.assertIn("tests passed", markdown)
        self.assertIn("runtime_code", markdown)

    def test_build_change_package_adds_theme_and_risk_flags(self) -> None:
        report = build_change_package("?? src/heatmap/annotation_ui.py\n M yolov5/train.py\n")

        self.assertEqual(change_theme("src/heatmap/annotation_ui.py"), "manual_heatmap_labels")
        self.assertIn("manual_heatmap_labels", report["themes"])
        self.assertTrue(any(flag["area"] == "legacy_yolov5_review_separately" for flag in report["risk_flags"]))


if __name__ == "__main__":
    unittest.main()
