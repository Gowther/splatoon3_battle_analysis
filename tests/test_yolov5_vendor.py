from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.yolov5_vendor import build_vendor_report, ensure_vendor_ready, render_markdown


def touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class Yolov5VendorTests(unittest.TestCase):
    def test_complete_vendor_runtime_passes_with_local_artifact_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                "hubconf.py",
                "models/common.py",
                "models/experimental.py",
                "models/yolo.py",
                "utils/general.py",
                "utils/torch_utils.py",
                "train.py",
                "yolov5s.pt",
            ):
                touch(root, relative)

            report = build_vendor_report(root)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["local_artifacts"], ["yolov5s.pt"])

    def test_project_scripts_inside_vendor_are_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in (
                "hubconf.py",
                "models/common.py",
                "models/experimental.py",
                "models/yolo.py",
                "utils/general.py",
                "utils/torch_utils.py",
                "230111_run_analysis.py",
            ):
                touch(root, relative)

            report = build_vendor_report(root)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["project_script_files"], ["230111_run_analysis.py"])

    def test_ensure_vendor_ready_raises_with_missing_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                ensure_vendor_ready(Path(tmp))

    def test_render_markdown_documents_contract(self) -> None:
        markdown = render_markdown(
            {
                "status": "passed",
                "root": "/tmp/yolov5",
                "required_runtime_files": ["hubconf.py"],
                "local_artifacts": [],
                "blockers": [],
                "warnings": [],
            }
        )

        self.assertIn("vendored runtime dependency", markdown)


if __name__ == "__main__":
    unittest.main()
