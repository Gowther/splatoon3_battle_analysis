from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.model_registry import build_model_registry_report, render_markdown


class ModelRegistryTests(unittest.TestCase):
    def test_build_model_registry_report_checks_files_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            content = b"model"
            path.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()

            report = build_model_registry_report(
                {
                    "models": [
                        {
                            "id": "m1",
                            "area": "detector",
                            "path": str(path),
                            "expected_sha256": expected,
                            "training_status": "supported",
                        }
                    ]
                },
                verify_hash=True,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["models"][0]["hash_status"], "passed")
        self.assertEqual(report["models"][0]["size_bytes"], 5)

    def test_build_model_registry_report_flags_missing_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_text("actual", encoding="utf-8")

            report = build_model_registry_report(
                {
                    "models": [
                        {"id": "bad_hash", "path": str(path), "expected_sha256": "wrong"},
                        {"id": "missing", "path": str(Path(tmp) / "missing.pt"), "expected_sha256": "anything"},
                    ]
                },
                verify_hash=True,
            )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["missing_models"], ["missing"])
        self.assertEqual(report["hash_mismatches"], ["bad_hash"])

    def test_render_markdown_includes_registry_rows(self) -> None:
        markdown = render_markdown(
            {
                "status": "passed",
                "model_count": 1,
                "verify_hash": False,
                "missing_models": [],
                "hash_mismatches": [],
                "models": [{"id": "m1", "area": "detector", "path": "models/m.pt", "exists": True}],
            }
        )

        self.assertIn("# Model Registry", markdown)
        self.assertIn("`m1`", markdown)
        self.assertIn("models/m.pt", markdown)


if __name__ == "__main__":
    unittest.main()
