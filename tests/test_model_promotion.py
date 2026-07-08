from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.model_promotion import apply_model_promotion, build_model_promotion_plan, render_markdown
from src.model_registry import load_model_registry, save_model_registry


class ModelPromotionTests(unittest.TestCase):
    def test_build_plan_marks_validated_candidate_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.pth"
            candidate.write_bytes(b"candidate")
            validation = root / "validation.json"
            validation.write_text(json.dumps({"status": "passed"}), encoding="utf-8")

            plan = build_model_promotion_plan(
                {
                    "models": [
                        {
                            "id": "weapon",
                            "path": str(root / "current.pth"),
                            "file_type": "pth",
                            "promotion_gate": "scripts/run_validation_suite.py",
                        }
                    ]
                },
                model_id="weapon",
                candidate_path=candidate,
                validation_report=validation,
            )

        self.assertEqual(plan["status"], "ready")
        self.assertEqual(plan["candidate_sha256"], hashlib.sha256(b"candidate").hexdigest())
        self.assertEqual(plan["blockers"], [])

    def test_build_plan_requires_validation_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.pt"
            candidate.write_bytes(b"candidate")

            plan = build_model_promotion_plan(
                {
                    "models": [
                        {
                            "id": "detector",
                            "path": str(root / "current.pt"),
                            "file_type": "pt",
                            "promotion_gate": "scripts/run_validation_suite.py",
                        }
                    ]
                },
                model_id="detector",
                candidate_path=candidate,
            )

        self.assertEqual(plan["status"], "needs_validation")
        self.assertIn("validation report is required", plan["warnings"][0])

    def test_build_plan_blocks_suffix_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.pth"
            candidate.write_bytes(b"candidate")

            plan = build_model_promotion_plan(
                {"models": [{"id": "detector", "path": str(root / "current.pt"), "file_type": "pt"}]},
                model_id="detector",
                candidate_path=candidate,
            )

        self.assertEqual(plan["status"], "failed")
        self.assertIn("does not match expected .pt", plan["blockers"][0])

    def test_apply_model_promotion_copies_model_updates_hash_and_backs_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "models.json"
            target = root / "current.pth"
            candidate = root / "candidate.pth"
            validation = root / "validation.json"
            backup_dir = root / "backups"
            target.write_bytes(b"old")
            candidate.write_bytes(b"new")
            validation.write_text(json.dumps({"status": "ready"}), encoding="utf-8")
            save_model_registry(
                {
                    "models": [
                        {
                            "id": "weapon",
                            "path": str(target),
                            "file_type": "pth",
                            "expected_sha256": hashlib.sha256(b"old").hexdigest(),
                            "promotion_gate": "scripts/run_validation_suite.py",
                        }
                    ]
                },
                registry_path,
            )
            plan = build_model_promotion_plan(
                load_model_registry(registry_path),
                model_id="weapon",
                candidate_path=candidate,
                validation_report=validation,
                backup_dir=backup_dir,
            )

            promoted = apply_model_promotion(registry_path, plan, backup_dir=backup_dir)
            registry = load_model_registry(registry_path)

            self.assertEqual(promoted["status"], "promoted")
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(registry["models"][0]["expected_sha256"], hashlib.sha256(b"new").hexdigest())
            self.assertTrue(Path(promoted["backup_path"]).exists())

    def test_render_markdown_lists_blockers(self) -> None:
        markdown = render_markdown(
            {
                "status": "failed",
                "model_id": "missing",
                "candidate_path": "candidate.pt",
                "target_path": "",
                "validation": {},
                "blockers": ["model id is not registered"],
                "warnings": [],
                "actions": [],
            }
        )

        self.assertIn("# Model Promotion Plan", markdown)
        self.assertIn("model id is not registered", markdown)


if __name__ == "__main__":
    unittest.main()
