from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.weapon_training import build_resnet18_classifier, load_initial_classifier, model_output_count


class WeaponTrainingInitialModelTests(unittest.TestCase):
    def test_load_initial_classifier_accepts_matching_output_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(build_resnet18_classifier(2), path)

            model = load_initial_classifier(path, 2)

        self.assertEqual(model_output_count(model), 2)

    def test_load_initial_classifier_rejects_mismatched_output_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(build_resnet18_classifier(2), path)

            with self.assertRaises(ValueError):
                load_initial_classifier(path, 3)


if __name__ == "__main__":
    unittest.main()
