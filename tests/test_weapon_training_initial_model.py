from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import optim
from torch.optim import lr_scheduler

from src.weapon_training import (
    build_resnet18_classifier,
    load_initial_classifier,
    load_training_checkpoint,
    model_output_count,
    save_training_checkpoint,
    validate_training_checkpoint,
)


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

    def test_training_checkpoint_round_trip_restores_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model = build_resnet18_classifier(2)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
            save_training_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=3,
                best_val_accuracy=0.75,
                best_model_state_dict=model.state_dict(),
                class_names=["a", "b"],
                initialization="resnet18",
                epoch_metrics=[{"epoch": 3}],
            )
            resumed_model = build_resnet18_classifier(2)
            resumed_optimizer = optim.SGD(resumed_model.parameters(), lr=0.01, momentum=0.9)
            resumed_scheduler = lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.1)

            metadata = validate_training_checkpoint(path, 2)
            resume = load_training_checkpoint(
                path,
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                2,
                torch.device("cpu"),
            )

        self.assertEqual(metadata["epoch"], 3)
        self.assertEqual(resume["completed_epochs"], 3)
        self.assertEqual(resume["best_val_accuracy"], 0.75)
        self.assertEqual(resume["epoch_metrics"], [{"epoch": 3}])

    def test_training_checkpoint_rejects_class_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            model = build_resnet18_classifier(2)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
            save_training_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                epoch=1,
                best_val_accuracy=0.5,
                best_model_state_dict=model.state_dict(),
                class_names=["a", "b"],
                initialization="resnet18",
                epoch_metrics=[],
            )

            with self.assertRaises(ValueError):
                validate_training_checkpoint(path, 3)


if __name__ == "__main__":
    unittest.main()
