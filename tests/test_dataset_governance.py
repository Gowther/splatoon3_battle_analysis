from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataset_governance import build_dataset_governance_report, registry_metadata_issues, render_markdown


def write_registry(path: Path, matches: list[dict]) -> None:
    path.write_text(json.dumps({"matches": matches}) + "\n", encoding="utf-8")


def make_dataset(root: Path, counts: dict[str, int]) -> None:
    for label, count in counts.items():
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (class_dir / f"{index}.png").write_bytes(b"image")


class DatasetGovernanceTests(unittest.TestCase):
    def test_build_dataset_governance_report_passes_aligned_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.json"
            dataset = root / "dataset"
            labels = root / "labels.txt"
            write_registry(
                registry,
                [
                    {
                        "id": "match_1",
                        "video": "footages/match_1.mp4",
                        "purpose": ["analysis_candidate"],
                        "notes": "sample",
                        "analysis_windows": [
                            {
                                "id": "match_1_10_20",
                                "start_seconds": 10,
                                "stop_seconds": 20,
                                "sample_fps": 5,
                                "device": "cpu",
                            }
                        ],
                    }
                ],
            )
            make_dataset(dataset, {"A": 2, "B": 2})
            labels.write_text("A\nB\n", encoding="utf-8")

            report = build_dataset_governance_report(
                registry_path=registry,
                dataset=dataset,
                labels=labels,
                model=None,
                min_images_per_class=2,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summary"]["small_class_count"], 0)

    def test_registry_metadata_issues_flags_missing_fields(self) -> None:
        issues = registry_metadata_issues({"matches": [{"id": "bad", "analysis_windows": [{"id": "bad_1"}]}]})

        fields = {issue["field"] for issue in issues}
        self.assertIn("purpose", fields)
        self.assertIn("notes", fields)
        self.assertIn("analysis_windows.start_seconds", fields)

    def test_render_markdown_includes_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "registry.json"
            dataset = root / "dataset"
            labels = root / "labels.txt"
            write_registry(registry, [])
            make_dataset(dataset, {"A": 1})
            labels.write_text("A\n", encoding="utf-8")
            report = build_dataset_governance_report(
                registry_path=registry,
                dataset=dataset,
                labels=labels,
                model=None,
                min_images_per_class=1,
            )

        self.assertIn("# Dataset Governance Report", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
