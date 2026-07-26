from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.outputs_retention import (
    annotation_referenced_dirs,
    apply_retention_plan,
    build_retention_plan,
    find_regenerable_dirs,
    format_size,
    render_markdown,
)


def write_file(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def write_annotation_template(path: Path, frame_dir: str, *, filled: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation_id", "x", "y", "frame_path"])
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "annotation_id": f"a{index}",
                    "x": "100" if filled else "",
                    "y": "200" if filled else "",
                    "frame_path": f"{frame_dir}/frame_{index}.jpg",
                }
            )


class OutputsRetentionTests(unittest.TestCase):
    def test_format_size_uses_binary_units(self) -> None:
        self.assertEqual(format_size(512), "512.0B")
        self.assertEqual(format_size(2048), "2.0KB")
        self.assertEqual(format_size(5 * 1024 * 1024), "5.0MB")

    def test_find_regenerable_dirs_skips_deliverables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            write_file(outputs / "heatmap_a" / "frames" / "f.jpg")
            write_file(outputs / "heatmap_a" / "debug_markers" / "d.jpg")
            write_file(outputs / "heatmap_a" / "rendered" / "heatmap.png")
            write_file(outputs / "heatmap_a" / "player_routes" / "p.png")

            found = find_regenerable_dirs(outputs)

            self.assertEqual({item["kind"] for item in found}, {"frames", "debug_markers"})

    def test_plan_reports_reclaimable_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            write_file(outputs / "heatmap_a" / "frames" / "f.jpg", size=100)
            write_file(outputs / "heatmap_a" / "rendered" / "keep.png", size=50)

            plan = build_retention_plan(outputs)

            self.assertEqual(plan["status"], "reclaimable")
            self.assertEqual(plan["total_bytes"], 150)
            self.assertEqual(plan["reclaimable_bytes"], 100)
            self.assertEqual(len(plan["candidates"]), 1)

    def test_plan_is_clean_without_regenerable_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            write_file(outputs / "heatmap_a" / "rendered" / "keep.png")

            plan = build_retention_plan(outputs)

            self.assertEqual(plan["status"], "clean")
            self.assertEqual(plan["candidates"], [])

    def test_min_size_threshold_skips_small_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            write_file(outputs / "heatmap_a" / "frames" / "f.jpg", size=10)

            plan = build_retention_plan(outputs, min_size_bytes=1024)

            self.assertEqual(plan["candidates"], [])
            self.assertEqual(len(plan["skipped_below_threshold"]), 1)

    def test_unfinished_annotation_round_holds_its_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            frames = outputs / "round1" / "frames"
            write_file(frames / "frame_0.jpg")
            write_annotation_template(
                outputs / "round1" / "annotation_template.csv",
                str(frames),
                filled=False,
            )

            referenced = annotation_referenced_dirs(outputs)

            self.assertIn(str(frames), referenced)

    def test_completed_annotation_round_does_not_hold_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            frames = outputs / "round1" / "frames"
            write_file(frames / "frame_0.jpg")
            write_annotation_template(
                outputs / "round1" / "annotation_template.csv",
                str(frames),
                filled=True,
            )

            self.assertEqual(annotation_referenced_dirs(outputs), {})

    def test_dry_run_apply_removes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            target = outputs / "heatmap_a" / "frames"
            write_file(target / "f.jpg")
            plan = build_retention_plan(outputs)

            result = apply_retention_plan(plan, dry_run=True)

            self.assertTrue(result["dry_run"])
            self.assertEqual(result["removed_count"], 1)
            self.assertTrue(target.exists())

    def test_apply_removes_only_planned_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            frames = outputs / "heatmap_a" / "frames"
            rendered = outputs / "heatmap_a" / "rendered"
            write_file(frames / "f.jpg")
            write_file(rendered / "keep.png")
            plan = build_retention_plan(outputs)

            result = apply_retention_plan(plan, dry_run=False)

            self.assertFalse(frames.exists())
            self.assertTrue(rendered.exists())
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["removed_count"], 1)

    def test_markdown_lists_candidates_and_held_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp)
            held_frames = outputs / "round1" / "frames"
            write_file(outputs / "heatmap_a" / "frames" / "f.jpg")
            write_file(held_frames / "frame_0.jpg")
            write_annotation_template(
                outputs / "round1" / "annotation_template.csv",
                str(held_frames),
                filled=False,
            )
            plan = build_retention_plan(outputs)

            markdown = render_markdown(plan, apply_retention_plan(plan, dry_run=True))

            self.assertIn("# Outputs Retention", markdown)
            self.assertIn("Held For Annotation", markdown)
            self.assertIn("heatmap_a/frames", markdown)
            self.assertEqual(len(plan["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
