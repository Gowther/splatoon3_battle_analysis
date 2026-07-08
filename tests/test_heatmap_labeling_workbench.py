from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.heatmap.labeling_workbench import (
    build_labeling_workbench_report,
    priority_queue_rows,
    render_markdown,
    workbench_status,
    write_priority_queue,
)


class HeatmapLabelingWorkbenchTests(unittest.TestCase):
    def test_priority_queue_rows_use_stable_fields_and_limit(self) -> None:
        rows = priority_queue_rows(
            [
                {"annotation_id": "a1", "match_id": "m1", "time": "1.0", "team": "blue"},
                {"annotation_id": "a2", "match_id": "m2", "time": "2.0", "team": "orange"},
            ],
            limit=1,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["annotation_id"], "a1")
        self.assertIn("preview_path", rows[0])

    def test_write_priority_queue_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "priority.csv"

            report = write_priority_queue(path, [{"annotation_id": "a1", "match_id": "m1"}], limit=10)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(report["rows"], 1)
        self.assertEqual(rows[0]["annotation_id"], "a1")

    def test_workbench_status_ready_to_label_when_tasks_exist(self) -> None:
        status = workbench_status({"progress": {"status": "needs_labels"}, "label_readiness": {}}, priority_rows=3)

        self.assertEqual(status, "ready_to_label")

    def test_build_report_includes_commands_and_nested_round_report(self) -> None:
        report = build_labeling_workbench_report(
            {
                "status": "needs_labels",
                "annotation_csv": "outputs/round/annotation_template.csv",
                "package_dir": "outputs/round",
                "progress": {"labeled_rows": 0},
                "label_readiness": {"status": "needs_labels"},
            },
            priority_queue={"path": "outputs/round/priority_queue.csv", "rows": 2, "limit": 2},
            annotation_ui={"output_html": "outputs/round/annotation_ui.html"},
        )
        markdown = render_markdown(report)

        self.assertEqual(report["status"], "ready_to_label")
        self.assertIn("evaluate_heatmap_annotations.py", report["next_commands"]["evaluate"])
        self.assertIn("priority_rows: 2", markdown)
        self.assertIn("round_report", report)


if __name__ == "__main__":
    unittest.main()
