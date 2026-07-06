from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.model_quality import evaluation_results_summary


class ModelQualityTests(unittest.TestCase):
    def test_evaluation_results_summary_detects_missing_configured_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "evaluation_results.json"
            results_path.write_text(
                json.dumps([{"kind": "analysis", "id": "match_1_10_150", "status": "passed"}]) + "\n",
                encoding="utf-8",
            )
            summary = evaluation_results_summary(
                results_path,
                configured_ids=["match_1_10_150", "heatmap_match9"],
            )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["missing_configured_results"], ["heatmap_match9"])


if __name__ == "__main__":
    unittest.main()
