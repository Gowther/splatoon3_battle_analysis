import json
import tempfile
import unittest
from pathlib import Path

from src.active_learning_workbench import (
    apply_staging_annotations,
    build_llm_review_pack,
    command_for_action,
    load_candidate_queue,
    scan_asset_inbox,
    upsert_staging_annotation,
    validate_staging_item,
    write_json,
)


class ActiveLearningWorkbenchTests(unittest.TestCase):
    def test_candidate_queue_merges_candidates_staging_and_reviews(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            analysis_csv = tmp / "ui_candidates.csv"
            analysis_csv.write_text(
                "candidate_id,target,reason,source_id,match_id,video,elapsed_time,row_index,frame_path,details\n"
                "ui:1,ui_detector_yolo,missing,n_best,n_match,video.mp4,1.0,1,frame.jpg,detail\n",
                encoding="utf-8",
            )
            heatmap_csv = tmp / "heatmap.csv"
            heatmap_csv.write_text(
                "match_id,heatmap_id,anomaly_type,time,track_slot,exported_frame,note\n"
                "match_9,heatmap_match9,jump_reset,2.0,3,heat.jpg,review\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            write_json(
                manifest,
                {
                    "analysis": {"targets": {"ui_detector_yolo": {"csv": str(analysis_csv), "rows": 1}}},
                    "heatmap": {"anomalies_csv": str(heatmap_csv)},
                },
            )
            staging = tmp / "staging.json"
            write_json(
                staging,
                {
                    "schema_version": 1,
                    "items": [{"id": "ui:1", "status": "done", "target": "ui_detector_yolo"}],
                },
            )
            reviews = tmp / "reviews.json"
            write_json(
                reviews,
                {"schema_version": 1, "reviews": {"ui:1": {"suggestion": "player", "confidence": 0.7}}},
            )

            queue = load_candidate_queue(manifest, staging, reviews)

        self.assertEqual(len(queue), 2)
        self.assertEqual(queue[0]["status"], "done")
        self.assertEqual(queue[0]["llm_review"]["suggestion"], "player")
        self.assertEqual(queue[1]["target"], "heatmap_tracker_labels")

    def test_upsert_annotation_and_apply_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            image = tmp / "frame.jpg"
            image.write_bytes(b"fake image bytes")
            staging = tmp / "staging.json"
            item = upsert_staging_annotation(
                {
                    "id": "ui:1",
                    "target": "ui_detector_yolo",
                    "annotation_type": "yolo_box",
                    "status": "done",
                    "split": "train",
                    "candidate": {"frame_path": str(image), "target": "ui_detector_yolo"},
                    "annotation": {
                        "boxes": [
                            {
                                "class_id": 0,
                                "x_center": 0.5,
                                "y_center": 0.5,
                                "width": 0.2,
                                "height": 0.3,
                            }
                        ]
                    },
                },
                staging_path=staging,
            )
            report = apply_staging_annotations(
                staging_path=staging,
                dry_run=True,
                report_path=tmp / "apply_report.json",
            )

        self.assertEqual(item["status"], "done")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["applied_count"], 1)
        self.assertEqual(report["skipped_count"], 0)

    def test_validate_staging_item_rejects_bad_box(self):
        errors = validate_staging_item(
            {
                "id": "bad",
                "status": "done",
                "target": "ui_detector_yolo",
                "candidate": {"frame_path": __file__},
                "annotation": {"boxes": [{"class_id": -1, "x_center": 2, "y_center": 0.5, "width": 0, "height": 1}]},
            }
        )

        self.assertGreaterEqual(len(errors), 2)

    def test_build_llm_review_pack_writes_tasks(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            candidates = tmp / "candidates.csv"
            candidates.write_text(
                "candidate_id,target,reason,source_id,match_id,video,elapsed_time,row_index,frame_path,details\n"
                "ui:1,ui_detector_yolo,missing,n_best,n_match,video.mp4,1.0,1,frame.jpg,detail\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            write_json(manifest, {"analysis": {"targets": {"ui_detector_yolo": {"csv": str(candidates), "rows": 1}}}})
            pack_path = tmp / "pack.json"

            pack = build_llm_review_pack(
                manifest_path=manifest,
                staging_path=tmp / "staging.json",
                reviews_path=tmp / "reviews.json",
                output_path=pack_path,
            )
            self.assertTrue(pack_path.exists())

        self.assertEqual(pack["status"], "ready")
        self.assertEqual(pack["tasks"][0]["id"], "ui:1")

    def test_command_for_action_requires_confirm_for_dangerous_actions(self):
        with self.assertRaises(ValueError):
            command_for_action("training_execute", {"target": "ui_detector_yolo"})
        command = command_for_action(
            "training_execute",
            {"target": "ui_detector_yolo", "confirm": "execute_training", "python": ".venv/bin/python"},
        )
        self.assertIn("--execute", command)

    def test_scan_asset_inbox_finds_unregistered_footage(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            footages = tmp / "footages"
            footages.mkdir()
            (footages / "n_match_6.mp4").write_bytes(b"")
            registry = tmp / "data_registry.json"
            registry.write_text(json.dumps({"matches": []}), encoding="utf-8")

            inbox = scan_asset_inbox(root=tmp, registry_path=registry)

        self.assertEqual(inbox["status"], "needs_intake")
        self.assertEqual(inbox["new_count"], 1)


if __name__ == "__main__":
    unittest.main()
