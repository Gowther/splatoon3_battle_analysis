import json
import tempfile
import unittest
from pathlib import Path

from src.active_learning_workbench import (
    action_catalog,
    auto_record_llm_reviews,
    apply_staging_annotations,
    build_automation_plan_from_state,
    build_candidate_preannotation,
    build_llm_review_pack,
    command_for_action,
    finish_job_record,
    load_candidate_queue,
    load_jobs,
    prefill_candidate_staging,
    scan_asset_inbox,
    start_job_record,
    upsert_staging_annotation,
    validate_staging_item,
    write_json,
)
from scripts.serve_active_learning_workbench import APP_HTML


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

        by_id = {item["id"]: item for item in queue}
        self.assertEqual(len(queue), 2)
        self.assertEqual(by_id["ui:1"]["status"], "done")
        self.assertEqual(by_id["ui:1"]["llm_review"]["suggestion"], "player")
        self.assertTrue(any(item["target"] == "heatmap_tracker_labels" for item in queue))

    def test_candidate_queue_includes_death_ocr_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            death_csv = tmp / "death_ocr_candidates.csv"
            death_csv.write_text(
                "candidate_id,target,reason,source_id,match_id,video,elapsed_time,row_index,frame_path,ocr_text,details,event_id,region\n"
                "death:1,death_event_ocr,review_death_message,e1,m1,video.mp4,2.0,1,crop.jpg,Blaster,detail,e1,death_message_center\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            write_json(manifest, {"death_events": {"ocr_candidates_csv": str(death_csv)}})

            queue = load_candidate_queue(manifest, tmp / "staging.json", tmp / "reviews.json")

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["target"], "death_event_ocr")
        self.assertEqual(queue[0]["annotation_type"], "ocr_box_text")
        self.assertEqual(queue[0]["preannotation"]["annotation"]["text"], "Blaster")

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
        self.assertEqual(item["validation_status"], "ready")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["applied_count"], 1)
        self.assertEqual(report["skipped_count"], 0)

    def test_apply_staging_annotations_writes_death_event_labels(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            crop = tmp / "crop.jpg"
            crop.write_bytes(b"fake crop")
            staging = tmp / "staging.json"
            write_json(
                staging,
                {
                    "schema_version": 1,
                    "items": [
                        {
                            "id": "death:1",
                            "target": "death_event_ocr",
                            "annotation_type": "ocr_box_text",
                            "status": "done",
                            "candidate": {
                                "match_id": "m1",
                                "elapsed_time": "2.0",
                                "frame_path": str(crop),
                                "source_id": "e1",
                                "raw": {"event_id": "e1", "region": "death_message_center"},
                            },
                            "annotation": {"text": "Blaster", "notes": "killer=team_2_slot_1; cause_weapon=Blaster"},
                        }
                    ],
                },
            )
            death_labels = tmp / "death_labels.csv"

            report = apply_staging_annotations(
                staging_path=staging,
                dry_run=False,
                report_path=tmp / "apply.json",
                death_labels_path=death_labels,
            )
            death_label_text = death_labels.read_text(encoding="utf-8")

            self.assertEqual(report["skipped_count"], 0)
            self.assertEqual(report["applied_count"], 1)
            self.assertEqual(report["death_event_labels_csv"], str(death_labels))
            self.assertIn("Blaster", death_label_text)

    def test_candidate_queue_dedupes_and_prioritizes_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            candidates = tmp / "candidates.csv"
            candidates.write_text(
                "candidate_id,target,reason,source_id,match_id,video,elapsed_time,row_index,frame_path,details\n"
                "ui:1,ui_detector_yolo,missing_player_state,src,n_match,video.mp4,100.0,1,frame.jpg,detail\n"
                "ui:2,ui_detector_yolo,missing_player_state,src,n_match,video.mp4,100.8,2,frame2.jpg,detail\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            write_json(manifest, {"analysis": {"targets": {"ui_detector_yolo": {"csv": str(candidates), "rows": 2}}}})

            queue = load_candidate_queue(manifest, tmp / "staging.json", tmp / "reviews.json")
            full_queue = load_candidate_queue(manifest, tmp / "staging.json", tmp / "reviews.json", dedupe=False)

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["duplicate_count"], 2)
        self.assertEqual(len(full_queue), 2)

    def test_build_candidate_preannotation_from_raw_box_and_heatmap_point(self):
        box_preannotation = build_candidate_preannotation(
            {
                "target": "ui_detector_yolo",
                "annotation_type": "yolo_box",
                "raw": {"x1": "10", "y1": "20", "x2": "30", "y2": "60", "image_width": "100", "image_height": "100"},
            }
        )
        heatmap_preannotation = build_candidate_preannotation(
            {"target": "heatmap_tracker_labels", "raw": {"x": "123.45", "y": "67.89", "confidence": "0.9"}}
        )

        self.assertEqual(box_preannotation["status"], "ready")
        self.assertAlmostEqual(box_preannotation["annotation"]["boxes"][0]["x_center"], 0.2)
        self.assertEqual(heatmap_preannotation["annotation"]["point"]["x"], "123.5")
        self.assertFalse(heatmap_preannotation["needs_human"])

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

    def test_auto_record_llm_reviews_writes_rule_based_reviews(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            candidates = tmp / "candidates.csv"
            candidates.write_text(
                "candidate_id,target,reason,source_id,match_id,video,elapsed_time,row_index,frame_path,details\n"
                "ui:1,ui_detector_yolo,missing,src,n_match,video.mp4,1.0,1,,detail\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            reviews = tmp / "reviews.json"
            write_json(manifest, {"analysis": {"targets": {"ui_detector_yolo": {"csv": str(candidates), "rows": 1}}}})

            report = auto_record_llm_reviews(
                manifest_path=manifest,
                staging_path=tmp / "staging.json",
                reviews_path=reviews,
            )

        self.assertEqual(report["recorded_count"], 1)
        self.assertEqual(report["reviews"][0]["suggestion"], "skip_missing_image")

    def test_prefill_candidate_staging_creates_draft_from_heatmap_point(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            heatmap_csv = tmp / "heatmap.csv"
            heatmap_csv.write_text(
                "match_id,heatmap_id,anomaly_type,time,track_slot,x,y,confidence,exported_frame\n"
                "match_9,heatmap_match9,jump_reset,2.0,3,12.3,45.6,0.9,heat.jpg\n",
                encoding="utf-8",
            )
            manifest = tmp / "manifest.json"
            staging = tmp / "staging.json"
            write_json(manifest, {"heatmap": {"anomalies_csv": str(heatmap_csv)}})

            report = prefill_candidate_staging(
                target="heatmap_tracker_labels",
                manifest_path=manifest,
                staging_path=staging,
                reviews_path=tmp / "reviews.json",
            )

        self.assertEqual(report["prefilled_count"], 1)
        self.assertEqual(report["prefills"][0]["status"], "draft")

    def test_automation_plan_separates_runnable_steps_and_human_gates(self):
        plan = build_automation_plan_from_state(
            {
                "reports": [
                    {"id": "training_datasets", "status": "needs_data"},
                    {"id": "model_data_readiness", "status": "needs_data"},
                    {"id": "heatmap_labels", "status": "needs_labels"},
                ],
                "asset_inbox": {"videos": [{"status": "new", "path": "footages/a.mp4", "suggested_match_id": "a"}]},
                "queue_summary": {"by_status": {"todo": 2}},
                "staging_summary": {"by_status": {"done": 1}},
            }
        )

        self.assertGreaterEqual(plan["runnable_count"], 3)
        self.assertGreaterEqual(plan["human_gate_count"], 1)

    def test_job_records_can_start_and_finish(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            jobs_path = Path(tmp_name) / "jobs.json"
            job = start_job_record("refresh_training_candidates", {}, path=jobs_path)
            finished = finish_job_record(job["id"], {"status": "passed"}, path=jobs_path)
            jobs = load_jobs(jobs_path)

        self.assertEqual(finished["status"], "passed")
        self.assertEqual(jobs["jobs"][0]["id"], job["id"])

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

    def test_action_catalog_includes_chinese_labels(self):
        actions = {item["id"]: item for item in action_catalog()}

        self.assertEqual(actions["refresh_training_candidates"]["label_zh"], "刷新候选样本")
        self.assertEqual(actions["promotion_apply"]["description_zh"], "把已验证候选模型复制到登记的正式模型路径。")

    def test_workbench_html_defaults_to_chinese(self):
        self.assertIn('<html lang="zh-CN">', APP_HTML)
        self.assertIn("主动学习工作台", APP_HTML)
        self.assertIn('id="languageSelect"', APP_HTML)
        self.assertIn("死亡事件 OCR", APP_HTML)


if __name__ == "__main__":
    unittest.main()
