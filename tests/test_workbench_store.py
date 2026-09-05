import json
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.active_learning_workbench import (
    finish_job_record,
    load_jobs,
    load_staging,
    start_job_record,
    upsert_staging_annotation,
)
from src.workbench_store import json_transaction, read_json, write_json


def increment_counter(path):
    with json_transaction(path, lambda target: read_json(target, {"count": 0})) as payload:
        payload["count"] += 1


class WorkbenchStoreTests(unittest.TestCase):
    def test_concurrent_process_updates_keep_every_increment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counter.json"
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
                list(pool.map(increment_counter, [path] * 24))
            self.assertEqual(read_json(path, {})["count"], 24)

    def test_concurrent_annotations_are_all_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "staging.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(upsert_staging_annotation, {"id": f"point_{i}", "status": "draft"}, staging_path=path)
                    for i in range(24)
                ]
                saved = [future.result() for future in futures]
            self.assertEqual({item["id"] for item in load_staging(path)["items"]}, {item["id"] for item in saved})

    def test_same_second_jobs_have_unique_ids_and_independent_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            with patch("src.active_learning_workbench.utc_now", return_value="2026-09-06T00:00:00+00:00"):
                jobs = [start_job_record("refresh_training_candidates", {"limit": i}, path=path) for i in range(16)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(finish_job_record, job["id"], {"status": "passed", "index": i}, path) for i, job in enumerate(jobs)]
                for future in futures:
                    future.result()
            stored = {job["id"]: job for job in load_jobs(path)["jobs"]}
            self.assertEqual(len(stored), 16)
            for i, job in enumerate(jobs):
                self.assertEqual(stored[job["id"]]["result"]["index"], i)
                self.assertEqual(stored[job["id"]]["payload"]["limit"], i)

    def test_failed_serialization_preserves_the_previous_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            write_json(path, {"previous": True})
            with self.assertRaises(TypeError):
                write_json(path, {"invalid": {1, 2}})
            self.assertEqual(json.loads(path.read_text()), {"previous": True})
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_failed_transaction_rolls_back_and_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counter.json"
            write_json(path, {"count": 4})
            with self.assertRaises(ValueError):
                with json_transaction(path, lambda target: read_json(target, {})) as payload:
                    payload["count"] = 0
                    raise ValueError("cancelled")
            increment_counter(path)
            self.assertEqual(read_json(path, {})["count"], 5)
