from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.core.paths import project_path
from src.experiment_manifest import file_record, sha256_file


def stage_metadata_path(csv_path: Path | str) -> Path:
    path = project_path(csv_path)
    return path.with_name(path.name + ".meta.json")


def write_stage_metadata(csv_path: Path | str, metadata: Mapping[str, Any]) -> Path:
    path = stage_metadata_path(csv_path)
    payload = {"schema_version": 1, **metadata, "artifact": file_record("stage_tracks", csv_path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def stage_artifact_status(
    csv_path: Path | str, *, match_id: str | None = None, stage_id: str | None = None
) -> dict[str, Any]:
    path = project_path(csv_path)
    if not path.is_file():
        return {"status": "no_points", "reason": "stage CSV is missing"}
    try:
        metadata = json.loads(stage_metadata_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "needs_calibration", "reason": "stage metadata is missing or unreadable"}
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        return {"status": "needs_calibration", "reason": "unsupported stage metadata"}
    quality_gate = metadata.get("quality_gate")
    if (
        metadata.get("status") != "ready"
        or metadata.get("quality") != "calibrated"
        or metadata.get("method") != "homography"
        or not isinstance(quality_gate, dict)
        or quality_gate.get("status") != "ready"
    ):
        return {"status": "needs_calibration", "reason": "calibrated coordinates with passing geometry checks are required"}
    if not metadata.get("stage_id") or (stage_id is not None and metadata["stage_id"] != stage_id):
        return {"status": "stage_mismatch", "reason": "stage id does not match the requested stage"}
    if match_id is not None and metadata.get("match_id") != match_id:
        return {"status": "match_mismatch", "reason": "match id does not match the registry"}
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("sha256") != sha256_file(path):
        return {"status": "stale_metadata", "reason": "stage CSV changed after calibration"}
    return {"status": "ready", "metadata": metadata}
