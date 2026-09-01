from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.paths import ROOT, display_path, project_path
from src.csv_contracts import (
    ANALYSIS_CSV_CONTRACT,
    CLEAN_POINT_CSV_CONTRACT,
    CORE_CSV_CONTRACTS,
    DEATH_EVENT_CSV_CONTRACT,
    DEATH_POSITION_CSV_CONTRACT,
    PLAYER_TRACK_CSV_CONTRACT,
    RAW_MARKER_CSV_CONTRACT,
    REJECTED_POINT_CSV_CONTRACT,
    STAGE_PLAYER_TRACK_CSV_CONTRACT,
    TRACK_CSV_CONTRACT,
    TRACK_GAP_CSV_CONTRACT,
    CsvContract,
)
from src.experiment_manifest import file_record
from src.model_registry import DEFAULT_MODEL_REGISTRY, build_model_registry_report, load_model_registry


ARTIFACT_CONTRACTS: Dict[str, CsvContract] = {
    "state_join.state_csv": ANALYSIS_CSV_CONTRACT,
    "outputs.raw_points_csv": RAW_MARKER_CSV_CONTRACT,
    "outputs.clean_points_csv": CLEAN_POINT_CSV_CONTRACT,
    "outputs.rejected_points_csv": REJECTED_POINT_CSV_CONTRACT,
    "outputs.tracks_csv": TRACK_CSV_CONTRACT,
    "outputs.player_tracks_csv": PLAYER_TRACK_CSV_CONTRACT,
    "outputs.player_tracks_stage_csv": STAGE_PLAYER_TRACK_CSV_CONTRACT,
    "outputs.player_track_gaps_csv": TRACK_GAP_CSV_CONTRACT,
    "outputs.death_events_csv": DEATH_EVENT_CSV_CONTRACT,
    "outputs.death_positions_csv": DEATH_POSITION_CSV_CONTRACT,
    "stage_tracks": STAGE_PLAYER_TRACK_CSV_CONTRACT,
    "death_positions.event_csv": DEATH_EVENT_CSV_CONTRACT,
    "death_positions.position_csv": DEATH_POSITION_CSV_CONTRACT,
}


def artifact_record(label: str, path: Path) -> Dict[str, object]:
    record = file_record(label, path)
    contract = ARTIFACT_CONTRACTS.get(label)
    if contract is None:
        return record
    record["csv_contract"] = contract.manifest_record()
    if not path.exists():
        record["contract_status"] = "missing"
        return record
    if not path.is_file():
        record["contract_status"] = "invalid_kind"
        return record
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle), [])
        contract.validate_header(fields)
    except ValueError:
        record["contract_status"] = "mismatch"
        record["actual_columns"] = fields
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        record["contract_status"] = "unreadable"
        record["contract_error"] = str(exc)
    else:
        record["contract_status"] = "passed"
    return record


def artifact_status(config: Dict, extra_paths: Dict[str, Path]) -> List[Dict[str, object]]:
    artifacts: List[Dict[str, object]] = []
    output_values = {
        **{
            f"outputs.{key}": value
            for key, value in config.get("outputs", {}).items()
            if isinstance(value, str)
        },
        "state_join.state_csv": config.get("state_join", {}).get("state_csv", ""),
    }
    for label, value in output_values.items():
        if value:
            artifacts.append(artifact_record(label, project_path(value)))
    for label, path in extra_paths.items():
        artifacts.append(artifact_record(label, path))
    return artifacts


def stable_fingerprint(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_snapshot() -> Dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"status": "unavailable", "commit": "", "dirty": False, "error": str(exc)}
    if revision.returncode != 0 or status.returncode != 0:
        return {"status": "unavailable", "commit": "", "dirty": False}
    return {
        "status": "ready",
        "commit": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def runtime_model_report(args: argparse.Namespace) -> Dict[str, Any]:
    if args.skip_ui_analysis or args.only_report:
        return {
            "schema_version": 1,
            "status": "not_used",
            "reason": "existing UI-state artifacts were reused",
            "models": [],
        }
    return build_model_registry_report(load_model_registry(), verify_hash=True)


def write_run_manifest(
    config: Dict,
    args: argparse.Namespace,
    *,
    source_config_path: Path,
    resolved_config_path: Path,
    color_report_path: Path,
    report_path: Path,
    command_hint: str,
    cleaned_paths: List[str],
    stage_normalization: Optional[Dict[str, Any]] = None,
    stage_rendering: Optional[Dict[str, Any]] = None,
    death_positions: Optional[Dict[str, Any]] = None,
    model_report: Optional[Dict[str, Any]] = None,
) -> Path:
    output_dir = project_path(config["match"]["output_dir"])
    manifest_path = output_dir / "run_manifest.json"
    extra_artifacts = {
        "resolved_config": resolved_config_path,
        "color_report": color_report_path,
        "report": report_path,
    }
    if stage_normalization and stage_normalization.get("output"):
        extra_artifacts["stage_tracks"] = project_path(stage_normalization["output"])
    if stage_rendering and stage_rendering.get("output_dir"):
        extra_artifacts["stage_rendering"] = project_path(stage_rendering["output_dir"])
    if death_positions:
        for key in ("event_csv", "event_json", "position_csv", "report_json"):
            if death_positions.get(key):
                extra_artifacts[f"death_positions.{key}"] = project_path(death_positions[key])
    options = {
        "device": args.device,
        "warmup_frames": args.warmup_frames,
        "contact_limit": args.contact_limit,
        "skip_ui_analysis": args.skip_ui_analysis,
        "only_report": args.only_report,
        "clean_output": args.clean_output,
        "event_csv": args.event_csv or "",
        "teams": args.teams or "",
        "disable_auto_colors": args.disable_auto_colors,
    }
    inputs = {
        "input_video": file_record("input_video", config["match"]["input_video"]),
        "source_config": file_record("source_config", project_path(source_config_path)),
        "resolved_config": file_record("resolved_config", resolved_config_path),
        "model_registry": file_record("model_registry", DEFAULT_MODEL_REGISTRY),
    }
    models = model_report or {"schema_version": 1, "status": "not_captured", "models": []}
    code = git_snapshot()
    contracts = [contract.manifest_record() for contract in CORE_CSV_CONTRACTS]
    artifacts = artifact_status(config, extra_artifacts)
    contract_mismatches = [
        str(item["label"])
        for item in artifacts
        if item.get("contract_status") in {"mismatch", "invalid_kind", "unreadable"}
    ]
    provenance_issues = [
        f"missing_input:{label}"
        for label, record in inputs.items()
        if not record.get("exists")
    ]
    if models.get("status") not in {"passed", "not_used"}:
        provenance_issues.append(f"models:{models.get('status', 'unknown')}")
    provenance_issues.extend(f"csv_contract:{label}" for label in contract_mismatches)
    fingerprint_payload = {
        "match_id": config["match"]["id"],
        "code": code,
        "inputs": inputs,
        "models": models,
        "options": options,
        "contracts": contracts,
    }
    manifest = {
        "schema_version": 2,
        "status": "needs_review" if provenance_issues else "ready",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_fingerprint": stable_fingerprint(fingerprint_payload),
        "match_id": config["match"]["id"],
        "input_video": config["match"]["input_video"],
        "source_config": display_path(project_path(source_config_path)),
        "resolved_config": display_path(resolved_config_path),
        "color_report": display_path(color_report_path),
        "report": display_path(report_path),
        "command": command_hint,
        "options": options,
        "code": code,
        "inputs": inputs,
        "models": models,
        "csv_contracts": contracts,
        "contract_mismatches": contract_mismatches,
        "provenance_issues": provenance_issues,
        "cleaned_paths": cleaned_paths,
        "stage_normalization": stage_normalization or {"status": "no_asset", "method": "", "output": ""},
        "stage_rendering": stage_rendering or {"status": "skipped", "rendered": {}},
        "death_positions": death_positions or {"status": "empty", "event_count": 0},
        "artifacts": artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path
