from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEATH_REVIEW_FIELDS = [
    "candidate_id",
    "event_id",
    "match_id",
    "time",
    "region",
    "frame_path",
    "source_frame_path",
    "corrected_text",
    "killer",
    "cause_weapon",
    "killer_weapon",
    "cause_text",
    "confidence",
    "notes",
    "updated_at",
]


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_notes(notes: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for part in re.split(r"[;\n]+", notes or ""):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key in {"killer", "cause_weapon", "killer_weapon", "cause_text", "confidence"}:
            output[key] = value.strip()
    return output


def staging_item_to_death_review_row(item: Mapping[str, Any]) -> dict[str, Any]:
    candidate = item.get("candidate", {}) if isinstance(item.get("candidate"), dict) else {}
    raw = candidate.get("raw", {}) if isinstance(candidate.get("raw"), dict) else {}
    annotation = item.get("annotation", {}) if isinstance(item.get("annotation"), dict) else {}
    notes = str(annotation.get("notes") or raw.get("notes") or raw.get("details") or "").strip()
    parsed = parse_notes(notes)
    corrected_text = str(annotation.get("text") or raw.get("corrected_text") or raw.get("ocr_text") or "").strip()
    return {
        "candidate_id": item.get("id", "") or candidate.get("id", "") or raw.get("candidate_id", ""),
        "event_id": raw.get("event_id") or candidate.get("source_id", ""),
        "match_id": candidate.get("match_id", "") or raw.get("match_id", ""),
        "time": candidate.get("elapsed_time", "") or raw.get("elapsed_time", "") or raw.get("time", ""),
        "region": raw.get("region", ""),
        "frame_path": candidate.get("frame_path", "") or raw.get("frame_path", ""),
        "source_frame_path": raw.get("source_frame_path", ""),
        "corrected_text": corrected_text,
        "killer": parsed.get("killer", raw.get("killer", "")),
        "cause_weapon": parsed.get("cause_weapon", raw.get("cause_weapon", "")),
        "killer_weapon": parsed.get("killer_weapon", raw.get("killer_weapon", "")),
        "cause_text": parsed.get("cause_text", corrected_text),
        "confidence": parsed.get("confidence", raw.get("confidence", "")),
        "notes": notes,
        "updated_at": item.get("updated_at", ""),
    }


def merge_review_rows(existing: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {str(row.get("candidate_id", "")): dict(row) for row in existing if row.get("candidate_id")}
    for row in incoming:
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id:
            merged[candidate_id] = dict(row)
    return [merged[key] for key in sorted(merged)]


def build_death_annotation_report(
    labels: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]] | None = None,
    attributed_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = list(candidates or [])
    attributed_events = list(attributed_events or [])
    candidate_ids = {str(row.get("candidate_id", "")) for row in candidates if row.get("candidate_id")}
    label_ids = {str(row.get("candidate_id", "")) for row in labels if row.get("candidate_id")}
    event_ids = {str(row.get("event_id", "")) for row in labels if row.get("event_id")}
    missing = sorted(candidate_ids - label_ids)
    label_count = len(labels)
    candidate_count = len(candidates)
    return {
        "schema_version": 1,
        "status": "ready" if label_count and not missing else "needs_review" if candidate_count else "empty",
        "label_count": label_count,
        "candidate_count": candidate_count,
        "coverage_ratio": round(label_count / candidate_count, 4) if candidate_count else 0.0,
        "unique_event_labels": len(event_ids),
        "labels_with_corrected_text": sum(1 for row in labels if row.get("corrected_text")),
        "labels_with_cause_weapon": sum(1 for row in labels if row.get("cause_weapon")),
        "labels_with_killer": sum(1 for row in labels if row.get("killer")),
        "attributed_event_count": len(attributed_events),
        "missing_candidate_count": len(missing),
        "missing_candidate_ids": missing[:100],
    }
