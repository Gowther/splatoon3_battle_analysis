from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.death_events import DEATH_EVENT_FIELDS, WEAPON_FIELDS, player_id_for_slot


ATTRIBUTION_FIELDS = DEATH_EVENT_FIELDS + [
    "attribution_status",
    "attribution_confidence",
    "attribution_evidence",
    "killer_candidates",
    "ocr_texts",
    "review_required",
]

TEXT_FIELDS = ("corrected_text", "ocr_text", "text", "recognized_text", "value")
EXPLICIT_KILLER_FIELDS = ("killer", "killer_name", "attacker", "attacker_name")
EXPLICIT_WEAPON_FIELDS = ("cause_weapon", "killer_weapon", "weapon")


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


def float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def weapon_matches_text(text: str, weapon_names: Sequence[str]) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    matches = []
    for weapon in weapon_names:
        weapon_key = normalize_text(weapon)
        if weapon_key and (weapon_key in normalized or normalized in weapon_key):
            matches.append(weapon)
    return sorted(set(matches), key=lambda item: (-len(normalize_text(item)), item))


def ocr_rows_by_event(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        event_id = str(row.get("event_id") or row.get("source_id") or "").strip()
        if event_id:
            grouped.setdefault(event_id, []).append(row)
    return grouped


def parse_time(row: Mapping[str, Any]) -> float | None:
    return float_or_none(row.get("time")) or float_or_none(row.get("elapsed_time"))


def nearest_analysis_row(rows: Sequence[Mapping[str, Any]], time_value: float | None) -> Mapping[str, Any] | None:
    if time_value is None or not rows:
        return None
    timed = [(parse_time(row), row) for row in rows]
    timed = [(time_item, row) for time_item, row in timed if time_item is not None]
    if not timed:
        return None
    return min(timed, key=lambda item: abs(float(item[0]) - time_value))[1]


def opposing_slots(victim_slot: int | None) -> list[int]:
    if victim_slot is None:
        return list(range(1, 9))
    return list(range(5, 9)) if victim_slot <= 4 else list(range(1, 5))


def weapon_slots_for_cause(
    cause_weapon: str,
    snapshot: Mapping[str, Any] | None,
    victim_slot: int | None,
) -> list[dict[str, Any]]:
    if not cause_weapon or snapshot is None:
        return []
    cause_key = normalize_text(cause_weapon)
    candidates: list[dict[str, Any]] = []
    for slot in opposing_slots(victim_slot):
        weapon = str(snapshot.get(WEAPON_FIELDS[slot - 1], "")).strip()
        weapon_key = normalize_text(weapon)
        if not weapon_key:
            continue
        if cause_key == weapon_key or cause_key in weapon_key or weapon_key in cause_key:
            candidates.append({"slot": slot, "player": player_id_for_slot(slot), "weapon": weapon})
    return candidates


def collect_ocr_evidence(rows: Sequence[Mapping[str, Any]], weapon_names: Sequence[str]) -> dict[str, Any]:
    texts = [first_text(row, TEXT_FIELDS) for row in rows]
    texts = [text for text in texts if text]
    explicit_killer = first_text({}, ())
    explicit_weapon = first_text({}, ())
    for row in rows:
        explicit_killer = explicit_killer or first_text(row, EXPLICIT_KILLER_FIELDS)
        explicit_weapon = explicit_weapon or first_text(row, EXPLICIT_WEAPON_FIELDS)
    matched_weapons: list[str] = []
    for text in texts:
        matched_weapons.extend(weapon_matches_text(text, weapon_names))
    return {
        "texts": texts,
        "explicit_killer": explicit_killer,
        "explicit_weapon": explicit_weapon,
        "matched_weapons": sorted(set(matched_weapons), key=lambda item: (-len(normalize_text(item)), item)),
    }


def choose_cause_weapon(event: Mapping[str, Any], evidence: Mapping[str, Any], weapon_names: Sequence[str]) -> tuple[str, str]:
    existing = str(event.get("cause_weapon", "")).strip()
    if existing:
        return existing, "event.cause_weapon"
    explicit = str(evidence.get("explicit_weapon", "")).strip()
    if explicit:
        matches = weapon_matches_text(explicit, weapon_names)
        return (matches[0] if matches else explicit), "ocr.explicit_weapon"
    matched = list(evidence.get("matched_weapons", []))
    if matched:
        return str(matched[0]), "ocr.text"
    return "", ""


def attribute_event(
    event: Mapping[str, Any],
    ocr_rows: Sequence[Mapping[str, Any]],
    analysis_rows: Sequence[Mapping[str, Any]],
    weapon_names: Sequence[str],
) -> dict[str, Any]:
    row = dict(event)
    event_id = str(row.get("event_id", ""))
    time_value = parse_time(row)
    victim_slot = int_or_none(row.get("victim_slot"))
    evidence_parts: list[str] = []

    ocr = collect_ocr_evidence(ocr_rows, weapon_names)
    ocr_texts = [str(text) for text in ocr.get("texts", [])]
    if ocr_texts:
        row["cause_text"] = row.get("cause_text") or ocr_texts[0]
        evidence_parts.append(f"ocr_texts={len(ocr_texts)}")
    else:
        evidence_parts.append("ocr_texts=0")

    cause_weapon, weapon_source = choose_cause_weapon(row, ocr, weapon_names)
    if cause_weapon:
        row["cause_weapon"] = cause_weapon
        evidence_parts.append(f"cause_weapon={cause_weapon} source={weapon_source}")

    explicit_killer = str(ocr.get("explicit_killer", "")).strip() or str(row.get("killer", "")).strip()
    if explicit_killer:
        row["killer"] = explicit_killer
        evidence_parts.append(f"killer={explicit_killer} source=explicit")

    snapshot = nearest_analysis_row(analysis_rows, time_value)
    if snapshot is not None:
        snapshot_time = parse_time(snapshot)
        if snapshot_time is not None:
            evidence_parts.append(f"weapon_snapshot_time={snapshot_time:.3f}")

    candidates = weapon_slots_for_cause(cause_weapon, snapshot, victim_slot)
    if candidates:
        evidence_parts.append(
            "killer_candidates="
            + ",".join(f"{candidate['player']}:{candidate['weapon']}" for candidate in candidates)
        )
    if len(candidates) == 1 and not row.get("killer"):
        candidate = candidates[0]
        row["killer_slot"] = candidate["slot"]
        row["killer"] = candidate["player"]
        row["killer_weapon"] = candidate["weapon"]
        evidence_parts.append("killer_slot=unique_weapon_match")
    elif len(candidates) == 1 and not row.get("killer_weapon"):
        row["killer_slot"] = candidates[0]["slot"]
        row["killer_weapon"] = candidates[0]["weapon"]

    confidence = 0.2
    if cause_weapon:
        confidence = 0.55
    if len(candidates) == 1:
        confidence += 0.25
    if explicit_killer:
        confidence += 0.2
    if ocr_texts:
        confidence += 0.05
    confidence = min(confidence, 0.95)

    if row.get("killer") and row.get("cause_weapon"):
        status = "attributed"
    elif row.get("cause_weapon"):
        status = "weapon_only"
    elif not ocr_texts:
        status = "no_ocr"
    else:
        status = "needs_review"

    row["attribution_status"] = status
    row["attribution_confidence"] = f"{confidence:.3f}"
    row["attribution_evidence"] = "; ".join(evidence_parts)
    row["killer_candidates"] = ";".join(f"{candidate['player']}:{candidate['weapon']}" for candidate in candidates)
    row["ocr_texts"] = " | ".join(ocr_texts)
    row["review_required"] = "false" if status == "attributed" else "true"
    if not event_id:
        row["attribution_evidence"] = "; ".join([row["attribution_evidence"], "missing_event_id"]).strip("; ")
    return row


def attribute_death_events(
    events: Sequence[Mapping[str, Any]],
    ocr_rows: Sequence[Mapping[str, Any]],
    analysis_rows: Sequence[Mapping[str, Any]],
    weapon_names: Sequence[str],
) -> dict[str, Any]:
    grouped_ocr = ocr_rows_by_event(ocr_rows)
    rows = [
        attribute_event(
            event,
            grouped_ocr.get(str(event.get("event_id", "")), []),
            analysis_rows,
            weapon_names,
        )
        for event in events
    ]
    by_status: dict[str, int] = {}
    for row in rows:
        status = str(row.get("attribution_status", ""))
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "status": "ready" if rows and by_status.get("attributed", 0) == len(rows) else "needs_review" if rows else "empty",
        "event_count": len(events),
        "attributed_count": by_status.get("attributed", 0),
        "by_status": by_status,
        "events": rows,
    }
