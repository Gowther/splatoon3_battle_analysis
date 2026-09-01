from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.csv_contracts import DEATH_EVENT_CSV_CONTRACT

PLAYER_STATE_FIELDS = tuple(f"player_state_{index}" for index in range(1, 9))
WEAPON_FIELDS = tuple(f"weapon_{index}" for index in range(1, 9))

# Older exports used class id 1; the current detector checkpoint maps `dead`
# to 3 and `special` to 20. Keep both dead ids for backwards compatibility.
DEFAULT_DEAD_STATE_IDS = ("1", "3")
DEFAULT_ALIVE_STATE_IDS = ("0", "14", "20")

DEAD_STATE_NAMES = {"dead", "death", "splatted", "splat", "map_player_dead"}
NON_DEAD_STATE_NAMES = {"alive", "live", "living", "special"}

DEATH_EVENT_FIELDS = list(DEATH_EVENT_CSV_CONTRACT.fields)


@dataclass(frozen=True)
class DeathEvent:
    event_id: str
    match_id: str
    time: float
    event: str
    team: str
    player: str
    killer: str
    victim: str
    clip_path: str
    segment_id: str
    victim_slot: int
    victim_weapon: str
    killer_slot: str
    killer_weapon: str
    cause_weapon: str
    cause_text: str
    confidence: float
    source: str
    evidence: str
    clip_start: float
    clip_end: float
    notes: str


def parse_time(row: Mapping[str, Any]) -> float | None:
    for key in ("elapsed_time", "time"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def canonical_state_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def normalize_state(
    value: Any,
    dead_state_ids: Sequence[Any] = DEFAULT_DEAD_STATE_IDS,
    alive_state_ids: Sequence[Any] = DEFAULT_ALIVE_STATE_IDS,
) -> str:
    if value in (None, ""):
        return "unknown"

    text = str(value).strip()
    lower = text.lower()
    if lower in DEAD_STATE_NAMES:
        return "dead"
    if lower in NON_DEAD_STATE_NAMES:
        return "alive"

    state_id = canonical_state_id(text)
    dead_ids = {canonical_state_id(item) for item in dead_state_ids if canonical_state_id(item)}
    alive_ids = {canonical_state_id(item) for item in alive_state_ids if canonical_state_id(item)}
    if state_id in dead_ids:
        return "dead"
    if state_id in alive_ids:
        return "alive"
    return "unknown"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def team_for_slot(slot: int, team_names: Sequence[str] | None = None) -> str:
    if team_names and len(team_names) >= 2:
        return str(team_names[0] if slot <= 4 else team_names[1])
    return "team_1" if slot <= 4 else "team_2"


def team_slot_for_slot(slot: int) -> int:
    return ((slot - 1) % 4) + 1


def player_id_for_slot(slot: int, team_names: Sequence[str] | None = None) -> str:
    return f"{team_for_slot(slot, team_names)}_slot_{team_slot_for_slot(slot)}"


def safe_identifier(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "match"


def death_event_id(match_id: str, slot: int, time_value: float) -> str:
    time_key = format_seconds(time_value).replace(".", "p")
    return f"death:{safe_identifier(match_id)}:slot{slot}:{time_key}"


def _timed_rows(
    rows: Sequence[Mapping[str, Any]],
    dead_state_ids: Sequence[Any],
    alive_state_ids: Sequence[Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        time_value = parse_time(row)
        if time_value is None:
            continue
        states = {
            slot: normalize_state(row.get(PLAYER_STATE_FIELDS[slot - 1]), dead_state_ids, alive_state_ids)
            for slot in range(1, 9)
        }
        output.append({"index": index, "time": time_value, "row": row, "states": states})
    output.sort(key=lambda item: (float(item["time"]), int(item["index"])))
    return output


def _dead_run_lengths(timed_rows: Sequence[Mapping[str, Any]]) -> list[dict[int, int]]:
    run_lengths = [{slot: 0 for slot in range(1, 9)} for _ in timed_rows]
    current = {slot: 0 for slot in range(1, 9)}
    for index in range(len(timed_rows) - 1, -1, -1):
        states = timed_rows[index]["states"]
        for slot in range(1, 9):
            if states[slot] == "dead":
                current[slot] += 1
            else:
                current[slot] = 0
            run_lengths[index][slot] = current[slot]
    return run_lengths


def extract_death_events(
    rows: Sequence[Mapping[str, Any]],
    match_id: str = "",
    dead_state_ids: Sequence[Any] = DEFAULT_DEAD_STATE_IDS,
    alive_state_ids: Sequence[Any] = DEFAULT_ALIVE_STATE_IDS,
    clip_before: float = 8.0,
    clip_after: float = 4.0,
    min_dead_frames: int = 1,
    include_initial_dead: bool = False,
    team_names: Sequence[str] | None = None,
) -> list[DeathEvent]:
    timed_rows = _timed_rows(rows, dead_state_ids, alive_state_ids)
    run_lengths = _dead_run_lengths(timed_rows)
    min_run = max(1, int(min_dead_frames))
    last_known = {slot: None for slot in range(1, 9)}
    last_raw = {slot: "" for slot in range(1, 9)}
    events: list[DeathEvent] = []

    for row_index, item in enumerate(timed_rows):
        row = item["row"]
        states = item["states"]
        time_value = float(item["time"])
        for slot in range(1, 9):
            field = PLAYER_STATE_FIELDS[slot - 1]
            state = states[slot]
            previous = last_known[slot]
            current_raw = str(row.get(field, "")).strip()

            if (
                state == "dead"
                and previous != "dead"
                and run_lengths[row_index][slot] >= min_run
                and (include_initial_dead or previous == "alive")
            ):
                team = team_for_slot(slot, team_names)
                player_id = player_id_for_slot(slot, team_names)
                weapon = str(row.get(WEAPON_FIELDS[slot - 1], "")).strip()
                run_bonus = min(0.2, 0.05 * (run_lengths[row_index][slot] - 1))
                event_index = len(events) + 1
                events.append(
                    DeathEvent(
                        event_id=death_event_id(match_id, slot, time_value),
                        match_id=match_id,
                        time=time_value,
                        event="death",
                        team=team,
                        player=player_id,
                        killer="",
                        victim=player_id,
                        clip_path="",
                        segment_id=f"{safe_identifier(match_id)}_death_{event_index:04d}",
                        victim_slot=slot,
                        victim_weapon=weapon,
                        killer_slot="",
                        killer_weapon="",
                        cause_weapon="",
                        cause_text=str(row.get("message", "")).strip(),
                        confidence=round(0.65 + run_bonus, 3),
                        source="player_state_transition",
                        evidence=f"{field}:{last_raw[slot] or previous or 'unknown'}->{current_raw or 'dead'}",
                        clip_start=max(0.0, time_value - float(clip_before)),
                        clip_end=time_value + float(clip_after),
                        notes="killer and cause pending OCR/attribution",
                    )
                )

            if state in {"alive", "dead"}:
                last_known[slot] = state
                last_raw[slot] = current_raw

    return events


def event_to_json_row(event: DeathEvent | Mapping[str, Any]) -> dict[str, Any]:
    row = asdict(event) if isinstance(event, DeathEvent) else dict(event)
    for field in ("time", "clip_start", "clip_end", "confidence"):
        if field in row and row[field] != "":
            row[field] = round(float(row[field]), 3)
    return row


def event_to_csv_row(event: DeathEvent | Mapping[str, Any]) -> dict[str, object]:
    row = event_to_json_row(event)
    for field in ("time", "clip_start", "clip_end"):
        if field in row and row[field] != "":
            row[field] = format_seconds(float(row[field]))
    if "confidence" in row and row["confidence"] != "":
        row["confidence"] = f"{float(row['confidence']):.3f}"
    return {field: row.get(field, "") for field in DEATH_EVENT_FIELDS}


def write_event_csv(path: Path, events: Iterable[DeathEvent | Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEATH_EVENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(event_to_csv_row(event))


def normalized_state_counts(
    rows: Sequence[Mapping[str, Any]],
    dead_state_ids: Sequence[Any] = DEFAULT_DEAD_STATE_IDS,
    alive_state_ids: Sequence[Any] = DEFAULT_ALIVE_STATE_IDS,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for field in PLAYER_STATE_FIELDS:
            counts[normalize_state(row.get(field), dead_state_ids, alive_state_ids)] += 1
    return dict(sorted(counts.items()))


def build_death_event_report(
    rows: Sequence[Mapping[str, Any]],
    match_id: str = "",
    events: Sequence[DeathEvent] | None = None,
    dead_state_ids: Sequence[Any] = DEFAULT_DEAD_STATE_IDS,
    alive_state_ids: Sequence[Any] = DEFAULT_ALIVE_STATE_IDS,
    team_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    resolved_events = list(events) if events is not None else extract_death_events(
        rows,
        match_id=match_id,
        dead_state_ids=dead_state_ids,
        alive_state_ids=alive_state_ids,
        team_names=team_names,
    )
    timed_row_count = sum(1 for row in rows if parse_time(row) is not None)
    slots = sorted({event.victim_slot for event in resolved_events})
    return {
        "status": "ready" if resolved_events else "empty",
        "blocking_reason": "" if resolved_events else "no player-state death transitions found",
        "match_id": match_id,
        "row_count": len(rows),
        "timed_row_count": timed_row_count,
        "event_count": len(resolved_events),
        "slots_with_death_events": slots,
        "state_counts": normalized_state_counts(rows, dead_state_ids, alive_state_ids),
        "events": [event_to_json_row(event) for event in resolved_events],
    }


def write_event_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
