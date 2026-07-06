from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_COLUMNS = 33
COUNT_JUMP_THRESHOLD = 20


def populated(row: dict, key: str) -> bool:
    return bool(row.get(key))


def int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def count_jump_warnings(rows: list[dict]) -> list[str]:
    warnings: list[str] = []
    previous: dict[str, tuple[str, int]] = {}
    for row in rows:
        elapsed = row.get("elapsed_time", "")
        for key in ("count_left", "count_right", "penalty_left", "penalty_right"):
            value = int_or_none(row.get(key))
            if value is None:
                continue
            if key in previous:
                prev_elapsed, prev_value = previous[key]
                if abs(value - prev_value) > COUNT_JUMP_THRESHOLD:
                    warnings.append(f"{key} jumps {prev_value}->{value} between {prev_elapsed}s and {elapsed}s")
            previous[key] = (elapsed, value)
    return warnings


def quality_warnings(rows: list[dict], fieldnames: list[str] | None) -> list[str]:
    warnings: list[str] = []
    if fieldnames and len(fieldnames) != EXPECTED_COLUMNS:
        warnings.append(f"CSV has {len(fieldnames)} columns, expected {EXPECTED_COLUMNS}")

    first_weapon_index = next((index for index, row in enumerate(rows) if populated(row, "weapon_1")), None)
    if first_weapon_index is not None:
        missing_after_warmup = sum(1 for row in rows[first_weapon_index:] if not populated(row, "weapon_1"))
        if missing_after_warmup:
            warnings.append(f"weapon_1 is empty in {missing_after_warmup} rows after first weapon row")

    no_player_state_rows = sum(
        1
        for row in rows
        if not any(populated(row, f"player_state_{i}") for i in range(1, 9))
    )
    if no_player_state_rows:
        warnings.append(f"{no_player_state_rows} rows have no player state columns populated")

    warnings.extend(count_jump_warnings(rows))
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a Splatoon analysis CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when quality warnings are found.")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        print(f"{csv_path}: empty")
        return 0

    eight_state_rows = sum(1 for row in rows if all(populated(row, f"player_state_{i}") for i in range(1, 9)))
    weapon_rows = sum(1 for row in rows if populated(row, "weapon_1"))
    count_rows = sum(1 for row in rows if populated(row, "count_left") or populated(row, "count_right"))
    penalty_rows = sum(1 for row in rows if populated(row, "penalty_left") or populated(row, "penalty_right"))
    object_rows = sum(
        1
        for row in rows
        if row.get("asari_count") != "0"
        or row.get("hoko_count") != "0"
        or row.get("area_count") != "0"
        or row.get("yagura_count") != "0"
    )
    player_rows = sum(1 for row in rows if row.get("player_detected") == "True")
    message_rows = sum(1 for row in rows if populated(row, "message"))
    first_weapon_row = next((row for row in rows if populated(row, "weapon_1")), None)

    print(f"file: {csv_path}")
    print(f"rows: {len(rows)}")
    print(f"elapsed: {rows[0].get('elapsed_time')} -> {rows[-1].get('elapsed_time')}")
    print(f"8-player state rows: {eight_state_rows}")
    print(f"weapon rows: {weapon_rows}")
    print(f"count rows: {count_rows}")
    print(f"penalty rows: {penalty_rows}")
    print(f"objective rows: {object_rows}")
    print(f"player rows: {player_rows}")
    print(f"message rows: {message_rows}")
    if first_weapon_row:
        weapons = [first_weapon_row.get(f"weapon_{i}", "") for i in range(1, 9)]
        print(f"weapons: {weapons}")

    count_samples = [
        (
            row.get("elapsed_time"),
            row.get("count_left"),
            row.get("count_right"),
            row.get("penalty_left"),
            row.get("penalty_right"),
        )
        for row in rows
        if populated(row, "count_left") or populated(row, "count_right")
    ][:10]
    if count_samples:
        print(f"first count samples: {count_samples}")

    message_samples = [(row.get("elapsed_time"), row.get("message")) for row in rows if populated(row, "message")][:10]
    if message_samples:
        print(f"message samples: {message_samples}")

    warnings = quality_warnings(rows, fieldnames)
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
