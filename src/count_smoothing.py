from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


COUNT_FIELDS = ("count_left", "count_right", "penalty_left", "penalty_right")


@dataclass(frozen=True)
class CountSmoothingConfig:
    max_jump: int = 20
    neighbor_tolerance: int = 3
    lookahead: int = 3


@dataclass
class CountCorrection:
    row_index: int
    elapsed_time: str
    field: str
    original: int
    replacement: int
    previous: int
    next_value: int
    reason: str


def int_or_none(value: object) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def next_stable_neighbor(
    values: list[Optional[int]],
    start: int,
    previous: int,
    config: CountSmoothingConfig,
) -> tuple[Optional[int], Optional[int]]:
    stop = min(len(values), start + config.lookahead + 1)
    for index in range(start + 1, stop):
        value = values[index]
        if value is not None and abs(value - previous) <= config.neighbor_tolerance:
            return index, value
    return None, None


def smooth_field(
    rows: list[dict[str, str]],
    field: str,
    config: CountSmoothingConfig,
) -> list[CountCorrection]:
    values = [int_or_none(row.get(field)) for row in rows]
    smoothed = list(values)
    corrections: list[CountCorrection] = []
    previous: Optional[int] = None
    index = 0

    while index < len(values):
        value = values[index]
        if value is None:
            index += 1
            continue
        if previous is None:
            previous = value
            index += 1
            continue

        if abs(value - previous) > config.max_jump:
            following_index, following = next_stable_neighbor(values, index, previous, config)
            if following_index is not None and following is not None:
                for correction_index in range(index, following_index):
                    original = values[correction_index]
                    if original is None:
                        continue
                    smoothed[correction_index] = previous
                    corrections.append(
                        CountCorrection(
                            row_index=correction_index,
                            elapsed_time=str(rows[correction_index].get("elapsed_time", "")),
                            field=field,
                            original=original,
                            replacement=previous,
                            previous=previous,
                            next_value=following,
                            reason="short_noise_run_between_stable_neighbors",
                        )
                    )
                previous = following
                index = following_index + 1
                continue

        previous = value
        index += 1

    for index, value in enumerate(smoothed):
        if value is not None:
            rows[index][field] = str(value)
    return corrections


def smooth_rows(
    rows: Iterable[dict[str, str]],
    fields: Iterable[str] = COUNT_FIELDS,
    config: CountSmoothingConfig = CountSmoothingConfig(),
) -> tuple[list[dict[str, str]], list[CountCorrection]]:
    output = [dict(row) for row in rows]
    corrections: list[CountCorrection] = []
    for field in fields:
        corrections.extend(smooth_field(output, field, config))
    return output, corrections


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def correction_summary(corrections: Iterable[CountCorrection]) -> dict[str, object]:
    corrections = list(corrections)
    by_field: dict[str, int] = {}
    for correction in corrections:
        by_field[correction.field] = by_field.get(correction.field, 0) + 1
    return {
        "total_corrections": len(corrections),
        "by_field": by_field,
        "corrections": [asdict(correction) for correction in corrections],
    }
