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
    leading_lookahead: int = 3
    max_value: int = 100
    digit_drop_max_raw: int = 30
    digit_drop_tolerance: int = 5


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


def next_non_null(values: list[Optional[int]], start: int, limit: int) -> tuple[Optional[int], Optional[int]]:
    stop = min(len(values), start + limit + 1)
    for index in range(start, stop):
        value = values[index]
        if value is not None:
            return index, value
    return None, None


def dropped_digit_replacement(
    value: int,
    anchors: Iterable[Optional[int]],
    config: CountSmoothingConfig,
) -> Optional[int]:
    if value > config.digit_drop_max_raw:
        return None
    for anchor in anchors:
        if anchor is None:
            continue
        for offset in range(10, config.max_value + 1, 10):
            candidate = value + offset
            if candidate > config.max_value:
                continue
            if abs(candidate - anchor) <= config.digit_drop_tolerance:
                return candidate
    return None


def has_stable_or_digit_drop_continuation(
    values: list[Optional[int]],
    start: int,
    target: int,
    config: CountSmoothingConfig,
) -> bool:
    stable_index, _stable = next_stable_neighbor(values, start, target, config)
    if stable_index is not None:
        return True
    stop = min(len(values), start + config.lookahead + 1)
    for index in range(start + 1, stop):
        value = values[index]
        if value is not None and dropped_digit_replacement(value, [target], config) is not None:
            return True
    return False


def smooth_field(
    rows: list[dict[str, str]],
    field: str,
    config: CountSmoothingConfig,
) -> list[CountCorrection]:
    values = [int_or_none(row.get(field)) for row in rows]
    smoothed = list(values)
    corrections: list[CountCorrection] = []
    previous: Optional[int] = None
    previous_index: Optional[int] = None
    index = 0

    while index < len(values):
        value = values[index]
        if value is None:
            index += 1
            continue
        if previous is None:
            following_index, following = next_non_null(values, index + 1, config.leading_lookahead)
            has_continuation = (
                has_stable_or_digit_drop_continuation(values, following_index, following, config)
                if following_index is not None and following is not None
                else False
            )
            replacement = dropped_digit_replacement(value, [following], config) if has_continuation and following is not None else None
            if replacement is not None:
                smoothed[index] = replacement
                corrections.append(
                    CountCorrection(
                        row_index=index,
                        elapsed_time=str(rows[index].get("elapsed_time", "")),
                        field=field,
                        original=value,
                        replacement=replacement,
                        previous=replacement,
                        next_value=following,
                        reason="dropped_digit_before_stable_neighbor",
                    )
                )
                previous = replacement
                previous_index = index
                index += 1
                continue
            if following_index is not None and following is not None and abs(value - following) > config.max_jump:
                stable_index, stable = next_stable_neighbor(values, following_index, following, config)
                if stable_index is not None and stable is not None:
                    smoothed[index] = following
                    corrections.append(
                        CountCorrection(
                            row_index=index,
                            elapsed_time=str(rows[index].get("elapsed_time", "")),
                            field=field,
                            original=value,
                            replacement=following,
                            previous=following,
                            next_value=stable,
                            reason="leading_noise_before_stable_neighbor",
                        )
                    )
                    previous = stable
                    previous_index = stable_index
                    index = stable_index + 1
                    continue
            previous = value
            previous_index = index
            index += 1
            continue

        previous_replacement = dropped_digit_replacement(previous, [value], config)
        if (
            previous_index is not None
            and previous_replacement is not None
            and previous < value - config.neighbor_tolerance
            and has_stable_or_digit_drop_continuation(values, index, value, config)
        ):
            smoothed[previous_index] = previous_replacement
            corrections.append(
                CountCorrection(
                    row_index=previous_index,
                    elapsed_time=str(rows[previous_index].get("elapsed_time", "")),
                    field=field,
                    original=previous,
                    replacement=previous_replacement,
                    previous=previous_replacement,
                    next_value=value,
                    reason="dropped_digit_near_next_value",
                )
            )
            previous = previous_replacement

        replacement = dropped_digit_replacement(value, [previous], config)
        if replacement is not None and value < previous - config.neighbor_tolerance:
            smoothed[index] = replacement
            corrections.append(
                CountCorrection(
                    row_index=index,
                    elapsed_time=str(rows[index].get("elapsed_time", "")),
                    field=field,
                    original=value,
                    replacement=replacement,
                    previous=previous,
                    next_value=replacement,
                    reason="dropped_digit_near_previous_value",
                )
            )
            previous = replacement
            previous_index = index
            index += 1
            continue

        following_index, following = next_stable_neighbor(values, index, previous, config)
        if following_index is not None and following is not None:
            if abs(value - previous) >= config.max_jump or abs(value - following) > config.max_jump:
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
                previous_index = following_index
                index = following_index + 1
                continue

        previous = value
        previous_index = index
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
