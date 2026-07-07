from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.core.paths import project_path
from src.data_registry import display_path


EXPECTED_COLUMNS = 33
COUNT_JUMP_THRESHOLD = 20
COUNT_FIELDS = ("count_left", "count_right", "penalty_left", "penalty_right")
OBJECTIVE_FIELDS = ("asari_count", "hoko_count", "area_count", "yagura_count")
WEAPON_FIELDS = tuple(f"weapon_{index}" for index in range(1, 9))
PLAYER_STATE_FIELDS = tuple(f"player_state_{index}" for index in range(1, 9))
SEVERITY_ORDER = {"info": 0, "warning": 1, "high": 2}


@dataclass(frozen=True)
class ErrorThresholds:
    player_state_missing_warning_ratio: float = 0.2
    player_state_missing_high_ratio: float = 0.5
    weapon_missing_after_first_warning_ratio: float = 0.1
    count_row_warning_ratio: float = 0.2
    objective_row_info_ratio: float = 0.2
    message_rows_info_min_count: int = 10
    message_rows_info_ratio: float = 0.1


def populated(row: dict[str, str], key: str) -> bool:
    return bool(row.get(key))


def int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def ratio(part: int, total: int) -> float:
    return round(part / total, 4) if total else 0.0


def read_csv(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    with project_path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def count_jump_warnings(rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    previous: dict[str, tuple[str, int]] = {}
    for row in rows:
        elapsed = row.get("elapsed_time", "")
        for key in COUNT_FIELDS:
            value = int_or_none(row.get(key))
            if value is None:
                continue
            if key in previous:
                prev_elapsed, prev_value = previous[key]
                if abs(value - prev_value) > COUNT_JUMP_THRESHOLD:
                    warnings.append(f"{key} jumps {prev_value}->{value} between {prev_elapsed}s and {elapsed}s")
            previous[key] = (elapsed, value)
    return warnings


def first_index(rows: list[dict[str, str]], predicate) -> int | None:
    return next((index for index, row in enumerate(rows) if predicate(row)), None)


def weapon_tuple(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in WEAPON_FIELDS)


def issue(category: str, severity: str, metric: str, value: Any, detail: str) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "metric": metric,
        "value": value,
        "detail": detail,
    }


def analyze_csv(path: Path, thresholds: ErrorThresholds = ErrorThresholds()) -> dict[str, Any]:
    csv_path = project_path(path)
    fieldnames, rows = read_csv(csv_path)
    row_count = len(rows)
    issues: list[dict[str, Any]] = []

    if fieldnames and len(fieldnames) != EXPECTED_COLUMNS:
        issues.append(
            issue("schema", "high", "column_count", len(fieldnames), f"expected {EXPECTED_COLUMNS} columns")
        )

    eight_state_rows = sum(1 for row in rows if all(populated(row, field) for field in PLAYER_STATE_FIELDS))
    no_state_rows = sum(1 for row in rows if not any(populated(row, field) for field in PLAYER_STATE_FIELDS))
    no_state_ratio = ratio(no_state_rows, row_count)
    if no_state_ratio >= thresholds.player_state_missing_high_ratio:
        issues.append(
            issue("detection_state", "high", "no_player_state_ratio", no_state_ratio, "many rows have no player state")
        )
    elif no_state_ratio >= thresholds.player_state_missing_warning_ratio:
        issues.append(
            issue(
                "detection_state",
                "warning",
                "no_player_state_ratio",
                no_state_ratio,
                "some rows have no player state",
            )
        )

    first_weapon_index = first_index(rows, lambda row: populated(row, "weapon_1"))
    weapon_rows = sum(1 for row in rows if populated(row, "weapon_1"))
    missing_weapon_after_first = 0
    weapon_set_change_rows = 0
    weapon_change_samples: list[dict[str, Any]] = []
    first_weapon_set: tuple[str, ...] | None = None
    if first_weapon_index is None:
        if row_count:
            issues.append(issue("weapon_classifier", "high", "weapon_rows", 0, "no weapon rows detected"))
    else:
        first_weapon_set = weapon_tuple(rows[first_weapon_index])
        after_first = rows[first_weapon_index:]
        missing_weapon_after_first = sum(1 for row in after_first if not populated(row, "weapon_1"))
        missing_ratio = ratio(missing_weapon_after_first, len(after_first))
        if missing_ratio > thresholds.weapon_missing_after_first_warning_ratio:
            issues.append(
                issue(
                    "weapon_classifier",
                    "warning",
                    "missing_after_first_weapon_ratio",
                    missing_ratio,
                    "weapon fields disappear after warmup",
                )
            )

        for row in after_first:
            current = weapon_tuple(row)
            if populated(row, "weapon_1") and current != first_weapon_set:
                weapon_set_change_rows += 1
                if len(weapon_change_samples) < 5:
                    weapon_change_samples.append({"elapsed_time": row.get("elapsed_time", ""), "weapons": current})
        if weapon_set_change_rows:
            issues.append(
                issue(
                    "weapon_classifier",
                    "warning",
                    "weapon_set_change_rows",
                    weapon_set_change_rows,
                    "weapon predictions changed after the first populated weapon row",
                )
            )

    count_rows = sum(1 for row in rows if populated(row, "count_left") or populated(row, "count_right"))
    count_row_ratio = ratio(count_rows, row_count)
    if row_count and count_row_ratio < thresholds.count_row_warning_ratio:
        issues.append(issue("count_ocr", "warning", "count_row_ratio", count_row_ratio, "count OCR is sparse"))

    jump_warnings = count_jump_warnings(rows)
    if jump_warnings:
        issues.append(
            issue(
                "count_ocr",
                "high",
                "count_jump_warnings",
                len(jump_warnings),
                "count or penalty values jump more than the configured threshold",
            )
        )

    objective_rows = sum(1 for row in rows if any(row.get(field) not in ("", "0", None) for field in OBJECTIVE_FIELDS))
    objective_row_ratio = ratio(objective_rows, row_count)
    if row_count and objective_row_ratio < thresholds.objective_row_info_ratio:
        issues.append(
            issue(
                "objective_ocr",
                "info",
                "objective_row_ratio",
                objective_row_ratio,
                "objective-specific OCR has few populated rows; this may be mode dependent",
            )
        )

    message_rows = [row for row in rows if populated(row, "message")]
    message_row_ratio = ratio(len(message_rows), row_count)
    if (
        len(message_rows) >= thresholds.message_rows_info_min_count
        and message_row_ratio >= thresholds.message_rows_info_ratio
    ):
        issues.append(
            issue(
                "message_ocr",
                "info",
                "message_rows",
                len(message_rows),
                "message OCR produced text; review samples for noise",
            )
        )

    player_detected_rows = sum(1 for row in rows if row.get("player_detected") == "True")
    if row_count and player_detected_rows == 0:
        issues.append(
            issue(
                "player_detector",
                "info",
                "player_detected_rows",
                0,
                "no direct player detections were recorded in this window",
            )
        )

    worst = "info"
    if issues:
        worst = max((item["severity"] for item in issues), key=lambda value: SEVERITY_ORDER[value])
    else:
        worst = "passed"

    return {
        "path": display_path(csv_path),
        "rows": row_count,
        "columns": len(fieldnames or []),
        "elapsed_start": rows[0].get("elapsed_time") if rows else "",
        "elapsed_end": rows[-1].get("elapsed_time") if rows else "",
        "metrics": {
            "eight_player_state_rows": eight_state_rows,
            "eight_player_state_ratio": ratio(eight_state_rows, row_count),
            "no_player_state_rows": no_state_rows,
            "no_player_state_ratio": no_state_ratio,
            "player_detected_rows": player_detected_rows,
            "player_detected_ratio": ratio(player_detected_rows, row_count),
            "weapon_rows": weapon_rows,
            "weapon_row_ratio": ratio(weapon_rows, row_count),
            "missing_weapon_after_first": missing_weapon_after_first,
            "weapon_set_change_rows": weapon_set_change_rows,
            "count_rows": count_rows,
            "count_row_ratio": count_row_ratio,
            "count_jump_warnings": len(jump_warnings),
            "objective_rows": objective_rows,
            "objective_row_ratio": objective_row_ratio,
            "message_rows": len(message_rows),
            "message_row_ratio": message_row_ratio,
        },
        "samples": {
            "count_jump_warnings": jump_warnings[:10],
            "weapon_change_samples": weapon_change_samples,
            "message_samples": [
                {"elapsed_time": row.get("elapsed_time", ""), "message": row.get("message", "")}
                for row in message_rows[:10]
            ],
            "first_weapon_set": list(first_weapon_set or []),
        },
        "issues": issues,
        "status": "passed" if not issues else "needs_review",
        "worst_severity": worst,
    }


def paths_from_evaluation_results(
    path: Path,
    use_smoothed: bool = False,
    only_ids: set[str] | None = None,
) -> list[Path]:
    results_path = project_path(path)
    with results_path.open(encoding="utf-8") as f:
        results = json.load(f)
    key = "smoothed_csv" if use_smoothed else "raw_csv"
    return [
        Path(item[key])
        for item in results
        if item.get("kind") == "analysis" and item.get(key) and (not only_ids or item.get("id") in only_ids)
    ]


def build_error_report(csv_paths: list[Path], thresholds: ErrorThresholds = ErrorThresholds()) -> dict[str, Any]:
    files = [analyze_csv(path, thresholds) for path in csv_paths]
    issue_counts = Counter(issue["category"] for file_result in files for issue in file_result["issues"])
    severity_counts = Counter(issue["severity"] for file_result in files for issue in file_result["issues"])
    high_issue_count = severity_counts.get("high", 0)
    warning_issue_count = severity_counts.get("warning", 0)
    status = "passed" if not high_issue_count and not warning_issue_count else "needs_review"
    return {
        "status": status,
        "files": files,
        "file_count": len(files),
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "recommendations": recommendations(issue_counts, severity_counts),
    }


def recommendations(issue_counts: Counter[str], severity_counts: Counter[str]) -> list[str]:
    result: list[str] = []
    if issue_counts.get("count_ocr"):
        result.append("Review count OCR crops/thresholds before changing the detector model.")
    if issue_counts.get("weapon_classifier"):
        result.append("Inspect weapon crop stability and classifier labels for windows with weapon issues.")
    if issue_counts.get("detection_state"):
        result.append("Check player state UI detections and frame quality around missing-state spans.")
    if issue_counts.get("message_ocr"):
        result.append("Treat message OCR as advisory until manually reviewed; it is noisy by design.")
    if severity_counts.get("high"):
        result.append("Prioritize high-severity issue samples before adding more expected metrics.")
    if not result:
        result.append("No major CSV-level model risk signals were found.")
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Error Report",
        "",
        f"- status: `{report['status']}`",
        f"- files: {report['file_count']}",
        f"- issue_counts: {json.dumps(report['issue_counts'], ensure_ascii=False)}",
        f"- severity_counts: {json.dumps(report['severity_counts'], ensure_ascii=False)}",
        "",
        "## Files",
        "",
        "| file | rows | severity | issues | state | weapons | counts | objective | messages |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for file_result in report["files"]:
        metrics = file_result["metrics"]
        lines.append(
            "| {path} | {rows} | {severity} | {issues} | {state:.1%} | {weapons:.1%} | {counts:.1%} | {objective:.1%} | {messages} |".format(
                path=file_result["path"],
                rows=file_result["rows"],
                severity=file_result["worst_severity"],
                issues=len(file_result["issues"]),
                state=metrics["eight_player_state_ratio"],
                weapons=metrics["weapon_row_ratio"],
                counts=metrics["count_row_ratio"],
                objective=metrics["objective_row_ratio"],
                messages=metrics["message_rows"],
            )
        )

    for file_result in report["files"]:
        if not file_result["issues"]:
            continue
        lines.extend(["", f"## {file_result['path']}", ""])
        for item in file_result["issues"]:
            lines.append(
                f"- `{item['severity']}` {item['category']}.{item['metric']}: {item['value']} - {item['detail']}"
            )
        samples = file_result["samples"]
        if samples["count_jump_warnings"]:
            lines.extend(["", "Count jump samples:"])
            lines.extend(f"- {sample}" for sample in samples["count_jump_warnings"])
        if samples["weapon_change_samples"]:
            lines.extend(["", "Weapon change samples:"])
            for sample in samples["weapon_change_samples"]:
                lines.append(f"- {sample['elapsed_time']}: {', '.join(sample['weapons'])}")
        if samples["message_samples"]:
            lines.extend(["", "Message samples:"])
            for sample in samples["message_samples"]:
                lines.append(f"- {sample['elapsed_time']}: {sample['message']}")

    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report["recommendations"])
    return "\n".join(lines) + "\n"


def thresholds_as_json(thresholds: ErrorThresholds) -> dict[str, Any]:
    return asdict(thresholds)
