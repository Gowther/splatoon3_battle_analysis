from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def populated(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    return value not in ("", None)


def seconds_slug(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def analysis_id_for_window(match_id: str, start_seconds: float, stop_seconds: float, prefix: str = "best") -> str:
    return f"{match_id}_{prefix}_{seconds_slug(start_seconds)}_{seconds_slug(stop_seconds)}"


def candidate_windows(
    duration_seconds: float,
    *,
    start_seconds: float = 20.0,
    window_seconds: float = 30.0,
    stride_seconds: float = 40.0,
    stop_margin_seconds: float = 20.0,
) -> list[dict[str, float]]:
    if duration_seconds <= 0:
        return []

    latest_stop = max(0.0, duration_seconds - max(0.0, stop_margin_seconds))
    if latest_stop <= start_seconds:
        latest_stop = duration_seconds

    windows: list[dict[str, float]] = []
    start = float(start_seconds)
    while start + window_seconds <= latest_stop + 0.001:
        windows.append({"start_seconds": round(start, 3), "stop_seconds": round(start + window_seconds, 3)})
        start += stride_seconds

    if not windows and duration_seconds > start_seconds + 5.0:
        stop = min(duration_seconds, start_seconds + window_seconds)
        windows.append({"start_seconds": round(start_seconds, 3), "stop_seconds": round(stop, 3)})
    return windows


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def count_jump_warnings(rows: list[dict[str, str]], max_jump: int = 20) -> list[str]:
    warnings: list[str] = []
    fields = ["count_left", "count_right", "penalty_left", "penalty_right"]
    for field in fields:
        previous_time: str | None = None
        previous_value: int | None = None
        for row in rows:
            current_value = int_or_none(row.get(field))
            if current_value is None:
                continue
            current_time = row.get("elapsed_time", "")
            if previous_value is not None and abs(current_value - previous_value) > max_jump:
                warnings.append(f"{field} jumps {previous_value}->{current_value} between {previous_time}s and {current_time}s")
            previous_value = current_value
            previous_time = current_time
    return warnings


def analysis_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    eight_state_rows = sum(1 for row in rows if all(populated(row, f"player_state_{i}") for i in range(1, 9)))
    weapon_rows = sum(1 for row in rows if populated(row, "weapon_1"))
    count_rows = sum(1 for row in rows if populated(row, "count_left") or populated(row, "count_right"))
    penalty_rows = sum(1 for row in rows if populated(row, "penalty_left") or populated(row, "penalty_right"))
    objective_rows = sum(
        1
        for row in rows
        if row.get("asari_count") != "0"
        or row.get("hoko_count") != "0"
        or row.get("area_count") != "0"
        or row.get("yagura_count") != "0"
    )
    player_rows = sum(1 for row in rows if row.get("player_detected") == "True")
    message_rows = sum(1 for row in rows if populated(row, "message"))

    def ratio(value: int) -> float:
        return round(value / total, 4) if total else 0.0

    jumps = count_jump_warnings(rows)

    return {
        "rows": total,
        "eight_player_state_rows": eight_state_rows,
        "weapon_rows": weapon_rows,
        "count_rows": count_rows,
        "penalty_rows": penalty_rows,
        "objective_rows": objective_rows,
        "player_rows": player_rows,
        "message_rows": message_rows,
        "count_jump_warning_count": len(jumps),
        "count_jump_samples": jumps[:5],
        "state_ratio": ratio(eight_state_rows),
        "weapon_ratio": ratio(weapon_rows),
        "count_ratio": ratio(count_rows),
        "objective_ratio": ratio(objective_rows),
        "player_ratio": ratio(player_rows),
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    score = (
        float(metrics.get("count_ratio", 0.0)) * 4.0
        + float(metrics.get("state_ratio", 0.0)) * 2.0
        + float(metrics.get("objective_ratio", 0.0)) * 1.5
        + float(metrics.get("weapon_ratio", 0.0)) * 1.2
        + float(metrics.get("player_ratio", 0.0)) * 1.0
    )
    if int(metrics.get("rows", 0)) <= 0:
        score -= 10.0
    if float(metrics.get("count_ratio", 0.0)) == 0.0:
        score -= 0.75
    score -= min(float(metrics.get("count_jump_warning_count", 0)) * 0.08, 1.5)
    return round(score, 4)


def rank_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        item = dict(candidate)
        metrics = dict(item.get("metrics", {}))
        item["score"] = score_metrics(metrics)
        ranked.append(item)
    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("metrics", {}).get("count_ratio", 0.0)),
            float(item.get("metrics", {}).get("state_ratio", 0.0)),
            -float(item.get("start_seconds", 0.0)),
        ),
        reverse=True,
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Analysis Window Scan",
        "",
        f"- matches: {len(report.get('matches', []))}",
        f"- window_seconds: {report.get('window_seconds')}",
        f"- scan_sample_fps: {report.get('sample_fps')}",
        "",
        "| match | selected | score | count | jumps | state | weapon | objective | player |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for match in report.get("matches", []):
        selected = match.get("selected") or {}
        metrics = selected.get("metrics", {})
        lines.append(
            "| {match_id} | {start}-{stop} | {score:.4f} | {count:.1%} | {jumps} | {state:.1%} | {weapon:.1%} | {objective:.1%} | {player:.1%} |".format(
                match_id=match.get("match_id", ""),
                start=selected.get("start_seconds", ""),
                stop=selected.get("stop_seconds", ""),
                score=float(selected.get("score", 0.0)),
                count=float(metrics.get("count_ratio", 0.0)),
                jumps=int(metrics.get("count_jump_warning_count", 0)),
                state=float(metrics.get("state_ratio", 0.0)),
                weapon=float(metrics.get("weapon_ratio", 0.0)),
                objective=float(metrics.get("objective_ratio", 0.0)),
                player=float(metrics.get("player_ratio", 0.0)),
            )
        )

    for match in report.get("matches", []):
        lines.extend(["", f"## {match.get('match_id')}", "", "| rank | window | score | count | jumps | state | weapon | objective | player | csv |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"] )
        for index, candidate in enumerate(match.get("candidates", [])[:8], start=1):
            metrics = candidate.get("metrics", {})
            lines.append(
                "| {rank} | {start}-{stop} | {score:.4f} | {count:.1%} | {jumps} | {state:.1%} | {weapon:.1%} | {objective:.1%} | {player:.1%} | `{csv}` |".format(
                    rank=index,
                    start=candidate.get("start_seconds", ""),
                    stop=candidate.get("stop_seconds", ""),
                    score=float(candidate.get("score", 0.0)),
                    count=float(metrics.get("count_ratio", 0.0)),
                    jumps=int(metrics.get("count_jump_warning_count", 0)),
                    state=float(metrics.get("state_ratio", 0.0)),
                    weapon=float(metrics.get("weapon_ratio", 0.0)),
                    objective=float(metrics.get("objective_ratio", 0.0)),
                    player=float(metrics.get("player_ratio", 0.0)),
                    csv=candidate.get("csv", ""),
                )
            )
    return "\n".join(lines) + "\n"
