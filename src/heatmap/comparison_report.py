from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from src.data_registry import display_path, iter_heatmap_matches, resolve_project_path
from src.heatmap.anomaly_export import collect_candidates, read_csv_rows
from src.heatmap.trajectory_quality import quality_from_registry_heatmap, status_from_checks


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 4)


def selected(match_id: str, match_ids: list[str] | None) -> bool:
    return not match_ids or match_id in set(match_ids)


def anomaly_summary(
    match_id: str,
    heatmap: dict[str, Any],
    low_confidence: float,
    large_step_px: float,
    max_samples: int,
) -> dict[str, Any]:
    tracks = read_csv_rows(resolve_project_path(heatmap.get("player_tracks")))
    gaps = read_csv_rows(resolve_project_path(heatmap.get("player_track_gaps")))
    candidates = collect_candidates(
        match_id,
        heatmap.get("id", ""),
        tracks,
        gaps,
        low_confidence=low_confidence,
        large_step_px=large_step_px,
    )
    counts = Counter(candidate["anomaly_type"] for candidate in candidates)
    return {
        "total": len(candidates),
        "by_type": dict(sorted(counts.items())),
        "samples": candidates[:max_samples],
    }


def build_match_summary(
    match: dict[str, Any],
    heatmap: dict[str, Any],
    low_confidence: float,
    large_step_px: float,
    max_anomaly_samples: int,
) -> dict[str, Any]:
    metrics, checks = quality_from_registry_heatmap(heatmap)
    status = status_from_checks(checks)
    anomalies = anomaly_summary(
        match["id"],
        heatmap,
        low_confidence=low_confidence,
        large_step_px=large_step_px,
        max_samples=max_anomaly_samples,
    )
    return {
        "match_id": match["id"],
        "heatmap_id": heatmap.get("id", ""),
        "video": match.get("video", ""),
        "teams": heatmap.get("teams", []),
        "start_seconds": heatmap.get("start_seconds"),
        "stop_seconds": heatmap.get("stop_seconds"),
        "sample_fps": heatmap.get("sample_fps"),
        "status": status,
        "metrics": metrics,
        "checks": checks,
        "anomalies": anomalies,
    }


def aggregate(matches: list[dict[str, Any]]) -> dict[str, Any]:
    gap_ratios = [float(match["metrics"]["gap_ratio"]) for match in matches]
    jump_ratios = [float(match["metrics"]["jump_reset_ratio"]) for match in matches]
    status_counts = Counter(match["status"] for match in matches)
    anomaly_counts: Counter[str] = Counter()
    for match in matches:
        anomaly_counts.update(match["anomalies"]["by_type"])

    worst_gap = max(matches, key=lambda match: match["metrics"]["gap_ratio"], default=None)
    worst_jump = max(matches, key=lambda match: match["metrics"]["jump_reset_ratio"], default=None)
    return {
        "match_count": len(matches),
        "status": "failed" if any(match["status"] != "passed" for match in matches) else "passed",
        "status_counts": dict(sorted(status_counts.items())),
        "total_track_rows": sum(int(match["metrics"]["track_rows"]) for match in matches),
        "average_gap_ratio": average(gap_ratios),
        "average_jump_reset_ratio": average(jump_ratios),
        "worst_gap_match": worst_gap["match_id"] if worst_gap else "",
        "worst_gap_ratio": worst_gap["metrics"]["gap_ratio"] if worst_gap else None,
        "worst_jump_match": worst_jump["match_id"] if worst_jump else "",
        "worst_jump_reset_ratio": worst_jump["metrics"]["jump_reset_ratio"] if worst_jump else None,
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
    }


def build_comparison_report(
    registry: dict[str, Any],
    match_ids: list[str] | None = None,
    low_confidence: float = 0.56,
    large_step_px: float = 420.0,
    max_anomaly_samples: int = 8,
) -> dict[str, Any]:
    matches = [
        build_match_summary(
            match,
            heatmap,
            low_confidence=low_confidence,
            large_step_px=large_step_px,
            max_anomaly_samples=max_anomaly_samples,
        )
        for match, heatmap in iter_heatmap_matches(registry)
        if selected(match["id"], match_ids)
    ]
    return {
        "status": "failed" if any(match["status"] != "passed" for match in matches) else "passed",
        "thresholds": {
            "low_confidence": low_confidence,
            "large_step_px": large_step_px,
            "max_anomaly_samples": max_anomaly_samples,
        },
        "aggregate": aggregate(matches),
        "matches": matches,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    aggregate_payload = report["aggregate"]
    lines = [
        "# Heatmap Comparison",
        "",
        f"- status: `{report['status']}`",
        f"- matches: {aggregate_payload['match_count']}",
        f"- total_track_rows: {aggregate_payload['total_track_rows']}",
        f"- average_gap_ratio: {aggregate_payload['average_gap_ratio']}",
        f"- average_jump_reset_ratio: {aggregate_payload['average_jump_reset_ratio']}",
        f"- worst_gap_match: `{aggregate_payload['worst_gap_match']}` ({aggregate_payload['worst_gap_ratio']})",
        f"- worst_jump_match: `{aggregate_payload['worst_jump_match']}` ({aggregate_payload['worst_jump_reset_ratio']})",
        f"- anomaly_counts: {json.dumps(aggregate_payload['anomaly_counts'], ensure_ascii=False)}",
        "",
        "## Matches",
        "",
        "| match | teams | status | rows | gap | jump reset | route images | anomalies |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for match in report["matches"]:
        metrics = match["metrics"]
        lines.append(
            "| {match_id} | {teams} | {status} | {rows} | {gap} | {jump} | {routes} | {anomalies} |".format(
                match_id=match["match_id"],
                teams=", ".join(match.get("teams", [])),
                status=match["status"],
                rows=metrics["track_rows"],
                gap=metrics["gap_ratio"],
                jump=metrics["jump_reset_ratio"],
                routes=metrics["route_images"],
                anomalies=match["anomalies"]["total"],
            )
        )

    lines.extend(["", "## Quality Gates", ""])
    for match in report["matches"]:
        lines.extend([f"### {match['match_id']}", "", "| check | expected | actual | status |", "| --- | --- | --- | --- |"])
        if match["checks"]:
            for key, check in match["checks"].items():
                status = "passed" if check["ok"] else "failed"
                lines.append(f"| {key} | {check['expected']} | {check['actual']} | {status} |")
        else:
            lines.append("| none |  |  | passed |")
        lines.append("")

    lines.extend(["## Anomaly Samples", ""])
    for match in report["matches"]:
        samples = match["anomalies"]["samples"]
        lines.extend([f"### {match['match_id']}", ""])
        if not samples:
            lines.append("- none")
            lines.append("")
            continue
        for sample in samples:
            lines.append(
                "- {type} severity={severity} time={time} team={team} slot={slot} note={note}".format(
                    type=sample.get("anomaly_type", ""),
                    severity=sample.get("severity", ""),
                    time=sample.get("time", ""),
                    team=sample.get("team", ""),
                    slot=sample.get("track_slot", ""),
                    note=sample.get("note", ""),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Next Steps",
            "",
            "- Use high-severity anomaly samples to choose frames for manual point labels.",
            "- Normalize coordinates to stage maps before comparing absolute movement between stages.",
            "- Join event CSVs once external kill/death/objective events are available.",
        ]
    )
    return "\n".join(lines) + "\n"
