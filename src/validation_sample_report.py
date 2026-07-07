from __future__ import annotations

from typing import Any


def sample_matches(registry: dict[str, Any], prefixes: tuple[str, ...] = ("n_match_", "f_match_")) -> list[dict[str, Any]]:
    return [match for match in registry.get("matches", []) if str(match.get("id", "")).startswith(prefixes)]


def index_by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("id")): item for item in items if item.get("id")}


def latest_best_window(match: dict[str, Any]) -> dict[str, Any] | None:
    prefix = f"{match.get('id')}_best_"
    for window in reversed(match.get("analysis_windows", [])):
        if str(window.get("id", "")).startswith(prefix):
            return window
    return None


def heatmap_comparison_by_match(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("match_id")): item for item in report.get("matches", [])}


def model_issues_by_csv(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("file")): item for item in report.get("files", [])}


def build_report(
    registry: dict[str, Any],
    evaluation_results: list[dict[str, Any]],
    heatmap_comparison: dict[str, Any],
    analysis_scan: dict[str, Any],
    model_error_report: dict[str, Any],
    heatmap_quality_loop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation_by_id = index_by_id(evaluation_results)
    heatmap_by_match = heatmap_comparison_by_match(heatmap_comparison)
    model_by_csv = model_issues_by_csv(model_error_report)
    scan_by_match = {str(item.get("match_id")): item for item in analysis_scan.get("matches", [])}

    normal_samples = []
    heatmap_samples = []
    for match in sample_matches(registry):
        match_id = str(match.get("id"))
        if match_id.startswith("n_match_"):
            window = latest_best_window(match)
            eval_result = evaluation_by_id.get(str(window.get("id"))) if window else None
            raw_csv = str(eval_result.get("raw_csv", "")) if eval_result else ""
            smoothed_csv = str(eval_result.get("smoothed_csv", "")) if eval_result else ""
            model_issue = model_by_csv.get(smoothed_csv) or model_by_csv.get(raw_csv)
            normal_samples.append(
                {
                    "id": match_id,
                    "video": match.get("video", ""),
                    "best_window": window or {},
                    "scan_selected": (scan_by_match.get(match_id) or {}).get("selected", {}),
                    "evaluation_status": eval_result.get("status", "missing") if eval_result else "missing",
                    "metrics": eval_result.get("smoothed_metrics", {}) if eval_result else {},
                    "model_status": model_issue.get("status", "missing") if model_issue else "missing",
                    "model_issues": model_issue.get("issues", []) if model_issue else [],
                }
            )
        elif match_id.startswith("f_match_"):
            heatmap = match.get("heatmap", {}) if isinstance(match.get("heatmap"), dict) else {}
            comparison = heatmap_by_match.get(match_id, {})
            heatmap_samples.append(
                {
                    "id": match_id,
                    "video": match.get("video", ""),
                    "teams": heatmap.get("teams", []),
                    "status": comparison.get("status", "missing"),
                    "metrics": comparison.get("metrics", {}),
                    "anomalies": comparison.get("anomalies", {}).get("total", ""),
                }
            )

    status = "passed"
    if any(item["evaluation_status"] == "missing" for item in normal_samples):
        status = "needs_review"
    if any(item["status"] != "passed" for item in heatmap_samples):
        status = "needs_review"
    if model_error_report.get("status") not in (None, "", "passed"):
        status = "needs_review"
    heatmap_quality_loop_status = (heatmap_quality_loop or {}).get("status", "missing")
    if heatmap_quality_loop and heatmap_quality_loop_status != "passed":
        status = "needs_review"

    return {
        "status": status,
        "normal_samples": normal_samples,
        "heatmap_samples": heatmap_samples,
        "model_error_status": model_error_report.get("status", "missing"),
        "heatmap_comparison_status": heatmap_comparison.get("status", "missing"),
        "heatmap_quality_loop_status": heatmap_quality_loop_status,
        "heatmap_quality_loop": heatmap_quality_loop or {},
    }


def ratio_text(metrics: dict[str, Any], key: str) -> str:
    rows = float(metrics.get("rows", 0) or 0)
    value = float(metrics.get(key, 0) or 0)
    if rows <= 0:
        return ""
    return f"{value / rows:.1%}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Validation Samples",
        "",
        f"- status: `{report.get('status')}`",
        f"- normal samples: {len(report.get('normal_samples', []))}",
        f"- heatmap samples: {len(report.get('heatmap_samples', []))}",
        f"- model error status: `{report.get('model_error_status')}`",
        f"- heatmap comparison status: `{report.get('heatmap_comparison_status')}`",
        f"- heatmap quality loop status: `{report.get('heatmap_quality_loop_status')}`",
        "",
        "## Normal Gameplay",
        "",
        "| match | best window | eval | model | counts | state | weapons | objective | player | issues |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("normal_samples", []):
        window = item.get("best_window", {})
        metrics = item.get("metrics", {})
        lines.append(
            "| {id} | {start}-{stop} | {eval} | {model} | {counts} | {state} | {weapons} | {objective} | {player} | {issues} |".format(
                id=item.get("id", ""),
                start=window.get("start_seconds", ""),
                stop=window.get("stop_seconds", ""),
                eval=item.get("evaluation_status", ""),
                model=item.get("model_status", ""),
                counts=ratio_text(metrics, "count_rows"),
                state=ratio_text(metrics, "eight_player_state_rows"),
                weapons=ratio_text(metrics, "weapon_rows"),
                objective=ratio_text(metrics, "objective_rows"),
                player=ratio_text(metrics, "player_rows"),
                issues=len(item.get("model_issues", [])),
            )
        )

    lines.extend(["", "## Heatmaps", "", "| match | teams | status | rows | gap | jump | routes | anomalies |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for item in report.get("heatmap_samples", []):
        metrics = item.get("metrics", {})
        lines.append(
            "| {id} | {teams} | {status} | {rows} | {gap} | {jump} | {routes} | {anomalies} |".format(
                id=item.get("id", ""),
                teams=",".join(item.get("teams", [])),
                status=item.get("status", ""),
                rows=metrics.get("track_rows", ""),
                gap=metrics.get("gap_ratio", ""),
                jump=metrics.get("jump_reset_ratio", ""),
                routes=metrics.get("route_images", ""),
                anomalies=item.get("anomalies", ""),
            )
        )
    quality_loop = report.get("heatmap_quality_loop", {})
    if quality_loop:
        metrics = quality_loop.get("metrics", {})
        lines.extend(
            [
                "",
                "## Heatmap Quality Loop",
                "",
                "| status | labeled | matched | recall | precision | mean error |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                "| {status} | {labeled} | {matched} | {recall} | {precision} | {mean_error} |".format(
                    status=quality_loop.get("status", ""),
                    labeled=metrics.get("labeled_rows", ""),
                    matched=metrics.get("matched_labels", ""),
                    recall=metrics.get("recall", ""),
                    precision=metrics.get("precision_on_complete_groups", ""),
                    mean_error=metrics.get("mean_error_px", ""),
                ),
            ]
        )
    return "\n".join(lines) + "\n"
