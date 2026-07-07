from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"

for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from summarize_csv import populated, quality_warnings
from src.count_smoothing import CountSmoothingConfig, correction_summary, read_csv, smooth_rows, write_csv
from src.data_registry import DEFAULT_REGISTRY, get_match, load_registry
from src.heatmap.trajectory_quality import (
    quality_from_registry_heatmap,
    status_from_checks as trajectory_status_from_checks,
    write_json as write_quality_json,
    write_markdown_report as write_trajectory_report,
)


ANALYSIS_OPTIONS = (
    ("start_seconds", "--start-seconds"),
    ("stop_seconds", "--stop-seconds"),
    ("sample_fps", "--sample-fps"),
    ("max_frames", "--max-frames"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Splatoon 3 match evaluations and reports.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "evaluation_matches.json")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--only", action="append", default=[], help="Evaluate only the matching id. May be repeated.")
    parser.add_argument("--run-analysis", action="store_true", help="Run src.run_analysis before reporting analysis matches.")
    parser.add_argument("--run-heatmap-report", action="store_true", help="Regenerate heatmap reports when a heatmap config is available.")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], help="Override device for analysis matches.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable used for subprocesses.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every selected item passes.")
    return parser.parse_args()


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_step(name: str, command: list[object], python_path: Path) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n== {name} ==", flush=True)
    print(f"$ {printable}", flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_project_path(path) or path
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def smoothing_config(defaults: dict[str, Any]) -> CountSmoothingConfig:
    raw = defaults.get("count_smoothing", {})
    return CountSmoothingConfig(
        max_jump=int(raw.get("max_jump", 20)),
        neighbor_tolerance=int(raw.get("neighbor_tolerance", 3)),
        lookahead=int(raw.get("lookahead", 3)),
        leading_lookahead=int(raw.get("leading_lookahead", 3)),
        max_value=int(raw.get("max_value", 100)),
        digit_drop_max_raw=int(raw.get("digit_drop_max_raw", 30)),
        digit_drop_tolerance=int(raw.get("digit_drop_tolerance", 5)),
    )


def analysis_metrics(rows: list[dict[str, str]], fieldnames: list[str] | None) -> dict[str, Any]:
    warnings = quality_warnings(rows, fieldnames)
    objective_rows = sum(
        1
        for row in rows
        if row.get("asari_count") != "0"
        or row.get("hoko_count") != "0"
        or row.get("area_count") != "0"
        or row.get("yagura_count") != "0"
    )
    return {
        "rows": len(rows),
        "columns": len(fieldnames or []),
        "elapsed_start": rows[0].get("elapsed_time") if rows else "",
        "elapsed_end": rows[-1].get("elapsed_time") if rows else "",
        "eight_player_state_rows": sum(1 for row in rows if all(populated(row, f"player_state_{i}") for i in range(1, 9))),
        "weapon_rows": sum(1 for row in rows if populated(row, "weapon_1")),
        "count_rows": sum(1 for row in rows if populated(row, "count_left") or populated(row, "count_right")),
        "penalty_rows": sum(1 for row in rows if populated(row, "penalty_left") or populated(row, "penalty_right")),
        "objective_rows": objective_rows,
        "player_rows": sum(1 for row in rows if row.get("player_detected") == "True"),
        "message_rows": sum(1 for row in rows if populated(row, "message")),
        "warnings": warnings,
        "warning_count": len(warnings),
        "count_jump_warning_count": sum(1 for warning in warnings if " jumps " in warning),
    }


def compare_expected(metrics: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "expected": expected_value,
            "actual": metrics.get(key),
            "ok": metrics.get(key) == expected_value,
        }
        for key, expected_value in expected.items()
    }


def evaluate_quality_gates(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if "max_smoothed_count_jump_warnings" in gates:
        limit = int(gates["max_smoothed_count_jump_warnings"])
        actual = int(metrics["count_jump_warning_count"])
        results["max_smoothed_count_jump_warnings"] = {
            "expected": f"<= {limit}",
            "actual": actual,
            "ok": actual <= limit,
        }
    return results


def status_from_checks(*groups: dict[str, dict[str, Any]]) -> str:
    for group in groups:
        if any(not result["ok"] for result in group.values()):
            return "failed"
    return "passed"


def metric_table(raw: dict[str, Any], smoothed: dict[str, Any]) -> list[str]:
    keys = (
        "rows",
        "columns",
        "eight_player_state_rows",
        "weapon_rows",
        "count_rows",
        "penalty_rows",
        "objective_rows",
        "player_rows",
        "message_rows",
        "count_jump_warning_count",
        "warning_count",
    )
    lines = ["| metric | raw | smoothed |", "| --- | ---: | ---: |"]
    for key in keys:
        lines.append(f"| {key} | {raw.get(key, '')} | {smoothed.get(key, '')} |")
    return lines


def checks_table(title: str, checks: dict[str, dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| check | expected | actual | status |", "| --- | --- | --- | --- |"]
    if not checks:
        lines.append("| none |  |  | passed |")
        return lines
    for key, check in checks.items():
        status = "passed" if check["ok"] else "failed"
        lines.append(f"| {key} | {check['expected']} | {check['actual']} | {status} |")
    return lines


def warning_lines(title: str, warnings: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    return lines


def write_analysis_report(
    output: Path,
    match: dict[str, Any],
    raw_csv: Path,
    smoothed_csv: Path,
    raw_metrics: dict[str, Any],
    smoothed_metrics: dict[str, Any],
    smoothing_summary: dict[str, Any],
    expected_checks: dict[str, dict[str, Any]],
    quality_checks: dict[str, dict[str, Any]],
) -> None:
    lines = [
        f"# Match Evaluation: {match['id']}",
        "",
        f"- input: {match.get('input', '')}",
        f"- raw_csv: {display_path(raw_csv)}",
        f"- smoothed_csv: {display_path(smoothed_csv)}",
        f"- elapsed: {raw_metrics['elapsed_start']} -> {raw_metrics['elapsed_end']}",
        f"- count corrections: {smoothing_summary['total_corrections']}",
        "",
        "## Metrics",
        "",
        *metric_table(raw_metrics, smoothed_metrics),
        "",
        *checks_table("Expected Baseline", expected_checks),
        "",
        *checks_table("Quality Gates", quality_checks),
        "",
        *warning_lines("Raw Warnings", raw_metrics["warnings"]),
        "",
        *warning_lines("Smoothed Warnings", smoothed_metrics["warnings"]),
        "",
        "## Count Smoothing",
        "",
    ]
    by_field = smoothing_summary.get("by_field", {})
    if by_field:
        lines.extend(f"- {field}: {count}" for field, count in by_field.items())
    else:
        lines.append("- no corrections")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_analysis_match(
    match: dict[str, Any],
    defaults: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    match_id = match["id"]
    match_dir = output_dir / match_id
    match_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = match_dir / "raw.csv"
    smoothed_csv = match_dir / "smoothed.csv"
    smoothing_report = match_dir / "count_smoothing.json"
    report_path = match_dir / "report.md"

    python_path = resolve_project_path(args.python) or args.python.expanduser()
    if args.run_analysis:
        input_path = match["input"]
        device = args.device or match.get("device") or defaults.get("analysis_device", "cpu")
        command: list[object] = [
            python_path,
            "-m",
            "src.run_analysis",
            "--input",
            input_path,
            "--output",
            raw_csv,
            "--device",
            device,
        ]
        for config_key, cli_option in ANALYSIS_OPTIONS:
            value = match.get(config_key, defaults.get(config_key))
            if value is not None:
                command.extend([cli_option, value])
        run_step(f"{match_id} analysis", command, python_path)

    if not raw_csv.exists():
        return {
            "kind": "analysis",
            "id": match_id,
            "status": "skipped",
            "reason": "raw CSV is missing; pass --run-analysis to generate it",
            "raw_csv": display_path(raw_csv),
        }

    fieldnames, rows = read_csv(raw_csv)
    raw_metrics = analysis_metrics(rows, fieldnames)
    smoothed_rows, corrections = smooth_rows(rows, config=smoothing_config(defaults))
    write_csv(smoothed_csv, fieldnames, smoothed_rows)
    smoothing_summary = correction_summary(corrections)
    smoothing_report.write_text(json.dumps(smoothing_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    smoothed_metrics = analysis_metrics(smoothed_rows, fieldnames)

    expected_checks = compare_expected(raw_metrics, match.get("expected", {}))
    quality_checks = evaluate_quality_gates(smoothed_metrics, match.get("quality_gates", {}))
    status = status_from_checks(expected_checks, quality_checks)
    write_analysis_report(
        report_path,
        match,
        raw_csv,
        smoothed_csv,
        raw_metrics,
        smoothed_metrics,
        smoothing_summary,
        expected_checks,
        quality_checks,
    )

    return {
        "kind": "analysis",
        "id": match_id,
        "status": status,
        "raw_csv": display_path(raw_csv),
        "smoothed_csv": display_path(smoothed_csv),
        "smoothing_report": display_path(smoothing_report),
        "report": display_path(report_path),
        "raw_metrics": raw_metrics,
        "smoothed_metrics": smoothed_metrics,
        "smoothing_summary": smoothing_summary,
        "expected_checks": expected_checks,
        "quality_checks": quality_checks,
    }


def merge_registry_heatmap(match: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    registry_id = match.get("registry_id")
    if not registry_id:
        return dict(match)

    registry_match = get_match(registry, registry_id)
    if not registry_match:
        merged = dict(match)
        merged["registry_error"] = f"registry match not found: {registry_id}"
        return merged

    registry_heatmap = registry_match.get("heatmap")
    if not isinstance(registry_heatmap, dict):
        merged = dict(match)
        merged["registry_error"] = f"registry match has no heatmap entry: {registry_id}"
        return merged

    merged = dict(registry_heatmap)
    merged["registry_match_id"] = registry_match["id"]
    merged["video"] = registry_match.get("video", "")
    merged.update(match)
    return merged


def evaluate_heatmap_match(
    match: dict[str, Any],
    args: argparse.Namespace,
    registry: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    match = merge_registry_heatmap(match, registry)
    match_id = match["id"]
    python_path = resolve_project_path(args.python) or args.python.expanduser()
    config_path = resolve_project_path(match.get("config"))
    report_path = resolve_project_path(match.get("report"))
    color_report = resolve_project_path(match.get("color_report"))
    notes: list[str] = []
    if match.get("registry_error"):
        notes.append(match["registry_error"])

    if args.run_heatmap_report:
        if config_path and config_path.exists():
            run_step(
                f"{match_id} heatmap report",
                [python_path, "-m", "src.heatmap.run_pipeline", "--config", config_path, "--only-report"],
                python_path,
            )
        else:
            notes.append("heatmap config missing; checked existing outputs only")

    report_exists = bool(report_path and report_path.exists())
    color_report_exists = bool(color_report and color_report.exists())
    quality_metrics: dict[str, Any] = {}
    quality_checks: dict[str, dict[str, Any]] = {}
    quality_json_path = output_dir / match_id / "trajectory_quality.json"
    quality_report_path = output_dir / match_id / "trajectory_quality.md"

    if match.get("player_tracks"):
        quality_metrics, quality_checks = quality_from_registry_heatmap(match)
        quality_payload = {
            "match_id": match.get("registry_match_id", ""),
            "heatmap_id": match_id,
            "status": trajectory_status_from_checks(quality_checks),
            "metrics": quality_metrics,
            "checks": quality_checks,
        }
        write_quality_json(quality_json_path, quality_payload)
        write_trajectory_report(
            quality_report_path,
            f"{match_id} Trajectory Quality",
            quality_metrics,
            quality_checks,
        )
    else:
        notes.append("player track path is not configured; trajectory quality skipped")

    artifact_status = report_exists and color_report_exists and not match.get("registry_error")
    trajectory_status = trajectory_status_from_checks(quality_checks)
    status = "passed" if artifact_status and trajectory_status == "passed" else "failed"
    return {
        "kind": "heatmap",
        "id": match_id,
        "registry_match_id": match.get("registry_match_id", ""),
        "status": status,
        "config": display_path(config_path),
        "report": display_path(report_path),
        "report_exists": report_exists,
        "color_report": display_path(color_report),
        "color_report_exists": color_report_exists,
        "trajectory_quality_json": display_path(quality_json_path) if quality_metrics else "",
        "trajectory_quality_report": display_path(quality_report_path) if quality_metrics else "",
        "trajectory_metrics": quality_metrics,
        "trajectory_checks": quality_checks,
        "notes": notes,
    }


def selected(match_id: str, only: list[str]) -> bool:
    return not only or match_id in only


def write_aggregate_report(path: Path, config_path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Splatoon 3 Evaluation",
        "",
        f"- config: {display_path(config_path)}",
        f"- results: {len(results)}",
        "",
        "| kind | id | status | details |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        if result["kind"] == "analysis":
            details = result.get("reason")
            if not details:
                smoothed = result["smoothed_metrics"]
                details = (
                    f"rows={smoothed['rows']}, "
                    f"count_jump_warnings={smoothed['count_jump_warning_count']}, "
                    f"corrections={result['smoothing_summary']['total_corrections']}"
                )
        else:
            metrics = result.get("trajectory_metrics", {})
            if metrics:
                details = (
                    f"report={result['report_exists']}, "
                    f"color_report={result['color_report_exists']}, "
                    f"track_rows={metrics.get('track_rows')}, "
                    f"gap_ratio={metrics.get('gap_ratio')}, "
                    f"jump_reset_ratio={metrics.get('jump_reset_ratio')}"
                )
            else:
                details = f"report={result['report_exists']}, color_report={result['color_report_exists']}"
            if result.get("notes"):
                details += f", notes={'; '.join(result['notes'])}"
        lines.append(f"| {result['kind']} | {result['id']} | {result['status']} | {details} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(args.config) or args.config.expanduser()
    registry_path = resolve_project_path(args.registry) or args.registry.expanduser()
    output_dir = resolve_project_path(args.output_dir) or args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    registry = load_registry(registry_path)
    defaults = config.get("defaults", {})
    results: list[dict[str, Any]] = []

    if not args.skip_analysis:
        for match in config.get("analysis_matches", []):
            if selected(match["id"], args.only):
                results.append(evaluate_analysis_match(match, defaults, args, output_dir))

    if not args.skip_heatmap:
        for match in config.get("heatmap_matches", []):
            if selected(match["id"], args.only):
                results.append(evaluate_heatmap_match(match, args, registry, output_dir))

    results_json = output_dir / "evaluation_results.json"
    report_path = output_dir / "evaluation_report.md"
    results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_aggregate_report(report_path, config_path, results)
    print(f"\nwrote evaluation results: {results_json}")
    print(f"wrote evaluation report: {report_path}")

    if args.strict and any(result["status"] != "passed" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
