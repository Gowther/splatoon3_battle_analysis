from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.core.paths import display_path as rel
from src.heatmap.color_calibration import resolve_config
from src.heatmap.death_positions import run_death_position_pipeline
from src.heatmap.extract_frames import ROOT, load_config, resolve_path
from src.heatmap.run_manifest import runtime_model_report, write_run_manifest
from src.heatmap.render_stage_space import render_stage_heatmaps
from src.heatmap.stage_coordinates import (
    build_stage_coordinate_report,
    discover_control_point_asset,
)
def stage_tracks_csv_path(config: Dict) -> Path:
    outputs = config.get("outputs", {})
    configured = outputs.get("player_tracks_stage_csv") if isinstance(outputs, dict) else None
    if isinstance(configured, str) and configured:
        return resolve_path(configured)
    return resolve_path(config["match"]["output_dir"]) / "player_tracks_stage.csv"


def run_stage_normalization(config: Dict) -> Dict[str, Any]:
    """Normalize tracks with homography, or an explicit provisional ROI fallback."""
    asset = discover_control_point_asset(config)
    points_csv = resolve_path(config["outputs"]["player_tracks_csv"])
    if not points_csv.exists():
        return {
            "status": "no_points",
            "method": "",
            "output": "",
            "asset": asset.get("path", "") if asset else "",
        }
    output_csv = stage_tracks_csv_path(config)
    report = build_stage_coordinate_report(
        config,
        points_csv=points_csv,
        normalized_csv=output_csv,
        control_point_asset=asset,
    )
    transform = report.get("transform", {})
    summary = report.get("points", {}).get("summary", {})
    return {
        "status": report.get("status", ""),
        "method": transform.get("method", ""),
        "quality": "calibrated" if transform.get("method") == "homography" else "provisional",
        "homography_status": transform.get("homography_status", ""),
        "reprojection": transform.get("reprojection", {}),
        "asset": asset.get("path", "") if asset else "",
        "stage_id": asset.get("stage_id", "") if asset else config.get("stage_coordinates", {}).get("stage_id", ""),
        "output": rel(output_csv),
        "normalized_rows": summary.get("normalized_rows", 0),
        "outside_roi_rows": summary.get("outside_roi_rows", 0),
    }


def stage_render_dir(config: Dict) -> Path:
    configured = config.get("outputs", {}).get("rendered_stage_dir")
    if isinstance(configured, str) and configured:
        return resolve_path(configured)
    return resolve_path(config["match"]["output_dir"]) / "rendered_stage"


def run_stage_rendering(config: Dict, stage_normalization: Dict[str, Any]) -> Dict[str, Any]:
    if stage_normalization.get("status") != "ready":
        return {"status": "skipped", "rendered": {}, "output_dir": rel(stage_render_dir(config))}
    report = render_stage_heatmaps(stage_tracks_csv_path(config), config, stage_render_dir(config))
    report["coordinate_method"] = stage_normalization.get("method", "")
    report["coordinate_quality"] = stage_normalization.get("quality", "")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the match_9 heatmap MVP pipeline.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--contact-limit", type=int, default=24)
    parser.add_argument("--event-csv", help="Optional external kill/death event CSV.")
    parser.add_argument("--skip-ui-analysis", action="store_true", help="Reuse an existing UI-state CSV.")
    parser.add_argument("--only-report", action="store_true", help="Only regenerate report.md from existing outputs.")
    parser.add_argument("--clean-output", action="store_true", help="Remove generated files under this match output_dir before running.")
    parser.add_argument("--teams", help="Comma-separated color preset override, for example orange,purple.")
    parser.add_argument("--disable-auto-colors", action="store_true", help="Use teams from the config without auto color calibration.")
    return parser.parse_args()


def run_command(label: str, command: Sequence[str]) -> None:
    print(f"\n== {label} ==")
    print(" ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def module_command(module: str, args: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", module, *args]


def read_metric_csv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f) if row.get("metric")}


def read_rows_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        row_count = sum(1 for _ in reader)
    return max(0, row_count - 1)


def artifact_line(label: str, path: Path) -> str:
    exists = "yes" if path.exists() else "no"
    return f"- {label}: `{rel(path)}` ({exists})"


def metric_lines(title: str, metrics: Dict[str, str], keys: Iterable[str]) -> List[str]:
    lines = [f"### {title}"]
    for key in keys:
        lines.append(f"- {key}: {metrics.get(key, '')}")
    return lines


def inside_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def configured_output_paths(config: Dict) -> List[Path]:
    output_dir = resolve_path(config["match"]["output_dir"])
    paths: set[Path] = {
        output_dir / "resolved_config.yaml",
        output_dir / "color_calibration_report.csv",
        output_dir / "run_manifest.json",
        stage_tracks_csv_path(config),
    }
    for value in config.get("outputs", {}).values():
        if isinstance(value, str):
            paths.add(resolve_path(value))
    state_csv = config.get("state_join", {}).get("state_csv")
    if isinstance(state_csv, str):
        paths.add(resolve_path(state_csv))
    event_csv = config.get("event_join", {}).get("event_csv")
    if isinstance(event_csv, str):
        paths.add(resolve_path(event_csv))
    return sorted(
        (path for path in paths if path != output_dir and inside_directory(path, output_dir)),
        key=lambda path: len(path.parts),
        reverse=True,
    )


def clean_generated_outputs(config: Dict) -> List[str]:
    removed: List[str] = []
    for path in configured_output_paths(config):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(rel(path))
    return removed


def write_report(
    config: Dict,
    command_hint: Optional[str] = None,
    *,
    stage_normalization: Optional[Dict[str, Any]] = None,
    stage_rendering: Optional[Dict[str, Any]] = None,
    death_positions: Optional[Dict[str, Any]] = None,
) -> Path:
    outputs = config["outputs"]
    output_dir = resolve_path(config["match"]["output_dir"])
    report_path = resolve_path(outputs["report_md"])
    cleaning = read_metric_csv(resolve_path(outputs["cleaning_report_csv"]))
    render = read_metric_csv(resolve_path(outputs["render_report_csv"]))
    state_join = read_metric_csv(resolve_path(outputs["state_join_report_csv"]))
    event_join = read_metric_csv(resolve_path(outputs["event_join_report_csv"]))
    identity = read_metric_csv(resolve_path(outputs["identity_report_csv"]))
    color_rows = read_rows_csv(resolve_path(outputs["color_calibration_report_csv"])) if outputs.get("color_calibration_report_csv") else []

    lines: List[str] = [
        f"# {config['match']['id']} Heatmap Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Input",
        "",
        f"- match id: `{config['match']['id']}`",
        f"- video: `{config['match']['input_video']}`",
        f"- sampled range: `{config['sampling']['start_seconds']}s` to `{config['sampling']['stop_seconds']}s`",
        f"- sample fps: `{config['sampling']['sample_fps']}`",
        f"- coordinate space: `{config['map_view']['coordinate_space']}`",
        f"- output directory: `{rel(output_dir)}`",
        "",
        "## Color Calibration",
        "",
    ]
    if color_rows:
        for row in color_rows:
            lines.append(
                f"- order {row.get('order', '')}: `{row.get('team', '')}` "
                f"(hue `{row.get('detected_hue', '')}`, source `{row.get('source', '')}`)"
            )
        resolved_config = config.get("color_calibration", {}).get("resolved_config")
        if resolved_config:
            lines.append(f"- resolved config: `{resolved_config}`")
    else:
        lines.append("- color calibration report: not available")

    lines.extend(
        [
        "",
        "## One Command",
        "",
        "```bash",
        command_hint
        or "PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.run_pipeline --config src/heatmap/config_match9.yaml",
        "```",
        "",
        "## Core Artifacts",
        "",
        ]
    )

    lines.extend(
        [
        artifact_line("valid frames", resolve_path(outputs["valid_frames_csv"])),
        artifact_line("map mask", resolve_path(outputs["map_mask"])),
        artifact_line("raw points", resolve_path(outputs["raw_points_csv"])),
        artifact_line("clean points", resolve_path(outputs["clean_points_csv"])),
        artifact_line("team tracks", resolve_path(outputs["tracks_csv"])),
        artifact_line("color calibration", resolve_path(outputs["color_calibration_report_csv"]))
        if outputs.get("color_calibration_report_csv")
        else "- color calibration: `(not configured)` (no)",
        artifact_line("UI state", resolve_path(config["state_join"]["state_csv"])),
        artifact_line("enriched points", resolve_path(outputs["enriched_points_csv"])),
        artifact_line("points with events", resolve_path(outputs["points_with_events_csv"])),
        artifact_line("experimental player tracks", resolve_path(outputs["player_tracks_csv"])),
        "",
        "## Visual Artifacts",
        "",
        artifact_line("all players heatmap", resolve_path(outputs["rendered_dir"]) / "heatmap_all.png"),
    ]
    )
    for team in config["teams"]:
        lines.append(artifact_line(f"{team} heatmap", resolve_path(outputs["rendered_dir"]) / f"heatmap_{team}.png"))
    lines.extend(
        [
            artifact_line("combined heatmap", resolve_path(outputs["rendered_dir"]) / "heatmap_combined.png"),
            artifact_line("team routes", resolve_path(outputs["rendered_dir"]) / "team_routes.png"),
            artifact_line("player routes directory", resolve_path(outputs["player_routes_dir"])),
            "",
            "## Counts",
        ]
    )

    lines.extend(
        [
        "",
        f"- valid frame rows: {count_csv_rows(resolve_path(outputs['valid_frames_csv']))}",
        f"- raw point rows: {count_csv_rows(resolve_path(outputs['raw_points_csv']))}",
        f"- clean point rows: {count_csv_rows(resolve_path(outputs['clean_points_csv']))}",
        f"- enriched point rows: {count_csv_rows(resolve_path(outputs['enriched_points_csv']))}",
        f"- player track rows: {count_csv_rows(resolve_path(outputs['player_tracks_csv']))}",
        "",
        ]
    )

    lines.extend(
        metric_lines(
            "Cleaning",
            cleaning,
            [
                "raw_points",
                "clean_points",
                "rejected_points",
                "track_rows",
                "track_unassigned_points",
                "track_coverage_ratio",
                "mean_tracking_confidence",
                "track_status_matched",
                "track_status_jump_reset",
                "track_status_new",
                "track_status_reacquired",
            ],
        )
    )
    lines.append("")
    lines.extend(metric_lines("State Join", state_join, ["state_rows", "matched_rows", "unmatched_rows", "max_observed_delta_seconds"]))
    lines.append("")
    render_metric_keys = ["clean_points", "track_points"]
    render_metric_keys.extend(f"clean_team_{team}" for team in config["teams"])
    lines.extend(metric_lines("Rendering", render, render_metric_keys))
    lines.append("")
    lines.extend(metric_lines("Events", event_join, ["event_rows", "points_with_nearby_events", "events_with_nearby_points", "segments"]))
    death_info = death_positions or {"status": "empty", "event_count": 0}
    reason_counts = death_info.get("reason_counts", {})
    reason_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
    lines.extend(
        [
            "",
            "### Death Positions",
            f"- status: `{death_info.get('status', '')}`",
            f"- death events: {death_info.get('event_count', 0)}",
            f"- located: {death_info.get('located_count', 0)}",
            f"- ambiguous: {death_info.get('ambiguous_count', 0)}",
            f"- unknown: {death_info.get('unknown_count', 0)}",
            f"- location reasons: `{reason_summary or 'none'}`",
            artifact_line("death events", resolve_path(outputs["death_events_csv"])),
            artifact_line("death positions", resolve_path(outputs["death_positions_csv"])),
            artifact_line("routes with deaths", resolve_path(outputs["routes_with_deaths"])),
            artifact_line("stage routes with deaths", resolve_path(outputs["stage_routes_with_deaths"])),
        ]
    )
    lines.append("")
    lines.extend(
        metric_lines(
            "Experimental Identity",
            identity,
            [
                "player_track_rows",
                "gap_rows",
                "gap_ratio",
                "large_step_rows",
                "matched_large_step_rows",
                "reacquired_large_step_rows",
                "max_matched_step_px",
                "max_reacquired_step_px",
                "route_images",
                "identity_warning",
            ],
        )
    )
    stage_info = stage_normalization or {"status": "no_asset"}
    lines.extend(["", "### Stage Normalization", ""])
    if stage_info.get("status") == "ready":
        reprojection = stage_info.get("reprojection", {}) if isinstance(stage_info.get("reprojection"), dict) else {}
        lines.extend(
            [
                f"- method: `{stage_info.get('method', '')}`",
                f"- quality: `{stage_info.get('quality', '')}`",
                f"- control point asset: `{stage_info.get('asset', '')}`",
                f"- stage tracks: `{stage_info.get('output', '')}`",
                f"- normalized rows: {stage_info.get('normalized_rows', 0)}",
                f"- reprojection max error: {reprojection.get('max_error', 0):.6f}"
                if reprojection.get("point_errors")
                else "- reprojection: not available",
            ]
        )
    else:
        lines.append(
            f"- status: `{stage_info.get('status', 'no_asset')}`; label control points in the /stage-labeling "
            "workbench page to enable homography output."
        )
    stage_limit_line = (
        "- Stage coordinates use the promoted control-point homography; verify landmarks before trusting cross-match comparisons."
        if stage_info.get("status") == "ready" and stage_info.get("method") == "homography"
        else "- Stage coordinates are a provisional 0..1 mapping of the configured video ROI; promote real control points before cross-match comparisons."
    )
    stage_render_info = stage_rendering or {"status": "skipped"}
    if stage_render_info.get("status") == "ready":
        lines.extend(
            [
                "",
                "### Stage-Space Rendering",
                "",
                artifact_line("normalized routes", stage_render_dir(config) / "stage_routes.png"),
                artifact_line("normalized combined heatmap", stage_render_dir(config) / "stage_heatmap_combined.png"),
            ]
        )
    marker_method = str(config.get("marker_detection", {}).get("method", ""))
    marker_limit_line = (
        "- Marker positions require both replay-specific player-name and triangle-template matches; occlusion creates explicit route gaps, and another replay needs new reference seeds."
        if marker_method == "seeded_name_marker_tracking"
        else "- Marker detection uses color components near map labels; some points can still land on nearby ink patches."
    )
    slot_mapping_verified = bool(config.get("identity_tracking", {}).get("slot_mapping_verified", False))
    player_identity_limit_line = (
        "- `player_tracks.csv` contains eight stable per-match identities whose HUD slots were verified from match_9 result-screen and status-bar weapon icons; another replay must be verified independently."
        if slot_mapping_verified
        else "- `player_tracks.csv` contains eight stable per-match identities seeded from the reference frame; their numeric suffixes are not verified HUD weapon-slot mappings."
    )
    death_limit_line = (
        "- Death positions use the verified HUD slot and the latest visible point for that identity; gaps or weak visual matches remain explicit unknowns rather than interpolated locations."
        if slot_mapping_verified
        else "- Death positions use one-to-one disappearance assignment while HUD-slot binding is unverified; inspect ambiguous and unknown rows before relying on them."
    )
    lines.extend(
        [
            "",
            "## Known Limitations",
            "",
            marker_limit_line,
            "- `team_routes.png` shows team-level local movement segments, not verified player paths.",
            player_identity_limit_line,
            death_limit_line,
            stage_limit_line,
            "",
            "## Embedded Preview",
            "",
            "![combined heatmap](rendered/heatmap_combined.png)",
            "",
            "![all players heatmap](rendered/heatmap_all.png)",
            "",
            "![team routes](rendered/team_routes.png)",
            "",
        ]
    )
    if stage_render_info.get("status") == "ready":
        lines.extend(
            [
                "![normalized routes](rendered_stage/stage_routes.png)",
                "",
                "![routes with deaths](rendered/routes_with_deaths.png)",
                "",
                "![normalized routes with deaths](rendered_stage/stage_routes_with_deaths.png)",
                "",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pipeline(args: argparse.Namespace, config: Dict, config_path: Path) -> None:
    config_path = str(resolve_path(config_path))
    run_command(
        "extract frames",
        module_command("src.heatmap.extract_frames", ["--config", config_path, "--contact-limit", str(args.contact_limit)]),
    )
    if not args.skip_ui_analysis:
        run_command(
            "run UI state analysis",
            module_command(
                "src.run_analysis",
                [
                    "--input",
                    config["match"]["input_video"],
                    "--output",
                    config["state_join"]["state_csv"],
                    "--start-seconds",
                    str(config["sampling"]["start_seconds"]),
                    "--stop-seconds",
                    str(config["sampling"]["stop_seconds"]),
                    "--sample-fps",
                    str(config["sampling"]["sample_fps"]),
                    "--device",
                    args.device,
                    "--warmup-frames",
                    str(args.warmup_frames),
                ],
            ),
        )
    run_command("build map mask", module_command("src.heatmap.build_map_mask", ["--config", config_path]))
    run_command("detect markers", module_command("src.heatmap.detect_markers", ["--config", config_path]))
    run_command("clean points", module_command("src.heatmap.clean_points", ["--config", config_path]))
    run_command("join UI state", module_command("src.heatmap.join_state", ["--config", config_path]))
    run_command("render heatmaps", module_command("src.heatmap.render_heatmaps", ["--config", config_path]))

    run_command("infer experimental player tracks", module_command("src.heatmap.infer_player_tracks", ["--config", config_path]))


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    cleaned_paths: List[str] = []
    if args.clean_output and not args.only_report:
        cleaned_paths = clean_generated_outputs(config)
    team_override = [item.strip() for item in args.teams.split(",")] if args.teams else None
    config, resolved_config_path, color_report_path = resolve_config(
        config,
        team_override=team_override,
        disable_auto=args.disable_auto_colors,
    )
    command_parts = [
        "PYTHONPYCACHEPREFIX=.cache/pycache",
        ".venv/bin/python",
        "-m",
        "src.heatmap.run_pipeline",
        "--config",
        args.config,
    ]
    if args.teams:
        command_parts.extend(["--teams", args.teams])
    if args.disable_auto_colors:
        command_parts.append("--disable-auto-colors")
    if args.clean_output:
        command_parts.append("--clean-output")
    command_hint = " ".join(command_parts)
    print(f"resolved color config: {resolved_config_path}")
    print(f"color calibration report: {color_report_path}")
    if cleaned_paths:
        print("cleaned generated outputs:")
        for path in cleaned_paths:
            print(f"- {path}")
    if not args.only_report:
        run_pipeline(args, config, resolved_config_path)
    stage_normalization = run_stage_normalization(config)
    if stage_normalization.get("status") == "ready":
        print(
            f"stage normalization: {stage_normalization['method']} -> {stage_normalization['output']} "
            f"({stage_normalization['normalized_rows']} rows)"
        )
    else:
        print(f"stage normalization: {stage_normalization.get('status', 'no_asset')}")
    stage_rendering = run_stage_rendering(config, stage_normalization)
    print(f"stage rendering: {stage_rendering.get('status', 'skipped')}")
    death_positions = run_death_position_pipeline(config, args.event_csv)
    print(
        "death positions: "
        f"events={death_positions.get('event_count', 0)} "
        f"located={death_positions.get('located_count', 0)} "
        f"ambiguous={death_positions.get('ambiguous_count', 0)} "
        f"unknown={death_positions.get('unknown_count', 0)}"
    )
    if not args.only_report:
        run_command(
            "join events",
            module_command(
                "src.heatmap.join_events",
                ["--config", str(resolved_config_path), "--events", death_positions["event_csv"]],
            ),
        )
    report_path = write_report(
        config,
        command_hint,
        stage_normalization=stage_normalization,
        stage_rendering=stage_rendering,
        death_positions=death_positions,
    )
    manifest_path = write_run_manifest(
        config,
        args,
        source_config_path=Path(args.config),
        resolved_config_path=resolved_config_path,
        color_report_path=color_report_path,
        report_path=report_path,
        command_hint=command_hint,
        cleaned_paths=cleaned_paths,
        stage_normalization=stage_normalization,
        stage_rendering=stage_rendering,
        death_positions=death_positions,
        model_report=runtime_model_report(args),
    )
    print(f"\nreport: {report_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
