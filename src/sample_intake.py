from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.match_intake import build_intake_plan


def match_id_from_video(video: str | Path) -> str:
    return Path(video).stem


def resolve_match_ids(videos: Sequence[str | Path], explicit_ids: Sequence[str]) -> list[str]:
    if explicit_ids and len(explicit_ids) != len(videos):
        raise ValueError("--match-id count must match --video count")
    return list(explicit_ids) if explicit_ids else [match_id_from_video(video) for video in videos]


def build_sample_intake_plans(
    videos: Sequence[str | Path],
    match_ids: Sequence[str],
    *,
    purpose: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    stage: str | None = None,
    start_seconds: float | None = None,
    stop_seconds: float | None = None,
    sample_fps: float | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    return [
        build_intake_plan(
            match_id,
            video,
            purpose=purpose,
            notes=notes,
            mode=mode,
            stage=stage,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            sample_fps=sample_fps,
            device=device,
        )
        for match_id, video in zip(match_ids, videos)
    ]


def scan_analysis_windows_command(
    python: str | Path,
    match_ids: Sequence[str],
    *,
    registry: str | Path,
    evaluation_config: str | Path,
    window_seconds: float,
    stride_seconds: float,
    start_seconds: float,
    stop_margin_seconds: float,
    sample_fps: float,
    selected_sample_fps: float,
    device: str,
    warmup_frames: int,
    force: bool = False,
) -> list[object]:
    command: list[object] = [
        python,
        "scripts/scan_analysis_windows.py",
        "--registry",
        registry,
        "--evaluation-config",
        evaluation_config,
        "--window-seconds",
        window_seconds,
        "--stride-seconds",
        stride_seconds,
        "--start-seconds",
        start_seconds,
        "--stop-margin-seconds",
        stop_margin_seconds,
        "--sample-fps",
        sample_fps,
        "--selected-sample-fps",
        selected_sample_fps,
        "--device",
        device,
        "--warmup-frames",
        warmup_frames,
        "--write-selection",
    ]
    for match_id in match_ids:
        command.extend(["--match-id", match_id])
    if force:
        command.append("--force")
    return command


def render_sample_intake_report(
    plans: Sequence[dict[str, Any]],
    *,
    write_results: Sequence[dict[str, Any]] | None = None,
    scan_command: Sequence[object] | None = None,
    scan_returncode: int | None = None,
) -> str:
    lines = ["# Sample Intake Report", ""]
    for index, plan in enumerate(plans):
        probe = plan["video_probe"]
        lines.extend(
            [
                f"## {plan['match_id']}",
                "",
                f"- analysis_id: `{plan['analysis_id']}`",
                f"- video: `{plan['video']}`",
                f"- exists: {probe['exists']}",
                f"- readable: {probe['readable']}",
            ]
        )
        if probe.get("duration_seconds") is not None:
            lines.append(f"- duration_seconds: {probe['duration_seconds']}")
        if write_results:
            result = write_results[index]
            lines.append(f"- registry_status: {result['registry_status']}")
            lines.append(f"- evaluation_status: {result['evaluation_status']}")
        lines.extend(
            [
                "",
                "```json",
                json.dumps(plan["registry_entry"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )

    if scan_command:
        lines.extend(["## Analysis Window Scan", "", "```bash", " ".join(str(part) for part in scan_command), "```"])
        if scan_returncode is not None:
            lines.append(f"- returncode: {scan_returncode}")
    return "\n".join(lines).rstrip() + "\n"
