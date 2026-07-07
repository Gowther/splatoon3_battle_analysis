from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.data_registry import DEFAULT_REGISTRY
from src.match_intake import DEFAULT_EVALUATION_CONFIG, IntakeConflict, IntakePaths, apply_intake_plan, load_json
from src.sample_intake import (
    build_sample_intake_plans,
    render_sample_intake_report,
    resolve_match_ids,
    scan_analysis_windows_command,
)

DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-register normal gameplay samples and optionally scan best windows.")
    parser.add_argument("--video", action="append", required=True, help="Video path. May be repeated.")
    parser.add_argument("--match-id", action="append", default=[], help="Explicit match id. Count must match --video.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--start-seconds", type=float, default=20.0)
    parser.add_argument("--stop-seconds", type=float, default=40.0)
    parser.add_argument("--sample-fps", type=float, help="Defaults to evaluation config defaults.sample_fps.")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], help="Defaults to evaluation config defaults.analysis_device.")
    parser.add_argument("--purpose", action="append", default=[], help="Registry purpose tag. Defaults to analysis_candidate.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--write", action="store_true", help="Write registry/evaluation updates.")
    parser.add_argument("--replace", action="store_true", help="Allow replacing existing matching ids.")
    parser.add_argument("--strict", action="store_true", help="Fail if any video is missing or unreadable.")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "sample_intake.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "sample_intake.json")
    parser.add_argument("--scan-analysis-windows", action="store_true", help="Run scan_analysis_windows.py after --write.")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=40.0)
    parser.add_argument("--stop-margin-seconds", type=float, default=20.0)
    parser.add_argument("--scan-sample-fps", type=float, default=2.0)
    parser.add_argument("--selected-sample-fps", type=float, help="Defaults to --sample-fps or evaluation config default.")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    return parser.parse_args()


def defaults_from_config(path: Path) -> dict[str, object]:
    config_path = project_path(path)
    if not config_path.exists():
        return {}
    return load_json(config_path).get("defaults", {})


def write_text(path: Path, content: str) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote: {target}")


def main() -> int:
    args = parse_args()
    if args.scan_analysis_windows and not args.write:
        raise SystemExit("--scan-analysis-windows requires --write so the scanner can read registry entries")

    defaults = defaults_from_config(args.evaluation_config)
    sample_fps = args.sample_fps if args.sample_fps is not None else defaults.get("sample_fps")
    selected_sample_fps = args.selected_sample_fps or sample_fps or defaults.get("sample_fps") or 5.0
    device = args.device or defaults.get("analysis_device") or "cpu"

    try:
        match_ids = resolve_match_ids(args.video, args.match_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    plans = build_sample_intake_plans(
        args.video,
        match_ids,
        purpose=args.purpose or None,
        notes=args.notes or None,
        mode=args.mode or None,
        stage=args.stage or None,
        start_seconds=args.start_seconds,
        stop_seconds=args.stop_seconds,
        sample_fps=float(sample_fps) if sample_fps is not None else None,
        device=str(device),
    )
    unreadable = [plan["match_id"] for plan in plans if not plan["video_probe"]["exists"] or not plan["video_probe"]["readable"]]
    if args.strict and unreadable:
        print(f"unreadable videos: {', '.join(unreadable)}", file=sys.stderr)
        return 1

    write_results = []
    if args.write:
        for plan in plans:
            try:
                write_results.append(
                    apply_intake_plan(
                        plan,
                        paths=IntakePaths(registry=args.registry, evaluation_config=args.evaluation_config),
                        replace=args.replace,
                    )
                )
            except IntakeConflict as exc:
                print(f"intake conflict: {exc}", file=sys.stderr)
                return 2

    scan_command = None
    scan_returncode = None
    if args.scan_analysis_windows:
        scan_command = scan_analysis_windows_command(
            args.python,
            match_ids,
            registry=args.registry,
            evaluation_config=args.evaluation_config,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            start_seconds=args.start_seconds,
            stop_margin_seconds=args.stop_margin_seconds,
            sample_fps=args.scan_sample_fps,
            selected_sample_fps=float(selected_sample_fps),
            device=str(device),
            warmup_frames=args.warmup_frames,
            force=args.force_scan,
        )
        print("$ " + " ".join(str(part) for part in scan_command), flush=True)
        scan_result = subprocess.run([str(part) for part in scan_command], cwd=ROOT)
        scan_returncode = scan_result.returncode
        if scan_returncode != 0:
            return scan_returncode

    report = render_sample_intake_report(
        plans,
        write_results=write_results or None,
        scan_command=scan_command,
        scan_returncode=scan_returncode,
    )
    payload = {
        "status": "passed",
        "match_ids": match_ids,
        "plans": plans,
        "write_results": write_results,
        "scan_command": [str(part) for part in scan_command] if scan_command else None,
        "scan_returncode": scan_returncode,
    }
    write_text(args.report, report)
    write_text(args.json_output, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"sample intake status: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
