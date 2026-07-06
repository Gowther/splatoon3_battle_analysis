from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.match_intake import (
    DEFAULT_EVALUATION_CONFIG,
    IntakeConflict,
    IntakePaths,
    apply_intake_plan,
    build_intake_plan,
    load_json,
    render_intake_report,
)
from src.data_registry import DEFAULT_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a local match video and prepare evaluation config.")
    parser.add_argument("--match-id", required=True, help="Registry id, for example match_12.")
    parser.add_argument("--video", required=True, help="Video path, preferably under footages/.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--analysis-id", help="Evaluation/window id. Defaults to match_id_start_stop.")
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--stop-seconds", type=float)
    parser.add_argument("--sample-fps", type=float, help="Defaults to evaluation config defaults.sample_fps when omitted.")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "auto"],
        help="Defaults to evaluation config defaults.analysis_device when omitted.",
    )
    parser.add_argument(
        "--purpose",
        action="append",
        default=[],
        help="Registry purpose tag. May be repeated. Defaults to analysis_candidate.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--skip-analysis-window", action="store_true", help="Do not add analysis_windows to registry.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not add analysis_matches entry.")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed entries without writing files.")
    parser.add_argument("--write", action="store_true", help="Write registry and evaluation config updates.")
    parser.add_argument("--replace", action="store_true", help="Allow replacing an existing matching id.")
    parser.add_argument("--strict", action="store_true", help="Fail if the video is missing or unreadable.")
    parser.add_argument("--report", type=Path, help="Optional Markdown intake report path.")
    return parser.parse_args()


def defaults_from_config(path: Path) -> dict[str, object]:
    config_path = project_path(path)
    if not config_path.exists():
        return {}
    return load_json(config_path).get("defaults", {})


def main() -> int:
    args = parse_args()
    if args.write and args.dry_run:
        raise SystemExit("--write and --dry-run are mutually exclusive")

    defaults = defaults_from_config(args.evaluation_config)
    sample_fps = args.sample_fps if args.sample_fps is not None else defaults.get("sample_fps")
    device = args.device or defaults.get("analysis_device")

    plan = build_intake_plan(
        args.match_id,
        args.video,
        purpose=args.purpose or None,
        notes=args.notes or None,
        mode=args.mode or None,
        stage=args.stage or None,
        analysis_id=args.analysis_id,
        start_seconds=args.start_seconds,
        stop_seconds=args.stop_seconds,
        sample_fps=float(sample_fps) if sample_fps is not None else None,
        device=str(device) if device else None,
        include_analysis_window=not args.skip_analysis_window,
        include_evaluation_match=not args.skip_evaluation,
    )

    probe = plan["video_probe"]
    if args.strict and (not probe["exists"] or not probe["readable"]):
        print(render_intake_report(plan), end="")
        return 1

    write_result = None
    if args.write:
        try:
            write_result = apply_intake_plan(
                plan,
                paths=IntakePaths(registry=args.registry, evaluation_config=args.evaluation_config),
                replace=args.replace,
            )
        except IntakeConflict as exc:
            print(f"intake conflict: {exc}", file=sys.stderr)
            return 2

    report = render_intake_report(plan, write_result=write_result)
    if args.report:
        report_path = project_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"wrote intake report: {report_path}")

    action = "wrote" if args.write else "dry-run"
    print(f"match intake {action}: {plan['match_id']} -> {plan['analysis_id']}")
    if not args.report:
        print(report, end="")
    else:
        print(json.dumps({"video_probe": probe, "write_result": write_result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
