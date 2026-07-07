from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis_window_scan import (
    analysis_id_for_window,
    analysis_metrics,
    candidate_windows,
    rank_candidates,
    read_rows,
    render_markdown,
)
from src.data_registry import DEFAULT_REGISTRY, display_path, get_match, load_registry, resolve_project_path
from src.match_intake import probe_video

DEFAULT_EVALUATION_CONFIG = ROOT / "config" / "evaluation_matches.json"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan normal gameplay videos and select stronger analysis windows.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--match-id", action="append", default=[], help="Registry match id to scan. Defaults to analysis candidates.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "analysis_window_scan")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=40.0)
    parser.add_argument("--start-seconds", type=float, default=20.0)
    parser.add_argument("--stop-margin-seconds", type=float, default=20.0)
    parser.add_argument("--sample-fps", type=float, default=2.0, help="FPS used during scanning.")
    parser.add_argument("--selected-sample-fps", type=float, default=5.0, help="FPS written for selected evaluation windows.")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="mps")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--force", action="store_true", help="Re-run analysis even when candidate CSVs already exist.")
    parser.add_argument("--write-selection", action="store_true", help="Append selected windows to registry and evaluation config.")
    return parser.parse_args()


def selected_match_ids(registry: dict[str, Any], explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    return [
        match["id"]
        for match in registry.get("matches", [])
        if "analysis_candidate" in match.get("purpose", []) and match.get("video")
    ]


def run_analysis(args: argparse.Namespace, video: str, csv_path: Path, start: float, stop: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        "-m",
        "src.run_analysis",
        "--input",
        video,
        "--output",
        str(csv_path),
        "--device",
        args.device,
        "--start-seconds",
        str(start),
        "--stop-seconds",
        str(stop),
        "--sample-fps",
        str(args.sample_fps),
        "--warmup-frames",
        str(args.warmup_frames),
    ]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_json(path: Path) -> dict[str, Any]:
    target = resolve_project_path(path) or path.expanduser()
    with target.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def compact_registry_json(data: dict[str, Any]) -> str:
    import re

    text = json.dumps(data, indent=2, ensure_ascii=False)
    pattern = re.compile(r"\[\n(?P<body>(?:\s+\"[^\n\"]+\",?\n)+)\s+\]")

    def compact(match: re.Match[str]) -> str:
        lines = [line.strip().rstrip(",") for line in match.group("body").splitlines() if line.strip()]
        if not lines or not all(line.startswith('\"') and line.endswith('\"') for line in lines):
            return match.group(0)
        return "[" + ", ".join(lines) + "]"

    return pattern.sub(compact, text) + "\n"


def upsert_selection(report: dict[str, Any], registry_path: Path, evaluation_path: Path, selected_sample_fps: float, device: str) -> None:
    registry_target = resolve_project_path(registry_path) or registry_path.expanduser()
    evaluation_target = resolve_project_path(evaluation_path) or evaluation_path.expanduser()
    registry = load_json(registry_target)
    evaluation = load_json(evaluation_target)
    evaluation.setdefault("analysis_matches", [])

    for match_report in report.get("matches", []):
        selected = match_report.get("selected")
        if not selected:
            continue
        match = get_match(registry, match_report["match_id"])
        if not match:
            continue
        analysis_id = analysis_id_for_window(match["id"], float(selected["start_seconds"]), float(selected["stop_seconds"]))
        window = {
            "id": analysis_id,
            "start_seconds": float(selected["start_seconds"]),
            "stop_seconds": float(selected["stop_seconds"]),
            "sample_fps": float(selected_sample_fps),
            "device": device,
        }
        windows = match.setdefault("analysis_windows", [])
        best_prefix = f"{match['id']}_best_"
        windows[:] = [item for item in windows if not str(item.get("id", "")).startswith(best_prefix)]
        windows.append(window)

        entry = {
            "id": analysis_id,
            "input": match["video"],
            "start_seconds": window["start_seconds"],
            "stop_seconds": window["stop_seconds"],
            "sample_fps": window["sample_fps"],
            "device": window["device"],
        }
        evaluation["analysis_matches"] = [
            item
            for item in evaluation["analysis_matches"]
            if not str(item.get("id", "")).startswith(best_prefix)
        ]
        evaluation["analysis_matches"].append(entry)
        selected["analysis_id"] = analysis_id

    registry_target.write_text(compact_registry_json(registry), encoding="utf-8")
    write_json(evaluation_target, evaluation)


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    match_ids = selected_match_ids(registry, args.match_id)
    output_dir = args.output_dir.expanduser()
    scan_root = output_dir / "candidates"
    report: dict[str, Any] = {
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "sample_fps": args.sample_fps,
        "matches": [],
    }

    for match_id in match_ids:
        match = get_match(registry, match_id)
        if not match:
            raise SystemExit(f"registry match not found: {match_id}")
        video = match.get("video")
        if not video:
            raise SystemExit(f"registry match has no video: {match_id}")
        probe = probe_video(video)
        if not probe.get("readable") or not probe.get("duration_seconds"):
            raise SystemExit(f"video is not readable: {match_id}: {probe}")

        candidates = []
        for window in candidate_windows(
            float(probe["duration_seconds"]),
            start_seconds=args.start_seconds,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            stop_margin_seconds=args.stop_margin_seconds,
        ):
            start = float(window["start_seconds"])
            stop = float(window["stop_seconds"])
            candidate_id = analysis_id_for_window(match_id, start, stop, prefix="scan")
            csv_path = scan_root / match_id / f"{candidate_id}.csv"
            if args.force or not csv_path.exists():
                run_analysis(args, video, csv_path, start, stop)
            metrics = analysis_metrics(read_rows(csv_path))
            candidates.append(
                {
                    "id": candidate_id,
                    "start_seconds": start,
                    "stop_seconds": stop,
                    "csv": display_path(csv_path),
                    "metrics": metrics,
                }
            )

        ranked = rank_candidates(candidates)
        selected = ranked[0] if ranked else None
        report["matches"].append(
            {
                "match_id": match_id,
                "video": video,
                "duration_seconds": probe.get("duration_seconds"),
                "selected": selected,
                "candidates": ranked,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_selection:
        upsert_selection(report, args.registry, args.evaluation_config, args.selected_sample_fps, args.device)

    json_path = output_dir / "analysis_window_scan.json"
    md_path = output_dir / "analysis_window_scan.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote analysis window scan: {md_path}")
    print(f"wrote analysis window scan json: {json_path}")
    for match in report["matches"]:
        selected = match.get("selected") or {}
        metrics = selected.get("metrics", {})
        print(
            "- {match_id}: {start}-{stop} score={score} count={count:.1%} state={state:.1%} player={player:.1%}".format(
                match_id=match["match_id"],
                start=selected.get("start_seconds", ""),
                stop=selected.get("stop_seconds", ""),
                score=selected.get("score", 0.0),
                count=float(metrics.get("count_ratio", 0.0)),
                state=float(metrics.get("state_ratio", 0.0)),
                player=float(metrics.get("player_ratio", 0.0)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
