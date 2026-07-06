from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"

ACTIVE_COMPILE_TARGETS = ("src", "scripts", "yolov5")


def project_env() -> dict[str, str]:
    cache_root = ROOT / ".cache"
    paths = {
        "UV_CACHE_DIR": cache_root / "uv",
        "PIP_CACHE_DIR": cache_root / "pip",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root,
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({key: str(value) for key, value in paths.items()})
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    env.setdefault("YOLOv5_AUTOINSTALL", "false")
    return env


def run_step(name: str, command: Sequence[object], env: dict[str, str]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n== {name} ==", flush=True)
    print(f"$ {printable}", flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, env=env, check=True)


def existing(paths: Iterable[Path]) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in paths if path.exists()]


def compile_targets() -> list[str]:
    return existing(ROOT / target for target in ACTIVE_COMPILE_TARGETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Splatoon 3 analysis health checks.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for temporary CSVs and YOLOv5 smoke outputs. Defaults to a /tmp directory.",
    )
    parser.add_argument("--video-smoke", action="store_true", help="Also run a 40-frame match_1 video smoke test.")
    parser.add_argument("--long-mps", action="store_true", help="Also run the 10-150s match_1 MPS baseline.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"], help="Device for --video-smoke.")
    parser.add_argument("--match-video", default="footages/match_1.mp4", help="Video used by smoke checks.")
    parser.add_argument("--skip-yolov5-detect", action="store_true", help="Skip the raw YOLOv5 detect.py smoke test.")
    parser.add_argument("--tooling", action="store_true", help="Also check inventory, report, and training-plan helper scripts.")
    parser.add_argument("--evaluation", action="store_true", help="Also run the fixed match evaluation suite.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PYTHON.exists():
        raise FileNotFoundError(f"Missing virtualenv Python: {PYTHON}")

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="splatoon3_check_"))
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    env = project_env()
    targets = compile_targets()
    if not targets:
        raise RuntimeError("No compile targets found.")

    run_step("compile python files", [PYTHON, "-m", "compileall", "-q", *targets], env)
    run_step(
        "load model names",
        [
            PYTHON,
            "-m",
            "src.run_analysis",
            "--input",
            "sample/battle.png",
            "--output",
            work_dir / "model_names.csv",
            "--device",
            "cpu",
            "--max-frames",
            "1",
            "--list-model-names",
        ],
        env,
    )
    run_step(
        "sample image analysis",
        [
            PYTHON,
            "-m",
            "src.run_analysis",
            "--input",
            "sample/battle.png",
            "--output",
            work_dir / "sample.csv",
            "--device",
            "cpu",
            "--max-frames",
            "1",
        ],
        env,
    )
    run_step("summarize sample csv", [PYTHON, "scripts/summarize_csv.py", work_dir / "sample.csv"], env)
    run_step(
        "protocol 33-column check",
        [
            PYTHON,
            "-c",
            (
                "from src.protocol import create_game_state; "
                "row=['0']*33; row[0]='12.3'; row[27]='True'; "
                "msg=create_game_state(row); "
                "assert msg.type == 'game_state'; "
                "assert msg.payload.elapsed_time == '12.3'; "
                "assert len(msg.payload.player_states) == 8; "
                "print(msg.type, msg.payload.elapsed_time, msg.payload.player_detected)"
            ),
        ],
        env,
    )

    if not args.skip_yolov5_detect:
        run_step(
            "raw yolov5 detect smoke",
            [
                PYTHON,
                "yolov5/detect.py",
                "--weights",
                "models/the_model.pt",
                "--source",
                "sample/battle.png",
                "--project",
                work_dir,
                "--name",
                "yolov5-detect",
                "--exist-ok",
                "--device",
                "cpu",
                "--nosave",
            ],
            env,
        )

    if args.video_smoke:
        video_csv = work_dir / "match1_smoke.csv"
        run_step(
            "match_1 40-frame video smoke",
            [
                PYTHON,
                "-m",
                "src.run_analysis",
                "--input",
                args.match_video,
                "--output",
                video_csv,
                "--device",
                args.device,
                "--start-seconds",
                "10",
                "--sample-fps",
                "5",
                "--max-frames",
                "40",
            ],
            env,
        )
        run_step("summarize video smoke csv", [PYTHON, "scripts/summarize_csv.py", video_csv], env)

    if args.tooling:
        run_step(
            "inventory helper",
            [PYTHON, "scripts/inventory_project.py", "--output", work_dir / "project_inventory.json"],
            env,
        )
        run_step(
            "csv report helper",
            [PYTHON, "scripts/report_csv.py", work_dir / "sample.csv", "--output", work_dir / "sample.report.md"],
            env,
        )
        annotation_dir = work_dir / "annotation_samples"
        run_step(
            "heatmap annotation package helper",
            [
                PYTHON,
                "scripts/export_heatmap_annotation_package.py",
                "--output-dir",
                annotation_dir,
                "--match-id",
                "match_9",
                "--frames-per-match",
                "1",
            ],
            env,
        )
        run_step(
            "heatmap annotation evaluation helper",
            [
                PYTHON,
                "scripts/evaluate_heatmap_annotations.py",
                annotation_dir / "annotation_template.csv",
                "--output",
                annotation_dir / "evaluation.json",
                "--report",
                annotation_dir / "evaluation.md",
            ],
            env,
        )
        run_step(
            "heatmap anomaly export helper",
            [
                PYTHON,
                "scripts/export_heatmap_anomalies.py",
                "--output-dir",
                work_dir / "heatmap_anomalies",
                "--match-id",
                "match_9",
                "--max-items-per-match",
                "4",
            ],
            env,
        )
        run_step("weapon training plan helper", [PYTHON, "scripts/plan_weapon_training.py"], env)
        run_step("weapon training strict plan helper", [PYTHON, "scripts/plan_weapon_training.py", "--strict"], env)
        run_step(
            "weapon classifier training dry run",
            [
                PYTHON,
                "scripts/train_weapon_classifier.py",
                "--dry-run",
                "--max-samples-per-class",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--output",
                work_dir / "weapon_model.pth",
                "--metrics",
                work_dir / "weapon_training_metrics.json",
            ],
            env,
        )
        run_step(
            "weapon synthetic dataset dry run",
            [
                PYTHON,
                "scripts/generate_weapon_dataset.py",
                "--dry-run",
                "--images-per-class",
                "1",
                "--output-dir",
                work_dir / "generated_weapon_dataset",
            ],
            env,
        )
        run_step(
            "match intake dry run",
            [
                PYTHON,
                "scripts/intake_match.py",
                "--match-id",
                "match_intake_smoke",
                "--video",
                args.match_video,
                "--analysis-id",
                "match_intake_smoke_10_20",
                "--start-seconds",
                "10",
                "--stop-seconds",
                "20",
                "--sample-fps",
                "5",
                "--device",
                args.device,
                "--dry-run",
                "--report",
                work_dir / "match_intake.md",
            ],
            env,
        )
        run_step(
            "model quality overview helper",
            [
                PYTHON,
                "scripts/report_model_quality.py",
                "--output",
                work_dir / "model_quality.md",
                "--json-output",
                work_dir / "model_quality.json",
            ],
            env,
        )
        run_step(
            "model error report helper",
            [
                PYTHON,
                "scripts/report_model_errors.py",
                "--csv",
                work_dir / "sample.csv",
                "--output",
                work_dir / "model_errors.md",
                "--json-output",
                work_dir / "model_errors.json",
            ],
            env,
        )

    if args.evaluation:
        run_step(
            "data registry validation",
            [
                PYTHON,
                "scripts/validate_data_registry.py",
                "--output",
                work_dir / "data_registry.json",
                "--report",
                work_dir / "data_registry.md",
                "--strict",
            ],
            env,
        )
        run_step(
            "fixed match evaluation",
            [
                PYTHON,
                "scripts/evaluate_matches.py",
                "--output-dir",
                work_dir / "evaluation",
                "--run-analysis",
                "--strict",
            ],
            env,
        )

    if args.long_mps:
        long_csv = work_dir / "match1_10_150_mps.csv"
        run_step(
            "match_1 10-150s MPS baseline",
            [
                PYTHON,
                "-m",
                "src.run_analysis",
                "--input",
                args.match_video,
                "--output",
                long_csv,
                "--device",
                "mps",
                "--start-seconds",
                "10",
                "--stop-seconds",
                "150",
                "--sample-fps",
                "5",
            ],
            env,
        )
        run_step("summarize MPS baseline csv", [PYTHON, "scripts/summarize_csv.py", long_csv], env)

    print(f"\nAll selected checks passed. Temporary outputs: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
