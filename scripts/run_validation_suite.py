from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, load_registry, resolve_project_path
from src.validation_suite import validation_analysis_ids, validation_ids

DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Splatoon validation suite for registered local samples.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-analysis", action="store_true", help="Re-run normal gameplay analysis before reporting.")
    parser.add_argument("--run-heatmap-report", action="store_true", default=True, help="Regenerate heatmap report.md files before checking.")
    parser.add_argument("--no-run-heatmap-report", dest="run_heatmap_report", action="store_false")
    parser.add_argument("--strict-model-errors", action="store_true", help="Fail when model_error_report is needs_review.")
    parser.add_argument("--strict-validation-samples", action="store_true", help="Fail when validation_samples report is needs_review.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "validation_suite.json")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for suite-local reports. Defaults to the output path without its suffix.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[object], *, allow_failure: bool = False) -> dict[str, object]:
    printable = " ".join(str(part) for part in command)
    print(f"\n== {name} ==", flush=True)
    print(f"$ {printable}", flush=True)
    result = subprocess.run([str(part) for part in command], cwd=ROOT)
    status = "passed" if result.returncode == 0 else ("needs_review" if allow_failure else "failed")
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return {"name": name, "status": status, "returncode": result.returncode, "command": printable}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote validation suite report: {path}")


def stage_existing_raw_csvs(analysis_ids: list[str], evaluation_dir: Path) -> list[str]:
    staged: list[str] = []
    source_root = ROOT / "outputs" / "evaluation"
    for match_id in analysis_ids:
        source = source_root / match_id / "raw.csv"
        if not source.exists():
            continue
        target = evaluation_dir / match_id / "raw.csv"
        if source.resolve() == target.resolve():
            staged.append(match_id)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        staged.append(match_id)
    if staged:
        print("staged existing raw CSVs: " + ", ".join(staged), flush=True)
    return staged


def main() -> int:
    args = parse_args()
    python = resolve_project_path(args.python) or args.python.expanduser()
    output_path = args.output.expanduser()
    work_dir = args.work_dir.expanduser() if args.work_dir else output_path.with_suffix("")
    evaluation_dir = work_dir / "evaluation"
    heatmap_comparison_md = work_dir / "heatmap_comparison.md"
    heatmap_comparison_json = work_dir / "heatmap_comparison.json"
    model_registry_md = work_dir / "model_registry.md"
    model_registry_json = work_dir / "model_registry.json"
    model_training_plan_md = work_dir / "model_training_plan.md"
    model_training_plan_json = work_dir / "model_training_plan.json"
    model_training_datasets_md = work_dir / "model_training_datasets.md"
    model_training_datasets_json = work_dir / "model_training_datasets.json"
    model_error_md = work_dir / "model_error_report_smoothed.md"
    model_error_json = work_dir / "model_error_report_smoothed.json"
    validation_samples_md = work_dir / "validation_samples.md"
    validation_samples_json = work_dir / "validation_samples.json"
    registry = load_registry(args.registry)
    ids = validation_ids(registry)
    analysis_ids = validation_analysis_ids(registry)
    if not ids:
        raise SystemExit("no validation sample ids found in registry")
    staged_raw_csvs = [] if args.run_analysis else stage_existing_raw_csvs(analysis_ids, evaluation_dir)

    steps: list[dict[str, object]] = []
    steps.append(run_step("unit tests", [python, "-m", "unittest", "discover", "-s", "tests", "-q"]))
    steps.append(run_step("registry validation", [python, "scripts/validate_data_registry.py", "--strict"]))
    steps.append(run_step("heatmap config validation", [python, "scripts/validate_heatmap_configs.py", "--strict"]))
    steps.append(
        run_step(
            "model registry",
            [
                python,
                "scripts/report_model_registry.py",
                "--output",
                model_registry_md,
                "--json-output",
                model_registry_json,
                "--strict",
            ],
        )
    )
    steps.append(
        run_step(
            "model training plan",
            [
                python,
                "scripts/plan_model_training.py",
                "--output",
                model_training_plan_md,
                "--json-output",
                model_training_plan_json,
            ],
        )
    )
    steps.append(
        run_step(
            "model training dataset dry run",
            [
                python,
                "scripts/validate_model_training_datasets.py",
                "--output",
                model_training_datasets_md,
                "--json-output",
                model_training_datasets_json,
            ],
        )
    )

    evaluation_cmd: list[object] = [python, "scripts/evaluate_matches.py", "--strict", "--output-dir", evaluation_dir]
    if args.run_analysis:
        evaluation_cmd.append("--run-analysis")
    if args.run_heatmap_report:
        evaluation_cmd.append("--run-heatmap-report")
    for item in ids:
        evaluation_cmd.extend(["--only", item])
    steps.append(run_step("validation sample evaluation", evaluation_cmd))

    steps.append(
        run_step(
            "heatmap comparison",
            [
                python,
                "scripts/report_heatmap_comparison.py",
                "--output",
                heatmap_comparison_md,
                "--json-output",
                heatmap_comparison_json,
                "--strict",
            ],
        )
    )

    model_cmd: list[object] = [
        python,
        "scripts/report_model_errors.py",
        "--evaluation-results",
        evaluation_dir / "evaluation_results.json",
        "--smoothed",
        "--output",
        model_error_md,
        "--json-output",
        model_error_json,
    ]
    for item in analysis_ids:
        model_cmd.extend(["--only-id", item])
    if args.strict_model_errors:
        model_cmd.append("--strict")
    steps.append(run_step("model error report", model_cmd, allow_failure=not args.strict_model_errors))

    sample_cmd: list[object] = [
        python,
        "scripts/report_validation_samples.py",
        "--evaluation-results",
        evaluation_dir / "evaluation_results.json",
        "--heatmap-comparison",
        heatmap_comparison_json,
        "--model-error-report",
        model_error_json,
        "--output",
        validation_samples_md,
        "--json-output",
        validation_samples_json,
    ]
    if args.strict_validation_samples:
        sample_cmd.append("--strict")
    steps.append(run_step("validation sample report", sample_cmd, allow_failure=not args.strict_validation_samples))

    overall = "passed" if all(step["status"] in ("passed", "needs_review") for step in steps) else "failed"
    payload = {
        "status": overall,
        "work_dir": str(work_dir),
        "validation_ids": ids,
        "staged_raw_csvs": staged_raw_csvs,
        "steps": steps,
    }
    write_json(output_path, payload)
    print(f"validation suite status: {overall}")
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
