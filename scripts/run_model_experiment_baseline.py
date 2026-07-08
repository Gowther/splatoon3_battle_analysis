from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import project_path
from src.model_experiment_baseline import baseline_paths, build_baseline_steps, summarize_steps
from src.report_io import write_json_report


DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed baseline package before detector/OCR/model experiments.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "model_experiment_baseline")
    parser.add_argument("--run-validation-suite", action="store_true", help="Regenerate validation outputs inside the baseline package.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any baseline step fails.")
    return parser.parse_args()


def run_step(name: str, command: list[object], *, allow_failure: bool = False) -> dict[str, object]:
    printable = " ".join(str(part) for part in command)
    print(f"\n== {name} ==", flush=True)
    print(f"$ {printable}", flush=True)
    result = subprocess.run([str(part) for part in command], cwd=ROOT)
    status = "passed" if result.returncode == 0 else ("needs_review" if allow_failure else "failed")
    return {"name": name, "status": status, "returncode": result.returncode, "command": printable}


def main() -> int:
    args = parse_args()
    python = project_path(args.python)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = build_baseline_steps(python=python, output_dir=output_dir, run_validation_suite=args.run_validation_suite)
    results: list[dict[str, object]] = []
    for step in steps:
        result = run_step(step.name, list(step.command), allow_failure=step.allow_failure)
        results.append(result)
        if result["status"] == "failed":
            break
    summary = summarize_steps(results)
    paths = baseline_paths(output_dir, validation_suite_ran=args.run_validation_suite)
    payload = {
        **summary,
        "output_dir": str(output_dir),
        "run_validation_suite": args.run_validation_suite,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "steps": results,
    }
    write_json_report(paths["summary_json"], payload)
    print(f"model experiment baseline status: {payload['status']}")
    return 1 if args.strict and payload["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
