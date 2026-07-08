from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.core.paths import ROOT
from src.model_training_plan import build_model_training_plan


def normalized_command_args(command: str) -> list[str]:
    args = shlex.split(command)
    if args and args[0] in {"python", "python3"}:
        args[0] = sys.executable
    return args


def build_training_launch_plan(config: dict[str, Any], *, target_id: str) -> dict[str, Any]:
    training_plan = build_model_training_plan(config, target_ids=[target_id])
    target = training_plan["targets"][0] if training_plan.get("targets") else {}
    command = str(target.get("candidate_command", ""))
    command_args = normalized_command_args(command) if command else []
    blockers: list[str] = []
    warnings: list[str] = []

    if training_plan.get("missing_target_ids"):
        blockers.append(f"unknown training target: {target_id}")
    if not command:
        blockers.append("candidate_command is not configured")
    if target.get("status") == "needs_data":
        blockers.extend(str(path) for path in target.get("missing_paths", []))
    dataset = target.get("dataset_spec", {})
    if dataset.get("status") == "needs_data":
        blockers.extend(str(item) for item in dataset.get("blockers", []))
    if dataset.get("status") == "needs_review":
        warnings.extend(str(item) for item in dataset.get("warnings", []))

    status = "ready"
    if blockers:
        status = "needs_data" if target.get("status") == "needs_data" else "failed"
    if training_plan.get("missing_target_ids"):
        status = "failed"

    return {
        "schema_version": 1,
        "status": status,
        "target_id": target_id,
        "command": command,
        "command_args": command_args,
        "target": target,
        "training_plan_status": training_plan.get("status"),
        "blockers": blockers,
        "warnings": warnings,
        "execution": {"status": "not_run", "returncode": None},
    }


def execute_training_launch_plan(plan: dict[str, Any], *, cwd: Path = ROOT) -> dict[str, Any]:
    if plan.get("status") != "ready":
        raise ValueError(f"training launch plan is not ready: {plan.get('status')}")
    command_args = list(plan.get("command_args", []))
    if not command_args:
        raise ValueError("training launch plan has no command arguments")
    result = subprocess.run(command_args, cwd=cwd, check=False)
    report = dict(plan)
    report["execution"] = {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
    }
    report["status"] = "completed" if result.returncode == 0 else "failed"
    return report


def render_markdown(report: dict[str, Any]) -> str:
    target = report.get("target", {})
    execution = report.get("execution", {})
    lines = [
        "# Model Training Launch Plan",
        "",
        f"- status: `{report.get('status')}`",
        f"- target_id: `{report.get('target_id', '')}`",
        f"- area: `{target.get('area', '')}`",
        f"- model_id: `{target.get('model_id', '')}`",
        f"- dataset_status: `{target.get('dataset_status', '')}`",
        f"- candidate_output_dir: `{target.get('candidate_output_dir', '')}`",
        f"- baseline_model: `{target.get('baseline_model', '')}`",
        f"- promotion_gate: `{target.get('promotion_gate', '')}`",
        f"- execution_status: `{execution.get('status', '')}`",
        f"- returncode: {execution.get('returncode')}",
        "",
        "## Command",
        "",
        f"`{report.get('command', '')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers", [])
    lines.extend([f"- {item}" for item in blockers] or ["- -"])
    warnings = report.get("warnings", [])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- -"])
    lines.append("")
    return "\n".join(lines)
